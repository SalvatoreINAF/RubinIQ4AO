# This file is part of summit_extras.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = [
    "addColorbarToAxes",
    "makeTableFromSourceCatalogs",
    "makeFigureAndAxes",
    "extendTable",
    "makeFocalPlanePlot",
    "makeEquatorialPlot",
    "makeAzElPlot",
]


from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

from lsst.afw import cameraGeom
from lsst.afw.cameraGeom import FOCAL_PLANE
from lsst.afw.geom.ellipses import Quadrupole
from lsst.geom import LinearTransform, radians, Point2D

import astropy.units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time

from astropy.table import Table

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt
    from astropy.table import Table
    from matplotlib.colorbar import Colorbar

    from lsst.afw.cameraGeom import Camera
    from lsst.afw.image import VisitInfo
    from lsst.afw.table import SourceCatalog

def pseudo_parallactic_angle(
    ra: float,
    dec: float,
    mjd: float,
    lon: float = -70.7494,
    lat: float = -30.2444,
    height: float = 2650.0,
    pressure: float = 750.0,
    temperature: float = 11.5,
    relative_humidity: float = 0.4,
    obswl: float = 1.0,
):
    """Compute the pseudo parallactic angle.

    The (traditional) parallactic angle is the angle zenith - coord - NCP
    where NCP is the true-of-date north celestial pole.  This function instead
    computes zenith - coord - NCP_ICRF where NCP_ICRF is the north celestial
    pole in the International Celestial Reference Frame.

    Parameters
    ----------
    ra, dec : float
        ICRF coordinates in degrees.
    mjd : float
        Modified Julian Date.
    latitude, longitude : float
        Geodetic coordinates of observer in degrees.
    height : float
        Height of observer above reference ellipsoid in meters.
    pressure : float
        Atmospheric pressure in millibars.
    temperature : float
        Atmospheric temperature in degrees Celsius.
    relative_humidity : float
    obswl : float
        Observation wavelength in microns.

    Returns
    -------
    ppa : float
        The pseudo parallactic angle in degrees.
    """
    obstime = Time(mjd, format="mjd", scale="tai")
    location = EarthLocation.from_geodetic(
        lon=lon * u.deg,
        lat=lat * u.deg,
        height=height * u.m,
        ellipsoid="WGS84",  # For concreteness
    )

    coord_kwargs = dict(
        obstime=obstime,
        location=location,
        pressure=pressure * u.mbar,
        temperature=temperature * u.deg_C,
        relative_humidity=relative_humidity,
        obswl=obswl * u.micron,
    )

    coord = SkyCoord(ra * u.deg, dec * u.deg, **coord_kwargs)

    towards_zenith = SkyCoord(
        alt=coord.altaz.alt + 10 * u.arcsec,
        az=coord.altaz.az,
        frame=AltAz,
        **coord_kwargs
    )

    towards_north = SkyCoord(
        ra=coord.icrs.ra, dec=coord.icrs.dec + 10 * u.arcsec, **coord_kwargs
    )

    ppa = coord.position_angle(towards_zenith) - coord.position_angle(towards_north)
    return ppa.wrap_at(180 * u.deg).deg
    
def rsp_to_rtp(rotSkyPos: float, ra: float, dec: float, mjd: float, **kwargs: dict):
    """Convert RotTelPos -> RotSkyPos.

    Parameters
    ----------
    rotSkyPos : float
        Sky rotation angle in degrees.
    ra, dec : float
        ICRF coordinates in degrees.
    mjd : float
        Modified Julian Date.
    **kwargs : dict
        Other keyword arguments to pass to pseudo_parallactic_angle.  Defaults
        are generally appropriate for Rubin Observatory.

    Returns
    -------
    rsp : float
        RotSkyPos in degrees.
    """
    q = pseudo_parallactic_angle(ra, dec, mjd, **kwargs)
    return Angle((270 - rotSkyPos + q)*u.deg).wrap_at(180 * u.deg)

def pixel_to_camera(x, y, det):
    """
    Parameters
    ----------
    x, y : array
        Pixel coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.
    Returns
    -------
    cam_x, cam_y : array
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    cam_x, cam_y = tx.getMapping().applyForward(np.vstack((x, y)))
    return cam_x.ravel(), cam_y.ravel()
    
def pixel_to_camera_angle(x, y, det):
    """
    Parameters
    ----------
    x, y : array
        Pixel coordinates.lsst afw.detectordetectordetectordetector
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.
    Returns
    -------
    cam_x, cam_y : array
        Focal plane position in degrees in DVCS
        See https://lse-349.lsst.io/
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FIELD_ANGLE)
    cam_x, cam_y = tx.getMapping().applyForward(np.vstack((x, y)))
    return np.degrees(cam_x.ravel()), np.degrees(cam_y.ravel())
    
def randomRows(table: Table, maxRows: int) -> Table:
    """Select a random subset of rows from the given table.

    Parameters
    ----------
    table : `astropy.table.Table`
        The table containing the data to be plotted.
    maxRows : `int`
        The maximum number of rows to select.

    Returns
    -------
    table : `astropy.table.Table`
        The table containing the randomly selected subset of rows.
    """
    n = len(table)
    if n > maxRows:
        rng = np.random.default_rng()
        indices = rng.choice(n, maxRows, replace=False)
        table = table[indices]
    return table


def addColorbarToAxes(mappable: plt.Axes) -> Colorbar:
    """Add a colorbar to the given axes.

    Parameters
    ----------
    mappable : `matplotlib.cm.ScalarMappable`
        The mappable object to which the colorbar will be added.

    Returns
    -------
    cbar : `matplotlib.colorbar.Colorbar`
        The colorbar object that was added to the axes.
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    return cbar


def makeTableFromSourceCatalogs(icSrcs: dict[int, SourceCatalog], visitInfo: VisitInfo) -> Table:
    """Extract the shapes from the source catalogs into an astropy table.

    The shapes of the PSF candidates are extracted from the source catalogs and
    transformed into the required coordinate systems for plotting either focal
    plane coordinates, az/el coordinates, or equatorial coordinates.

    Parameters
    ----------
    icSrcs : `dict` [`int`, `lsst.afw.table.SourceCatalog`]
        A dictionary of source catalogs, keyed by the detector numbers.
    visitInfo : `lsst.afw.image.VisitInfo`
        The visit information for a representative visit.

    Returns
    -------
    table : `astropy.table.Table`
        The table containing the data from the source catalogs.
    """
    tables = []

    for detectorNum, icSrc in icSrcs.items():
        icSrc = icSrc.asAstropy()
        icSrc = icSrc[icSrc["calib_psf_candidate"]]
        icSrc["detector"] = detectorNum
        tables.append(icSrc)

    table = vstack(tables)

    # Add shape columns
    table["Ixx"] = table["slot_Shape_xx"] * (0.2) ** 2
    table["Ixy"] = table["slot_Shape_xy"] * (0.2) ** 2
    table["Iyy"] = table["slot_Shape_yy"] * (0.2) ** 2
    table["T"] = table["Ixx"] + table["Iyy"]
    table["e1"] = (table["Ixx"] - table["Iyy"]) / table["T"]
    table["e2"] = 2 * table["Ixy"] / table["T"]
    table["e"] = np.hypot(table["e1"], table["e2"])
    table["x"] = table["base_FPPosition_x"]
    table["y"] = table["base_FPPosition_y"]

    table.meta["rotTelPos"] = (
        visitInfo.boresightParAngle - visitInfo.boresightRotAngle - (np.pi / 2 * radians)
    ).asRadians()
    table.meta["rotSkyPos"] = visitInfo.boresightRotAngle.asRadians()

    rtp = table.meta["rotTelPos"]
    srtp, crtp = np.sin(rtp), np.cos(rtp)
    aaRot = np.array([[crtp, srtp], [-srtp, crtp]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
    table = extendTable(table, aaRot, "aa")
    table.meta["aaRot"] = aaRot

    rsp = table.meta["rotSkyPos"]
    srsp, crsp = np.sin(rsp), np.cos(rsp)
    nwRot = np.array([[crsp, -srsp], [srsp, crsp]])
    table = extendTable(table, nwRot, "nw")
    table.meta["nwRot"] = nwRot

    return table

def makeTableFromGrid(n_grid, calexp, psf, visitInfo: VisitInfo) -> Table:
    """Extract the shapes from a grid into an astropy table.

    The shapes of the PSF candidates are extracted from the source catalogs and
    transformed into the required coordinate systems for plotting either focal
    plane coordinates, az/el coordinates, or equatorial coordinates.

    Parameters
    ----------
    visitInfo : `lsst.afw.image.VisitInfo`
        The visit information for a representative visit.

    Returns
    -------
    table : `astropy.table.Table`
        The table containing the data from the source catalogs.
    """

    # # From wcs (come per griglia mio notebook)
    # wcs = calexp.getWcs()
    # rotskypos = (calexp.info.getVisitInfo().getBoresightRotAngle()).asDegrees()
    # rottelpos = rsp_to_rtp(rotskypos, \
    #         (calexp.info.getVisitInfo().getBoresightRaDec())[0].asDegrees(), \
    #         (calexp.info.getVisitInfo().getBoresightRaDec())[1].asDegrees(), \
    #         calexp.info.getVisitInfo().getDate().toAstropy()).deg
    # rottelpos_radians = np.radians(rottelpos)
    # crtp = np.cos(rottelpos_radians)
    # srtp = np.sin(rottelpos_radians)

    # From icsrc
    rottelpos_radians = (visitInfo.boresightParAngle - visitInfo.boresightRotAngle - (np.pi / 2 * radians)).asRadians()
    rotskypos_radians = visitInfo.boresightRotAngle.asRadians()
    crtp = np.cos(rottelpos_radians)
    srtp = np.sin(rottelpos_radians)

    # print('Rotskypos [rad]:', np.radians(rotskypos))
    # print('Rottelpos [rad]:', rottelpos_radians)
    # print('Rotskypos2 [rad]:', rotskypos2_radians)
    # print('Rottelpos2 [rad]:', rottelpos2_radians + 2*np.pi)
        
    # aa sta per Alt-Azimuth, trasformazione da codice di Josh:
    # https://github.com/lsst-sitcom/summit_extras/blob/main/python/lsst/summit/extras/plotting/psfPlotting.py
    aaRot = np.array([[crtp, srtp], [-srtp, crtp]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
    transform_for_ell = LinearTransform(aaRot)
    # print(aaRot)

    grid_separation_x = calexp.getDimensions()[0] / n_grid
    grid_separation_y = calexp.getDimensions()[1] / n_grid
    x_array = np.arange(n_grid)*grid_separation_x + grid_separation_x/2.
    y_array = np.arange(n_grid)*grid_separation_y + grid_separation_y/2.
    xx, yy = np.meshgrid(x_array, y_array)
    xx_for_zip = xx.flatten()
    yy_for_zip = yy.flatten()
    xxshape = n_grid*n_grid

    print('sss2')
    
    size = []
    i_xx = []
    i_yy = []
    i_xy = []
    points = []

    xx_star_dvcs = []
    yy_star_dvcs = []
    xx_rot_star_dvcs = []
    yy_rot_star_dvcs = []
    ra_star_dvcs = []
    dec_star_dvcs = []

    cam_x_mm_all = []
    cam_y_mm_all = []

    det = calexp.getDetector()
    wcs = calexp.getWcs()
    
    for x, y in zip(xx_for_zip, yy_for_zip):
        point = Point2D(x, y)        
        coo = wcs.pixelToSky(x, y)
        cam_x, cam_y = pixel_to_camera_angle(point[0], point[1], det)
        cam_x_mm, cam_y_mm = pixel_to_camera(point[0], point[1], det)
        x0, y0 = np.asarray(cam_x[0]), np.asarray(cam_y[0])
        # Rotazioni Josh
        xx_rot = aaRot[0, 0] * x0 + aaRot[0, 1] * y0
        yy_rot = aaRot[1, 0] * x0 + aaRot[1, 1] * y0

        cam_x_mm_all.append(cam_x_mm)
        cam_y_mm_all.append(cam_y_mm)
        
        shape = psf.computeShape(point)
        size.append(shape.getTraceRadius())
        i_xx.append(shape.getIxx())
        i_yy.append(shape.getIyy())
        i_xy.append(shape.getIxy())
        points.append(point)

        xx_star_dvcs.append(cam_x[0])
        yy_star_dvcs.append(cam_y[0])
        xx_rot_star_dvcs.append(xx_rot)
        yy_rot_star_dvcs.append(yy_rot)

        ra_star_dvcs.append(coo[0].asDegrees())
        dec_star_dvcs.append(coo[1].asDegrees())
        
    size = np.reshape(size, xxshape)
    i_xx = np.reshape(i_xx, xxshape)
    i_yy = np.reshape(i_yy, xxshape)
    i_xy = np.reshape(i_xy, xxshape)

    table_moments = {'Ixx': i_xx, 'Iyy': i_yy, 'Ixy': i_xy}
    print('sss3')

    # ------------Transform moments into ellipticities------------
    theta = np.arctan2(2. * i_xy, i_xx - i_yy) / 2.

    e1 = (i_xx - i_yy) / (i_xx + i_yy)
    e2 = (2. * i_xy) / (i_xx + i_yy)
    
    theta_alternate = np.arctan2(e2, e1) / 2.
    assert np.allclose(theta, theta_alternate)

    e_star = np.sqrt(e1**2 + e2**2)
    ex = e_star * np.cos(theta)
    ey = e_star * np.sin(theta)

    ey_star_dvcs = ey
    ex_star_dvcs = ex

    # Rotazioni Josh (Quadrupole)
    rot_shapes = []
    for i_xx1, i_yy1, i_xy1 in zip(i_xx, i_yy, i_xy):
        shape = Quadrupole(i_xx1, i_yy1, i_xy1)
        rot_shapes.append(shape.transform(transform_for_ell))
        
    aaIxx = np.asarray([sh.getIxx() for sh in rot_shapes])
    aaIyy = np.asarray([sh.getIyy() for sh in rot_shapes])
    aaIxy = np.asarray([sh.getIxy() for sh in rot_shapes])
    e1_rot_star_dvcs = (aaIxx - aaIyy) / (aaIxx + aaIyy)
    e2_rot_star_dvcs = 2 * aaIxy / (aaIxx + aaIyy)    
    
    theta_josh = np.arctan2(e2_rot_star_dvcs, e1_rot_star_dvcs) / 2.
    e_star = np.sqrt(e1_rot_star_dvcs**2 + e2_rot_star_dvcs**2)
    ex_rot_star_dvcs = e_star * np.cos(theta_josh)
    ey_rot_star_dvcs = e_star * np.sin(theta_josh)
    
    theta_star_dvcs = np.arctan2(ey, ex)

    table = Table()
    # Add shape columns
    table["Ixx"] = i_xx
    table["Ixy"] = i_xy
    table["Iyy"] = i_yy
    table["T"] = i_xx + i_yy
    table["e1"] = (table["Ixx"] - table["Iyy"]) / table["T"]
    table["e2"] = 2 * table["Ixy"] / table["T"]
    table["e"] = np.hypot(table["e1"], table["e2"])

    table["x"] = cam_x_mm_all
    table["y"] = cam_y_mm_all

    table.meta["rotTelPos"] = (
        visitInfo.boresightParAngle - visitInfo.boresightRotAngle - (np.pi / 2 * radians)
    ).asRadians()
    table.meta["rotSkyPos"] = visitInfo.boresightRotAngle.asRadians()

    table = extendTable(table, aaRot, "aa")
    table.meta["aaRot"] = aaRot

    rsp = table.meta["rotSkyPos"]
    srsp, crsp = np.sin(rsp), np.cos(rsp)
    nwRot = np.array([[crsp, -srsp], [srsp, crsp]])
    table = extendTable(table, nwRot, "nw")
    table.meta["nwRot"] = nwRot

    return table

def extendTable(table: Table, rot: npt.NDArray[np.float_], prefix: str) -> Table:
    """Extend the given table with additional columns for the rotated shapes.

    Parameters
    ----------
    table : `astropy.table.Table`
        The input table containing the original shapes.
    rot : `numpy.ndarray`
        The rotation matrix used to rotate the shapes.
    prefix : `str`
        The prefix to be added to the column names of the rotated shapes.

    Returns
    -------
    table : `astropy.table.Table`
        The extended table with additional columns representing the rotated
        shapes.
    """
    transform = LinearTransform(rot)
    rot_shapes = []
    for row in table:
        shape = Quadrupole(row["Ixx"], row["Iyy"], row["Ixy"])
        rot_shape = shape.transform(transform)
        rot_shapes.append(rot_shape)
    table[prefix + "_Ixx"] = [sh.getIxx() for sh in rot_shapes]
    table[prefix + "_Iyy"] = [sh.getIyy() for sh in rot_shapes]
    table[prefix + "_Ixy"] = [sh.getIxy() for sh in rot_shapes]
    table[prefix + "_e1"] = (table[prefix + "_Ixx"] - table[prefix + "_Iyy"]) / table["T"]
    table[prefix + "_e2"] = 2 * table[prefix + "_Ixy"] / table["T"]
    table[prefix + "_x"] = rot[0, 0] * table["x"] + rot[0, 1] * table["y"]
    table[prefix + "_y"] = rot[1, 0] * table["x"] + rot[1, 1] * table["y"]
    return table


def makeFigureAndAxes() -> tuple[plt.Figure, Any]:
    """Create a figure and axes for plotting.

    Returns
    -------
    fig : `matplotlib.figure.Figure`:
        The created figure.
    axes : `numpy.ndarray`
        The created axes.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    return fig, axes


def makeFocalPlanePlot(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPoints: int = 1000,
    saveAs: str = "",
):
    """Plot the PSFs in focal plane (detector) coordinates i.e. the raw shapes.

    Top left:
        A scatter plot of the T values in square arcseconds.
    Top right:
        A quiver plot of e1 and e2
    Bottom left:
        A scatter plot of e1
    Bottom right:
        A scatter plot of e2

    This function plots the data from the ``table`` on the provided ``fig`` and
    ``axes`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `numpy.ndarray`
        The table containing the data to be plotted.
    camera : `list`
        The list of camera detector objects.
    maxPoints : `int`, optional
        The maximum number of points to plot. If the number of points in the
        table is greater than this value, a random subset of points will be
        plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    table = randomRows(table, maxPoints)

    cbar = addColorbarToAxes(axes[0, 0].scatter(table["x"], table["y"], c=table["T"], s=5))
    cbar.set_label("T [arcsec$^2$]")

    emax = np.quantile(np.abs(np.concatenate([table["e1"], table["e2"]])), 0.98)
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(table["x"], table["y"], c=table["e1"], vmin=-emax, vmax=emax, cmap="bwr", s=5)
    )
    cbar.set_label("e1")

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(table["x"], table["y"], c=table["e2"], vmin=-emax, vmax=emax, cmap="bwr", s=5)
    )
    cbar.set_label("e2")

    Q = axes[0, 1].quiver(
        table["x"],
        table["y"],
        table["e"] * np.cos(0.5 * np.arctan2(table["e2"], table["e1"])),
        table["e"] * np.sin(0.5 * np.arctan2(table["e2"], table["e1"])),
        headlength=0,
        headaxislength=0,
        scale=1,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.05, label="0.05", labelpos="S")

    for ax in axes.ravel():
        ax.set_xlabel("Focal Plane x [mm]")
        ax.set_ylabel("Focal Plane y [mm]")
        ax.set_aspect("equal")

    # Plot camera detector outlines
    for det in camera:
        xs = []
        ys = []
        for corner in det.getCorners(FOCAL_PLANE):
            xs.append(corner.x)
            ys.append(corner.y)
        xs.append(xs[0])
        ys.append(ys[0])
        xs = np.array(xs)
        ys = np.array(ys)
        for ax in axes.ravel():
            ax.plot(xs, ys, c="k", lw=1, alpha=0.3)

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)


def makeEquatorialPlot(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPoints: int = 1000,
    saveAs: str = "",
):
    """Plot the PSFs on the focal plane, rotated to equatorial coordinates.

    Top left:
        A scatter plot of the T values in square arcseconds.
    Top right:
        A quiver plot of e1 and e2
    Bottom left:
        A scatter plot of e1
    Bottom right:
        A scatter plot of e2

    This function plots the data from the ``table`` on the provided ``fig`` and
    ``axes`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `numpy.ndarray`
        The table containing the data to be plotted.
    camera : `list`
        The list of camera detector objects.
    maxPoints : `int`, optional
        The maximum number of points to plot. If the number of points in the
        table is greater than this value, a random subset of points will be
        plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    table = randomRows(table, maxPoints)

    cbar = addColorbarToAxes(axes[0, 0].scatter(table["nw_x"], table["nw_y"], c=table["T"], s=5))
    cbar.set_label("T [arcsec$^2$]")

    emax = np.quantile(np.abs(np.concatenate([table["e1"], table["e2"]])), 0.98)
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(
            table["nw_x"], table["nw_y"], c=table["nw_e1"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e1")

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(
            table["nw_x"], table["nw_y"], c=table["nw_e2"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e2")

    Q = axes[0, 1].quiver(
        table["nw_x"],
        table["nw_y"],
        table["e"] * np.cos(0.5 * np.arctan2(table["nw_e2"], table["nw_e1"])),
        table["e"] * np.sin(0.5 * np.arctan2(table["nw_e2"], table["nw_e1"])),
        headlength=0,
        headaxislength=0,
        scale=1,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.05, label="0.05", labelpos="S")

    for ax in axes.ravel():
        ax.set_xlabel("West")
        ax.set_ylabel("North")
        ax.set_aspect("equal")
        ax.set_xlim(-90, 90)
        ax.set_ylim(-90, 90)

    # Plot camera detector outlines
    nwRot = table.meta["nwRot"]
    for det in camera:
        xs = []
        ys = []
        for corner in det.getCorners(FOCAL_PLANE):
            xs.append(corner.x)
            ys.append(corner.y)
        xs.append(xs[0])
        ys.append(ys[0])
        xs = np.array(xs)
        ys = np.array(ys)
        rxs = nwRot[0, 0] * xs + nwRot[0, 1] * ys
        rys = nwRot[1, 0] * xs + nwRot[1, 1] * ys
        for ax in axes.ravel():
            ax.plot(rxs, rys, c="k", lw=1, alpha=0.3)

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)


def makeAzElPlot(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPoints: int = 1000,
    saveAs: str = "",
    saveTableAs: str = "",  # New parameter for saving the table
):
    """Plot the PSFs on the focal plane, rotated to az/el coordinates.

    Top left:
        A scatter plot of the T values in square arcseconds.
    Top right:
        A quiver plot of e1 and e2
    Bottom left:
        A scatter plot of e1
    Bottom right:
        A scatter plot of e2

    This function plots the data from the ``table`` on the provided ``fig`` and
    ``axes`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `numpy.ndarray`
        The table containing the data to be plotted.
    camera : `list`
        The list of camera detector objects.
    maxPoints : `int`, optional
        The maximum number of points to plot. If the number of points in the
        table is greater than this value, a random subset of points will be
        plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    oneRaftOnly = camera.getName() in ["LSSTComCam", "LSSTComCamSim", "TS8"]
    # I think this is roughly right for plotting - diameter is 5x but we need
    # less border, and 4.5 looks about right by eye.
    fullCameraFactor = 4.5
    plotLimit = 90 if oneRaftOnly else 90 * fullCameraFactor
    quiverScale = 5 if oneRaftOnly else fullCameraFactor

    table = randomRows(table, maxPoints)

    cbar = addColorbarToAxes(axes[0, 0].scatter(table["aa_x"], table["aa_y"], c=table["T"], s=5))
    cbar.set_label("T [arcsec$^2$]")

    emax = np.quantile(np.abs(np.concatenate([table["e1"], table["e2"]])), 0.98)
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(
            table["aa_x"], table["aa_y"], c=table["aa_e1"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e1")

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(
            table["aa_x"], table["aa_y"], c=table["aa_e2"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e2")

    Q = axes[0, 1].quiver(
        table["aa_x"],
        table["aa_y"],
        table["e"] * np.cos(0.5 * np.arctan2(table["aa_e2"], table["aa_e1"])),
        table["e"] * np.sin(0.5 * np.arctan2(table["aa_e2"], table["aa_e1"])),
        headlength=0,
        headaxislength=0,
        scale=quiverScale,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.2, label="0.2", labelpos="S")

    for ax in axes.ravel():
        ax.set_xlabel("Az")
        ax.set_ylabel("Alt")
        ax.set_aspect("equal")
        ax.set_xlim(-plotLimit, plotLimit)
        ax.set_ylim(-plotLimit, plotLimit)
    if saveTableAs:    #D-IO
        # Create a DataFrame with the necessary columns
        data_to_save = {
            "Ixx": table["Ixx"],
            "Ixy": table["Ixy"],
            "Iyy": table["Iyy"],
            "T": table["T"],
            "e": table["e"],
            "e1": table["e1"],
            "e2": table["e2"],
            "x": table["x"],
            "y": table["y"], 
            "aa_Ixx": table["aa_Ixx"],
            "aa_Ixy": table["aa_Ixy"],
            "aa_Iyy": table["aa_Iyy"],
            "aa_e1": table["aa_e1"],
            "aa_e2": table["aa_e2"],
            "aa_x": table["aa_x"],
            "aa_y": table["aa_y"],   
        }
        df = pd.DataFrame(data_to_save)
        df.to_csv(saveTableAs, index=False)
        print(f"Table saved as {saveTableAs}")

    # Plot camera detector outlines - only plot labels on single-raft cameras;
    # otherwise it is overwhelming
    if oneRaftOnly:
        aaRot = table.meta["aaRot"]
        for det in camera:
            xs = []
            ys = []
            for corner in det.getCorners(FOCAL_PLANE):
                xs.append(corner.x)
                ys.append(corner.y)
            xs.append(xs[0])
            ys.append(ys[0])
            xs = np.array(xs)
            ys = np.array(ys)
            rxs = aaRot[0, 0] * xs + aaRot[0, 1] * ys
            rys = aaRot[1, 0] * xs + aaRot[1, 1] * ys
            # Place detector label
            x = min([c.x for c in det.getCorners(FOCAL_PLANE)])
            y = max([c.y for c in det.getCorners(FOCAL_PLANE)])
            rx = aaRot[0, 0] * x + aaRot[0, 1] * y
            ry = aaRot[1, 0] * x + aaRot[1, 1] * y
            rtp = table.meta["rotTelPos"]
            for ax in axes.ravel():
                ax.plot(rxs, rys, c="k", lw=1, alpha=0.3)
                ax.text(
                    rx,
                    ry,
                    det.getName(),
                    rotation_mode="anchor",
                    rotation=np.rad2deg(-rtp) - 90,
                    horizontalalignment="left",
                    verticalalignment="top",
                    color="k",
                    fontsize=6,
                    zorder=20,
                )

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)