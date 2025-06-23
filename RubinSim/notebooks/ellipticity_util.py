from __future__ import annotations

# __all__ = [
#     "calcFieldXY",
#     # "makeTableFromSourceCatalogs",
#     # "makeFigureAndAxes",
#     # "extendTable",
#     # "makeFocalPlanePlot",
#     # "makeEquatorialPlot",
#     # "makeAzElPlot",
# ]

from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.geom import Point2D
from lsst.summit.extras.plotting.psfPlotting import randomRows, addColorbarToAxes, extendTable
import numpy as np
from astropy.io import ascii
from astropy.table import Table

# import astropy.units as u
# from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
# from astropy.time import Time

# def pseudo_parallactic_angle(
#     ra: float,
#     dec: float,
#     mjd: float,
#     lon: float = -70.7494,
#     lat: float = -30.2444,
#     height: float = 2650.0,
#     pressure: float = 750.0,
#     temperature: float = 11.5,
#     relative_humidity: float = 0.4,
#     obswl: float = 1.0,
# ):
#     """Compute the pseudo parallactic angle.

#     The (traditional) parallactic angle is the angle zenith - coord - NCP
#     where NCP is the true-of-date north celestial pole.  This function instead
#     computes zenith - coord - NCP_ICRF where NCP_ICRF is the north celestial
#     pole in the International Celestial Reference Frame.

#     Parameters
#     ----------
#     ra, dec : float
#         ICRF coordinates in degrees.
#     mjd : float
#         Modified Julian Date.
#     latitude, longitude : float
#         Geodetic coordinates of observer in degrees.
#     height : float
#         Height of observer above reference ellipsoid in meters.
#     pressure : float
#         Atmospheric pressure in millibars.
#     temperature : float
#         Atmospheric temperature in degrees Celsius.
#     relative_humidity : float
#     obswl : float
#         Observation wavelength in microns.

#     Returns
#     -------
#     ppa : float
#         The pseudo parallactic angle in degrees.
#     """
#     obstime = Time(mjd, format="mjd", scale="tai")
#     location = EarthLocation.from_geodetic(
#         lon=lon * u.deg,
#         lat=lat * u.deg,
#         height=height * u.m,
#         ellipsoid="WGS84",  # For concreteness
#     )

#     coord_kwargs = dict(
#         obstime=obstime,
#         location=location,
#         pressure=pressure * u.mbar,
#         temperature=temperature * u.deg_C,
#         relative_humidity=relative_humidity,
#         obswl=obswl * u.micron,
#     )

#     coord = SkyCoord(ra * u.deg, dec * u.deg, **coord_kwargs)

#     towards_zenith = SkyCoord(
#         alt=coord.altaz.alt + 10 * u.arcsec,
#         az=coord.altaz.az,
#         frame=AltAz,
#         **coord_kwargs
#     )

#     towards_north = SkyCoord(
#         ra=coord.icrs.ra, dec=coord.icrs.dec + 10 * u.arcsec, **coord_kwargs
#     )

#     ppa = coord.position_angle(towards_zenith) - coord.position_angle(towards_north)
#     return ppa.wrap_at(180 * u.deg).deg
    
# def rtp_to_rsp(rotTelPos: float, ra: float, dec: float, mjd: float, **kwargs: dict):
#     """Convert RotTelPos -> RotSkyPos.

#     Parameters
#     ----------
#     rotTelPos : float
#         Camera rotation angle in degrees.
#     ra, dec : float
#         ICRF coordinates in degrees.
#     mjd : float
#         Modified Julian Date.
#     **kwargs : dict
#         Other keyword arguments to pass to pseudo_parallactic_angle.  Defaults
#         are generally appropriate for Rubin Observatory.

#     Returns
#     -------
#     rsp : float
#         RotSkyPos in degrees.
#     """
#     q = pseudo_parallactic_angle(ra, dec, mjd, **kwargs)
#     return Angle((270 - rotTelPos + q)*u.deg).wrap_at(180 * u.deg).deg

# def rsp_to_rtp(rotSkyPos: float, ra: float, dec: float, mjd: float, **kwargs: dict):
#     """Convert RotTelPos -> RotSkyPos.

#     Parameters
#     ----------
#     rotSkyPos : float
#         Sky rotation angle in degrees.
#     ra, dec : float
#         ICRF coordinates in degrees.
#     mjd : float
#         Modified Julian Date.
#     **kwargs : dict
#         Other keyword arguments to pass to pseudo_parallactic_angle.  Defaults
#         are generally appropriate for Rubin Observatory.

#     Returns
#     -------
#     rsp : float
#         RotSkyPos in degrees.
#     """
#     q = pseudo_parallactic_angle(ra, dec, mjd, **kwargs)
#     return Angle((270 - rotSkyPos + q)*u.deg).wrap_at(180 * u.deg)
    
def addFieldCoords_to_Table( table, camera ):
    """Extend the given table with additional columns.

    Parameters
    ----------
    table : `astropy.table.Table`
        The input table containing the original shapes.
    camera : `list`
        The list of camera detector objects.
    rot: `arr`
        rotation matrix
    Returns
    -------
    table : `astropy.table.Table`
        The extended table with additional columns representing field
        angle coordinates and rotated versions
    """
    field_points = []
    field_points_rot = []

    rot = table.meta['ocRot']
    
    for row in table:
        point = Point2D(row['slot_Centroid_x'], row['slot_Centroid_y'])
        det = camera[row['detector']]
        field_point = det.transform(point, PIXELS, FIELD_ANGLE)
        field_points.append( field_point )

        field_points_rot.append( rot @ [[field_point.x],[field_point.y]] )

    table['field_x'] = [ np.rad2deg( point.x ) for point in field_points ]
    table['field_y'] = [ np.rad2deg( point.y ) for point in field_points ]
    table['field_x'].unit='deg'
    table['field_y'].unit='deg'
    table['oc_field_x'] = [ np.rad2deg( point.flatten()[0] ) for point in field_points_rot ]
    table['oc_field_y'] = [ np.rad2deg( point.flatten()[1] ) for point in field_points_rot ]
    table['oc_field_x'].unit='deg'
    table['oc_field_y'].unit='deg'

    # table[prefix + "_x"] = rot[0, 0] * table["x"] + rot[0, 1] * table["y"]
    # table[prefix + "_y"] = rot[1, 0] * table["x"] + rot[1, 1] * table["y"]
    
    return table

def makeOCSPlot(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPoints: int = 1000,
    saveAs: str = "",
    autoscale = True, 
    scalemin_e = 0.,
    scalemax_e = 0.45,
    scalemin_e1 = -0.2,
    scalemax_e1 = 0.2,
    scalemin_e2 = -0.2,
    scalemax_e2 = 0.2,
    scale_quiver = 0.2,
    size_quiver = 0.1
):
    """Plot the PSFs on the focal plane, rotated to az/el which is taken to be OCS.

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
    fullCameraFactor = None
    # plotLimit = 90 if oneRaftOnly else 90 * fullCameraFactor
    quiverScale = None if oneRaftOnly else fullCameraFactor

    table = randomRows(table, maxPoints)

    if autoscale:
        cbar = addColorbarToAxes(axes[0, 0].scatter(table["oc_field_x"], table["oc_field_y"], c=table["T"], s=5))
        cbar.set_label("T [arcsec$^2$]")
    
        emax = np.quantile(np.abs(np.concatenate([table["oc_e1"], table["oc_e2"]])), 0.98)
        cbar = addColorbarToAxes(
            axes[1, 0].scatter(
                table["oc_field_x"], table["oc_field_y"], c=table["oc_e1"], vmin=-emax, vmax=emax, cmap="bwr", s=5
            )
        )
        cbar.set_label("e1")
    
        cbar = addColorbarToAxes(
            axes[1, 1].scatter(
                table["oc_field_x"], table["oc_field_y"], c=table["oc_e2"], vmin=-emax, vmax=emax, cmap="bwr", s=5
            )
        )
        cbar.set_label("e2")
    
        Q = axes[0, 1].quiver(
            table["oc_field_x"],
            table["oc_field_y"],
            table["e"] * np.cos(0.5 * np.arctan2(table["oc_e2"], table["oc_e1"])),
            table["e"] * np.sin(0.5 * np.arctan2(table["oc_e2"], table["oc_e1"])),
            headlength=0,
            headaxislength=0,
            scale=quiverScale,
            pivot="middle",
        )
        axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=size_quiver, label="{:4.2f}".format(size_quiver), labelpos="S")

    else:
        cbar = addColorbarToAxes(axes[0, 0].scatter(table["oc_field_x"], table["oc_field_y"], c=table["T"], 
                                                    s=5, vmin=scalemin_e, vmax=scalemax_e))
        cbar.set_label("T [arcsec$^2$]")
    
        emax = np.quantile(np.abs(np.concatenate([table["oc_e1"], table["oc_e2"]])), 0.98)
        cbar = addColorbarToAxes(
            axes[1, 0].scatter(
                table["oc_field_x"], table["oc_field_y"], c=table["oc_e1"], vmin=scalemin_e1, vmax=scalemax_e1, cmap="bwr", s=5
            )
        )
        cbar.set_label("e1")
    
        cbar = addColorbarToAxes(
            axes[1, 1].scatter(
                table["oc_field_x"], table["oc_field_y"], c=table["oc_e2"], vmin=scalemin_e2, vmax=scalemax_e2, cmap="bwr", s=5
            )
        )
        cbar.set_label("e2")
    
        Q = axes[0, 1].quiver(
            table["oc_field_x"],
            table["oc_field_y"],
            table["e"] * np.cos(0.5 * np.arctan2(table["oc_e2"], table["oc_e1"])),
            table["e"] * np.sin(0.5 * np.arctan2(table["oc_e2"], table["oc_e1"])),
            headlength=0,
            headaxislength=0,
            scale=scale_quiver,
            pivot="middle",
        )
        axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=size_quiver, label="{:4.2f}".format(size_quiver), labelpos="S")

    for ax in axes.ravel():
        ax.set_xlabel("hx [deg]")
        ax.set_ylabel("hy [deg]")
        ax.set_aspect("equal")
        # ax.set_xlim(-plotLimit, plotLimit)
        # ax.set_ylim(-plotLimit, plotLimit)

    # Plot camera detector outlines - only plot labels on single-raft cameras;
    # otherwise it is overwhelming
    
    # if oneRaftOnly:
    #     aaRot = table.meta["ocRot"]
    #     for det in camera:
    #         xs = []
    #         ys = []
    #         for corner in det.getCorners(FOCAL_PLANE):
    #             xs.append(corner.x)
    #             ys.append(corner.y)
    #         xs.append(xs[0])
    #         ys.append(ys[0])
    #         xs = np.array(xs)
    #         ys = np.array(ys)
    #         rxs = aaRot[0, 0] * xs + aaRot[0, 1] * ys
    #         rys = aaRot[1, 0] * xs + aaRot[1, 1] * ys
    #         # Place detector label
    #         x = min([c.x for c in det.getCorners(FOCAL_PLANE)])
    #         y = max([c.y for c in det.getCorners(FOCAL_PLANE)])
    #         rx = aaRot[0, 0] * x + aaRot[0, 1] * y
    #         ry = aaRot[1, 0] * x + aaRot[1, 1] * y
    #         rtp = table.meta["rotTelPos"]
    #         for ax in axes.ravel():
    #             ax.plot(rxs, rys, c="k", lw=1, alpha=0.3)
    #             ax.text(
    #                 rx,
    #                 ry,
    #                 det.getName(),
    #                 rotation_mode="anchor",
    #                 rotation=np.rad2deg(-rtp) - 90,
    #                 horizontalalignment="left",
    #                 verticalalignment="top",
    #                 color="k",
    #                 zorder=20,
    #             )

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)
        
def makeOCSPlot_bk(
    fig: plt.Figure,
    axes: np.ndarray[plt.Axes],
    table: Table,
    camera: Camera,
    maxPoints: int = 1000,
    saveAs: str = "",
):
    """Plot the PSFs on the focal plane, rotated to az/el which is taken to be OCS.

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
    fullCameraFactor = None
    # plotLimit = 90 if oneRaftOnly else 90 * fullCameraFactor
    quiverScale = None if oneRaftOnly else fullCameraFactor

    table = randomRows(table, maxPoints)

    cbar = addColorbarToAxes(axes[0, 0].scatter(table["oc_field_x"], table["oc_field_y"], c=table["T"], s=5))
    cbar.set_label("T [arcsec$^2$]")

    emax = np.quantile(np.abs(np.concatenate([table["e1"], table["e2"]])), 0.98)
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(
            table["oc_field_x"], table["oc_field_y"], c=table["oc_e1"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e1")

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(
            table["oc_field_x"], table["oc_field_y"], c=table["oc_e2"], vmin=-emax, vmax=emax, cmap="bwr", s=5
        )
    )
    cbar.set_label("e2")

    Q = axes[0, 1].quiver(
        table["oc_field_x"],
        table["oc_field_y"],
        table["e"] * np.cos(0.5 * np.arctan2(table["oc_e2"], table["oc_e1"])),
        table["e"] * np.sin(0.5 * np.arctan2(table["oc_e2"], table["oc_e1"])),
        headlength=0,
        headaxislength=0,
        scale=quiverScale,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.2, label="0.2", labelpos="S")

    for ax in axes.ravel():
        ax.set_xlabel("hx [deg]")
        ax.set_ylabel("hy [deg]")
        ax.set_aspect("equal")
        # ax.set_xlim(-plotLimit, plotLimit)
        # ax.set_ylim(-plotLimit, plotLimit)

    # Plot camera detector outlines - only plot labels on single-raft cameras;
    # otherwise it is overwhelming
    
    if oneRaftOnly:
        aaRot = table.meta["ocRot"]
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

def read_batoid_table( fname ):
    """['aa_field_x',
 'aa_field_y',
 'aa_Ixx',
 'aa_Iyy',
 'aa_Ixy',
 'hx',
 'hy',
 'mod_el',
 'mod_pa',
 'mod_mxx',
 'mod_myy',
 'mod_mxy']"""
    
    table = ascii.read( fname, data_start=1, delimiter=',',header_start=0)
    

    # table["Ixx"] = table["slot_Shape_xx"] * (0.2) ** 2
    # table["Ixy"] = table["slot_Shape_xy"] * (0.2) ** 2
    # table["Iyy"] = table["slot_Shape_yy"] * (0.2) ** 2

    # # assume some numbers for dispersion correction to check if we can make a square become a circle
    # table['aa_Iyy'] = table['aa_Iyy'] - (0.05)**2 
    
    table["T"] = table["aa_Ixx"] + table["aa_Iyy"]
    table["e1"] = (table["aa_Ixx"] - table["aa_Iyy"]) / table["T"]
    table["e2"] = 2 * table["aa_Ixy"] / table["T"]
    table["e"] = np.hypot(table["e1"], table["e2"])

    table["T_mod"] = table["mod_mxx"] + table["mod_myy"]
    table["e1_mod"] = (table["mod_mxx"] - table["mod_myy"]) / table["T_mod"]
    table["e2_mod"] = 2 * table["mod_mxy"] / table["T_mod"]
    table["e_mod"] = np.hypot(table["e1_mod"], table["e2_mod"])

    return table
    
def addOpticalCoords_to_Table( table ):
    # We call extendTable to add the optical system coordinate system which is almost aa but with an extra flip of the original Y axis.

    ocRot = np.array([[1, 0], [0, -1]]) @ table.meta["aaRot"]
    table = extendTable(table, ocRot, 'oc' )
    table.meta["ocRot"] = ocRot

    return table

def makeTableFromCalexps(calexps, sourcess, psfs, elevation_angle, visitInfo):

    for calexp, source, psf in zip(calexps, sourcess, psfs):
    
        e_star, ex_star_dvcs, ey_star_dvcs, ex_rot_star_dvcs, ey_rot_star_dvcs, i_xx_star, i_yy_star, i_xy_star, \
            aaIxx_star, aaIxy_star, aaIyy_star, e1, e2, xx_star_dvcs, yy_star_dvcs, theta_star_dvcs, \
            xx_rot_star_dvcs, yy_rot_star_dvcs, ra_star_dvcs, dec_star_dvcs, fwhm, size, fluxes_star = \
            calculate_ellipticity_on_xy(calexp, sources, psf, 1, n_grid=0, 
            rotation_sticks=2, do_flip=False, do_ad_correction=True, 
            zenith_angle=90.-elevation_angle)

    table = Table()

    table["oc_field_x"] = xx_rot_star_dvcs
    table["oc_field_y"] = yy_rot_star_dvcs

    return table

# def makeTableFromCalexps(icSrcs: dict[int, SourceCatalog], visitInfo: VisitInfo) -> Table:
    # """Extract the shapes from the source catalogs into an astropy table.

    # The shapes of the PSF candidates are extracted from the source catalogs and
    # transformed into the required coordinate systems for plotting either focal
    # plane coordinates, az/el coordinates, or equatorial coordinates.

    # Parameters
    # ----------
    # icSrcs : `dict` [`int`, `lsst.afw.table.SourceCatalog`]
    #     A dictionary of source catalogs, keyed by the detector numbers.
    # visitInfo : `lsst.afw.image.VisitInfo`
    #     The visit information for a representative visit.

    # Returns
    # -------
    # table : `astropy.table.Table`
    #     The table containing the data from the source catalogs.
    # """
    # tables = []

    # for detectorNum, icSrc in icSrcs.items():
    #     icSrc = icSrc.asAstropy()
    #     icSrc = icSrc[icSrc["calib_psf_candidate"]]
    #     icSrc["detector"] = detectorNum
    #     tables.append(icSrc)

    # table = vstack(tables)
    # # Add shape columns
    # table["Ixx"] = table["slot_Shape_xx"] * (0.2) ** 2
    # table["Ixy"] = table["slot_Shape_xy"] * (0.2) ** 2
    # table["Iyy"] = table["slot_Shape_yy"] * (0.2) ** 2
    # table["T"] = table["Ixx"] + table["Iyy"]
    # table["e1"] = (table["Ixx"] - table["Iyy"]) / table["T"]
    # table["e2"] = 2 * table["Ixy"] / table["T"]
    # table["e"] = np.hypot(table["e1"], table["e2"])
    # table["x"] = table["base_FPPosition_x"]
    # table["y"] = table["base_FPPosition_y"]

    # table.meta["rotTelPos"] = (
    #     visitInfo.boresightParAngle - visitInfo.boresightRotAngle - (np.pi / 2 * radians)
    # ).asRadians()
    # table.meta["rotSkyPos"] = visitInfo.boresightRotAngle.asRadians()

    # rtp = table.meta["rotTelPos"]
    # srtp, crtp = np.sin(rtp), np.cos(rtp)
    # aaRot = np.array([[crtp, srtp], [-srtp, crtp]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
    # table = extendTable(table, aaRot, "aa")
    # table.meta["aaRot"] = aaRot

    # rsp = table.meta["rotSkyPos"]
    # srsp, crsp = np.sin(rsp), np.cos(rsp)
    # nwRot = np.array([[crsp, -srsp], [srsp, crsp]])
    # table = extendTable(table, nwRot, "nw")
    # table.meta["nwRot"] = nwRot

    # return table

def MakeGridMedianPSF(table, nx, ny, min_cell):
    """Builds a table, in OC coordinates, where the PSF moments are stored
    on a grid (both regular and with weighted coordinates).

    Parameters
    ----------
    table : `astropy.table.Table`
        The input table containing the original shapes.
    nx : `int`
        The number of points of the grid to be created on the OC x axis.
    ny : `int`
        The number of points of the grid to be created on the OC y axis.
    min_cell : `int`
        The minimum number of stars in the cell around the x,y positions in the grid. 
        If the number of star is smaller than min_cell, no median will be performed. 
        
    Returns
    -------
    table_grid : `astropy.table.Table`
        The new table with gridded medians.
    """
    
# Build the grid
    min_x = min(table['oc_field_x'])
    min_y = min(table['oc_field_y'])
    max_x = max(table['oc_field_x'])
    max_y = max(table['oc_field_y'])

    step_x = (max_x - min_x) / nx
    step_y = (max_y - min_y) / ny

    x_array = min_x + step_x * (np.arange(nx) + .5)
    y_array = min_y + step_y * (np.arange(ny) + .5)

    xx, yy = np.meshgrid(x_array, y_array)
    xx_for_zip = xx.flatten()
    yy_for_zip = yy.flatten()

    Ixx_oc_grid = []
    Iyy_oc_grid = []
    Ixy_oc_grid = []
    oc_grid_x_median = []
    oc_grid_y_median = []
    n_stars_in_cells = []
    grid_id = []
    all_ids = []
    
    iii = 0
    for x, y in zip(xx_for_zip, yy_for_zip):

        # Search sources within each cell
        ind_temp = (table['oc_field_x'] > (x - step_x/2.)) & (table['oc_field_x'] <= (x + step_x/2.)) & (table['oc_field_y'] > (y - step_y/2.)) & (table['oc_field_y'] <= (y + step_y/2.))
        itemindex = np.where(ind_temp == True)
        n_stars_in_cell = len(itemindex[0])
        
        oc_grid_x_median.append(np.median(table['oc_field_x'][itemindex]))
        oc_grid_y_median.append(np.median(table['oc_field_y'][itemindex]))
        n_stars_in_cells.append(n_stars_in_cell)

        # assign median moments to the grid
        if n_stars_in_cell >= min_cell:
            Ixx_oc_grid.append(np.median(table['oc_Ixx'][itemindex]))
            Iyy_oc_grid.append(np.median(table['oc_Iyy'][itemindex]))
            Ixy_oc_grid.append(np.median(table['oc_Ixy'][itemindex]))
        else:
            Ixx_oc_grid.append(np.median(np.nan))
            Iyy_oc_grid.append(np.median(np.nan))
            Ixy_oc_grid.append(np.median(np.nan))

        iii = iii + 1

    # print(grid_id.flatten())
    # print(all_ids.flatten())
    
    table_grid = Table()

    table_grid["oc_id"] = np.arange(yy_for_zip.size)
              
    table_grid["oc_field_x"] = xx_for_zip
    table_grid["oc_field_y"] = yy_for_zip

    # Median grid coordinates
    table_grid["oc_x_median"] = oc_grid_x_median
    table_grid["oc_y_median"] = oc_grid_y_median

    table_grid["n_stars_in_cells"] = n_stars_in_cells
    
    # Median moments
    table_grid["oc_Ixx"] = Ixx_oc_grid
    table_grid["oc_Ixy"] = Ixy_oc_grid
    table_grid["oc_Iyy"] = Iyy_oc_grid
              
    # Ellipticities
    table_grid["T"] = table_grid["oc_Ixx"] + table_grid["oc_Iyy"]
    table_grid["oc_e1"] = (table_grid["oc_Ixx"] - table_grid["oc_Iyy"]) / table_grid["T"]
    table_grid["oc_e2"] = 2 * table_grid["oc_Ixy"] / table_grid["T"]
    table_grid["e"] = np.hypot(table_grid["oc_e1"], table_grid["oc_e2"])

    # Add metadata
    table_grid.meta["ocRot"] = table.meta["ocRot"]
    table_grid.meta["rotTelPos"] = table.meta["rotTelPos"]
    
    return table_grid

# #XXX
# def calculate_ellipticity_on_xy(calexp, sources, psf, regular_grid_or_star_positions, 
#                                 rotation_sticks=1, do_flip=True, do_ad_correction=True, 
#                                 zenith_angle=0., pressure=727, temperature=6.85, n_grid=200, fileout=''):

#     #rotation_sticks= 1: rotazione di entità +rottelpos_radians degli sticks
#     #rotation_sticks= 0: nessuna rotazione degli sticks
#     #rotation_sticks=-1: rotazione di entità -rottelpos_radians degli sticks
#     #rotation_sticks=2: rotazione alla Josh con Quadrupole

#     det = calexp.getDetector()
#     wcs = calexp.getWcs()
#     visit_id = calexp.info.getVisitInfo().getId()

#     passband = calexp.getInfo().getFilter().bandLabel
    
#     rotskypos = (calexp.info.getVisitInfo().getBoresightRotAngle()).asDegrees()
#     rottelpos = rsp_to_rtp(rotskypos, \
#             (calexp.info.getVisitInfo().getBoresightRaDec())[0].asDegrees(), \
#             (calexp.info.getVisitInfo().getBoresightRaDec())[1].asDegrees(), \
#             calexp.info.getVisitInfo().getDate().toAstropy()).deg
#     rottelpos_radians = np.radians(rottelpos)

#     if rotation_sticks>=1:
#         rottelpos_radians_for_ellipticitysticks = rottelpos_radians
#     elif rotation_sticks==0:
#         rottelpos_radians_for_ellipticitysticks = 0.
#     elif rotation_sticks==-1:
#         rottelpos_radians_for_ellipticitysticks = -rottelpos_radians

#     crtp = np.cos(rottelpos_radians)
#     srtp = np.sin(rottelpos_radians)

#     # aa sta per Alt-Azimuth, trasformazione da codice di Josh:
#     # https://github.com/lsst-sitcom/summit_extras/blob/main/python/lsst/summit/extras/plotting/psfPlotting.py
#     aaRot = np.array([[crtp, srtp], [-srtp, crtp]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
#     ocRot = np.array([[1, 0], [0, -1]]) @ aaRot # Rotazione Ricardo
#     transform_for_ell = LinearTransform(aaRot)

#     # ------------Get the points on grid/star positions (in CCS)------------
#     if regular_grid_or_star_positions == 0:
#         # Per la visualizzazione della PSF su griglia regolare
#         grid_separation_x = calexp.getDimensions()[0] / n_grid
#         grid_separation_y = calexp.getDimensions()[1] / n_grid
#         x_array = np.arange(n_grid)*grid_separation_x + grid_separation_x/2.
#         y_array = np.arange(n_grid)*grid_separation_y + grid_separation_y/2.
#         xx, yy = np.meshgrid(x_array, y_array)
#         xx_for_zip = xx.flatten()
#         yy_for_zip = yy.flatten()
#         xxshape = n_grid*n_grid

#     elif regular_grid_or_star_positions == 1:
#         # Per la visualizzazione della PSF sulle coordinate delle stelle
#         xx = [l.getCentroid()[0] for l in sources]
#         yy = [l.getCentroid()[1] for l in sources]
#         xx_for_zip = xx
#         yy_for_zip = yy
#         xxshape = len(xx)
#         fluxes = [l.getPsfInstFlux() for l in sources]

#         # xx_fpposition = sources['base_FPPosition_x']
#         # yy_fpposition = sources['base_FPPosition_y']

#     elif regular_grid_or_star_positions == 2:
#         xx_for_zip = [2000.]
#         yy_for_zip = [2000.]
#         xxshape = len(xx_for_zip)

#     # ------------convert CCS into DVCS and extract moments------------
#     size = []
#     i_xx = []
#     i_yy = []
#     i_xy = []
#     points = []

#     xx_star_dvcs = []
#     yy_star_dvcs = []
#     xx_rot_star_dvcs = []
#     yy_rot_star_dvcs = []
#     xx_ocrot_star = []
#     yy_ocrot_star = []
#     ra_star_dvcs = []
#     dec_star_dvcs = []
    
#     for x, y in zip(xx_for_zip, yy_for_zip):
#         point = Point2D(x, y)        
#         coo = wcs.pixelToSky(x, y)
#         cam_x, cam_y = pixel_to_camera_angle(point[0], point[1], det)
#         x0, y0 = np.asarray(cam_x[0]), np.asarray(cam_y[0])
#         # Rotazioni mie
#         # xx_rot = x0*crtp - y0*srtp
#         # yy_rot = x0*srtp + y0*crtp
#         # Rotazioni Josh
#         xx_rot = aaRot[0, 0] * x0 + aaRot[0, 1] * y0
#         yy_rot = aaRot[1, 0] * x0 + aaRot[1, 1] * y0
#         xx_ocrot = ocRot[0, 0] * x0 + ocRot[0, 1] * y0
#         yy_ocrot = ocRot[1, 0] * x0 + ocRot[1, 1] * y0

#         shape = psf.computeShape(point)
#         size.append(shape.getTraceRadius())
#         i_xx.append(shape.getIxx())
#         i_yy.append(shape.getIyy())
#         i_xy.append(shape.getIxy())
#         points.append(point)

#         if do_flip:
#             xx_star_dvcs.append(cam_y[0])
#             yy_star_dvcs.append(cam_x[0])
#             xx_rot_star_dvcs.append(yy_rot)
#             yy_rot_star_dvcs.append(xx_rot)
#         else:
#             xx_star_dvcs.append(cam_x[0])
#             yy_star_dvcs.append(cam_y[0])
#             xx_rot_star_dvcs.append(xx_rot)
#             yy_rot_star_dvcs.append(yy_rot)
#             xx_ocrot_star.append(xx_ocrot)
#             yy_ocrot_star.append(yy_ocrot)

#         ra_star_dvcs.append(coo[0].asDegrees())
#         dec_star_dvcs.append(coo[1].asDegrees())

#     size = np.reshape(size, xxshape)
#     i_xx = np.reshape(i_xx, xxshape)
#     i_yy = np.reshape(i_yy, xxshape)
#     i_xy = np.reshape(i_xy, xxshape)

#     table_moments = {'Ixx': i_xx, 'Iyy': i_yy, 'Ixy': i_xy}

#     # ------------Transform moments into ellipticities------------
#     theta = np.arctan2(2. * i_xy, i_xx - i_yy) / 2.

#     e1 = (i_xx - i_yy) / (i_xx + i_yy)
#     e2 = (2. * i_xy) / (i_xx + i_yy)
    
#     theta_alternate = np.arctan2(e2, e1) / 2.
#     assert np.allclose(theta, theta_alternate)

#     e_star = np.sqrt(e1**2 + e2**2)
#     ex = e_star * np.cos(theta)
#     ey = e_star * np.sin(theta)

#     if do_flip:
#     # --- Con inversione XY degli stick--- OBSOLETE!!!!
#         # Rotazioni mie
#         ey_star_dvcs = ex
#         ex_star_dvcs = ey
#         crtp_e = np.cos(rottelpos_radians_for_ellipticitysticks)
#         srtp_e = np.sin(rottelpos_radians_for_ellipticitysticks)
#         ey_rot_star_dvcs = ex*crtp_e - ey*srtp_e
#         ex_rot_star_dvcs = ex*srtp_e + ey*crtp_e
#     else:
#     # --- Senza inversione XY degli stick--- DEFAULT!!!!
#         ey_star_dvcs = ey
#         ex_star_dvcs = ex

#         # Rotazioni mie
#         # ex_rot_star_dvcs = ex*crtp_e - ey*srtp_e
#         # ey_rot_star_dvcs = ex*srtp_e + ey*crtp_e

#         if rotation_sticks==2:
#             # Rotazioni Josh (Quadrupole)
#             rot_shapes = []
#             for i_xx1, i_yy1, i_xy1 in zip(i_xx, i_yy, i_xy):
#                 shape = Quadrupole(i_xx1, i_yy1, i_xy1)
#                 rot_shapes.append(shape.transform(transform_for_ell))
                
#             aaIxx = np.asarray([sh.getIxx() for sh in rot_shapes])
#             aaIyy = np.asarray([sh.getIyy() for sh in rot_shapes])
#             aaIxy = np.asarray([sh.getIxy() for sh in rot_shapes])

#             if do_ad_correction:
#                 aaIyy = aaIyy - compute_atm_dispersion(zenith_angle, passband, pression=pressure, temperature=temperature)
            
#             e1_rot_star_dvcs = (aaIxx - aaIyy) / (aaIxx + aaIyy)
#             e2_rot_star_dvcs = 2 * aaIxy / (aaIxx + aaIyy)    
            
#             theta_josh = np.arctan2(e2_rot_star_dvcs, e1_rot_star_dvcs) / 2.
#             e_star = np.sqrt(e1_rot_star_dvcs**2 + e2_rot_star_dvcs**2)
#             ex_rot_star_dvcs = e_star * np.cos(theta_josh)
#             ey_rot_star_dvcs = e_star * np.sin(theta_josh)
#         else:
#             # Rotazioni con aaRot, tutte sbagliate???
#             ex_rot_star_dvcs = aaRot[0, 0] * ex + aaRot[0, 1] * ey
#             ey_rot_star_dvcs = aaRot[1, 0] * ex + aaRot[1, 1] * ey
    
#     theta_star_dvcs = np.arctan2(ey, ex)

#     fwhm = []
#     # FWHM
#     for point in points:
#         sigma = psf.computeShape(point).getDeterminantRadius()
#         pixelScale = calexp.getWcs().getPixelScale().asArcseconds()
#         fwhm_temp = sigma * pixelScale * 2.355
#         fwhm.append(fwhm_temp)
    
#     if (regular_grid_or_star_positions == 0) | (regular_grid_or_star_positions == 2):
#         if fileout != '':
#             df = pd.DataFrame(data={'x_pixel_ccs': xx_for_zip, 'y_pixel_ccs': yy_for_zip, 'e_star': e_star, 
#                                'ex_star_dvcs': ex_star_dvcs, 'ey_star_dvcs': ey_star_dvcs, 
#                                 'ex_rot_star_dvcs': ex_rot_star_dvcs, 'ey_rot_star_dvcs': ey_rot_star_dvcs, 
#                                 'i_xx': i_xx, 'i_yy': i_yy, 'i_xx': i_xy,  
#                                 'e1': e1, 'e2': e2, 'theta_star_dvcs': theta_star_dvcs,
#                                 'xx_star_dvcs': xx_star_dvcs, 'yy_star_dvcs': yy_star_dvcs, 
#                                 'xx_rot_star_dvcs': xx_rot_star_dvcs, 'yy_rot_star_dvcs': yy_rot_star_dvcs, 
#                                 'ra_star_dvcs': ra_star_dvcs, 'dec_star_dvcs': dec_star_dvcs,
#                                'theta_alternate':theta_alternate, 'fwhm': fwhm, 'detector': [det.getId()] * len(xx_for_zip), 
#                                'visit_id':  [visit_id] * len(xx_for_zip)})
#             df.to_csv(fileout, index=None)
#             # df['theta_rot_star_dvcs'] = np.degrees(np.arctan2(ex_rot_star_dvcs, ey_rot_star_dvcs))
#             # df[['xx_rot_star_dvcs', 'yy_rot_star_dvcs', 'e_star', 'theta_rot_star_dvcs']].to_csv(fileout+'_for_ricardo', index=None)
    
#         return e_star, ex_star_dvcs, ey_star_dvcs, ex_rot_star_dvcs, ey_rot_star_dvcs, i_xx, i_yy, i_xy, \
#             aaIxx, aaIxy, aaIyy, e1, e2, xx_star_dvcs, yy_star_dvcs, theta_star_dvcs, \
#             xx_rot_star_dvcs, yy_rot_star_dvcs, ra_star_dvcs, dec_star_dvcs, fwhm, size

#     elif regular_grid_or_star_positions == 1:
#         if fileout != '':
#             df = pd.DataFrame(data={'x_pixel_ccs': xx_for_zip, 'y_pixel_ccs': yy_for_zip, 'e_star': e_star, 
#                                'ex_star_dvcs': ex_star_dvcs, 'ey_star_dvcs': ey_star_dvcs, 
#                                 'ex_rot_star_dvcs': ex_rot_star_dvcs, 'ey_rot_star_dvcs': ey_rot_star_dvcs, 
#                                 'i_xx': i_xx, 'i_yy': i_yy, 'i_xx': i_xy,  
#                                 'e1': e1, 'e2': e2, 'theta_star_dvcs': theta_star_dvcs,
#                                 'xx_star_dvcs': xx_star_dvcs, 'yy_star_dvcs': yy_star_dvcs, 
#                                 'xx_rot_star_dvcs': xx_rot_star_dvcs, 'yy_rot_star_dvcs': yy_rot_star_dvcs, 
#                                 'ra_star_dvcs': ra_star_dvcs, 'dec_star_dvcs': dec_star_dvcs,
#                                'theta_alternate':theta_alternate, 'fwhm': fwhm, 'fluxes': fluxes, 'detector': [det.getId()] * len(xx_for_zip), 
#                                'visit_id':  [visit_id] * len(xx_for_zip)})
#             df.to_csv(fileout, index=None)
    
#         return e_star, ex_star_dvcs, ey_star_dvcs, ex_rot_star_dvcs, ey_rot_star_dvcs, i_xx, i_yy, i_xy, \
#             aaIxx, aaIxy, aaIyy, e1, e2, xx_star_dvcs, yy_star_dvcs, theta_star_dvcs, \
#             xx_rot_star_dvcs, yy_rot_star_dvcs, ra_star_dvcs, dec_star_dvcs, fwhm, size, fluxes
