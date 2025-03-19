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

# def calcFieldXY(self):
#         """
#         Calculate the X, Y field position of the centroid in degrees.

#         Returns
#         -------
#         `float`
#             Field x position in degrees.
#         `float`
#             Field y position in degrees.
#         """

#         cam = self.getCamera()
#         det = cam.get(self.detector_name)

#         field_x, field_y = det.transform(self.centroid_position, PIXELS, FIELD_ANGLE)

#         return np.degrees(field_x), np.degrees(field_y)

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



    