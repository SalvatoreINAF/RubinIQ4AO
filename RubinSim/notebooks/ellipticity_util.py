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
from treegp import meanify

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
    size_quiver = 0.1,
    coords: str = "aa"
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
    
    if coords == 'aa':
        col_x = 'aa_x'
        col_y = 'aa_y'
        col_e1 = 'aa_e1'
        col_e2 = 'aa_e2'
        col_e = 'aa_e'
        col_t = 'aa_T'
    else:
        col_x = 'oc_field_x'
        col_y = 'oc_field_y'
        col_e1 = 'oc_e1'
        col_e2 = 'oc_e2'
        col_e = 'e'
        col_t = 'T'

    if autoscale:
        cbar = addColorbarToAxes(axes[0, 0].scatter(table[col_x], table[col_y], c=table[col_t], s=5))
        cbar.set_label("T [arcsec2]")

        emax = np.quantile(np.abs(np.concatenate([table[col_e1], table[col_e2]])), 0.98)
        cbar = addColorbarToAxes(
            axes[1, 0].scatter(
                table[col_x], table[col_y], c=table[col_e1], vmin=-emax, vmax=emax, cmap="bwr", s=5
            )
        )
        cbar.set_label("e1")

        cbar = addColorbarToAxes(
            axes[1, 1].scatter(
                table[col_x], table[col_y], c=table[col_e2], vmin=-emax, vmax=emax, cmap="bwr", s=5
            )
        )
        cbar.set_label("e2")

        Q = axes[0, 1].quiver(
            table[col_x],
            table[col_y],
            table[col_e] * np.cos(0.5 * np.arctan2(table[col_e2], table[col_e1])),
            table[col_e] * np.sin(0.5 * np.arctan2(table[col_e2], table[col_e1])),
            headlength=0,
            headaxislength=0,
            scale=quiverScale,
            pivot="middle",
        )
        axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=size_quiver, label="{:4.2f}".format(size_quiver), labelpos="S")

    else:
        cbar = addColorbarToAxes(axes[0, 0].scatter(table[col_x], table[col_y], c=table[col_t],
                                                    s=5, vmin=scalemin_e, vmax=scalemax_e))
        cbar.set_label("T [arcsec2]")

        emax = np.quantile(np.abs(np.concatenate([table[col_e1], table[col_e2]])), 0.98)
        cbar = addColorbarToAxes(
            axes[1, 0].scatter(
                table[col_x], table[col_y], c=table[col_e1], vmin=scalemin_e1, vmax=scalemax_e1, cmap="bwr", s=5
            )
        )
        cbar.set_label("e1")

        cbar = addColorbarToAxes(
            axes[1, 1].scatter(
                table[col_x], table[col_y], c=table[col_e2], vmin=scalemin_e2, vmax=scalemax_e2, cmap="bwr", s=5
            )
        )
        cbar.set_label("e2")

        Q = axes[0, 1].quiver(
            table[col_x],
            table[col_y],
            table[col_e] * np.cos(0.5 * np.arctan2(table[col_e2], table[col_e1])),
            table[col_e] * np.sin(0.5 * np.arctan2(table[col_e2], table[col_e1])),
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
    cbar.set_label("T [arcsec2]")

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
    ext_shapeHSM_HigherOrderMomentsSource_30_grid = []
    ext_shapeHSM_HigherOrderMomentsSource_12_grid = []
    ext_shapeHSM_HigherOrderMomentsSource_21_grid = []
    ext_shapeHSM_HigherOrderMomentsSource_03_grid = []
    ext_shapeHSM_HigherOrderMomentsSource_40_grid = []
    ext_shapeHSM_HigherOrderMomentsSource_04_grid = []
    ext_shapeHSM_HigherOrderMomentsSource_22_grid = []

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
            ext_shapeHSM_HigherOrderMomentsSource_30_grid.append(np.median(table['ext_shapeHSM_HigherOrderMomentsSource_30'][itemindex]))
            ext_shapeHSM_HigherOrderMomentsSource_12_grid.append(np.median(table['ext_shapeHSM_HigherOrderMomentsSource_12'][itemindex]))
            ext_shapeHSM_HigherOrderMomentsSource_21_grid.append(np.median(table['ext_shapeHSM_HigherOrderMomentsSource_21'][itemindex]))
            ext_shapeHSM_HigherOrderMomentsSource_03_grid.append(np.median(table['ext_shapeHSM_HigherOrderMomentsSource_03'][itemindex]))
            ext_shapeHSM_HigherOrderMomentsSource_40_grid.append(np.median(table['ext_shapeHSM_HigherOrderMomentsSource_40'][itemindex]))
            ext_shapeHSM_HigherOrderMomentsSource_04_grid.append(np.median(table['ext_shapeHSM_HigherOrderMomentsSource_04'][itemindex]))
            ext_shapeHSM_HigherOrderMomentsSource_22_grid.append(np.median(table['ext_shapeHSM_HigherOrderMomentsSource_22'][itemindex]))
        else:
            Ixx_oc_grid.append(np.median(np.nan))
            Iyy_oc_grid.append(np.median(np.nan))
            Ixy_oc_grid.append(np.median(np.nan))
            ext_shapeHSM_HigherOrderMomentsSource_30_grid.append(np.median(np.nan))
            ext_shapeHSM_HigherOrderMomentsSource_12_grid.append(np.median(np.nan))
            ext_shapeHSM_HigherOrderMomentsSource_21_grid.append(np.median(np.nan))
            ext_shapeHSM_HigherOrderMomentsSource_03_grid.append(np.median(np.nan))
            ext_shapeHSM_HigherOrderMomentsSource_40_grid.append(np.median(np.nan))
            ext_shapeHSM_HigherOrderMomentsSource_04_grid.append(np.median(np.nan))
            ext_shapeHSM_HigherOrderMomentsSource_22_grid.append(np.median(np.nan))

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

    table_grid["ext_shapeHSM_HigherOrderMomentsSource_30"] = ext_shapeHSM_HigherOrderMomentsSource_30_grid
    table_grid["ext_shapeHSM_HigherOrderMomentsSource_12"] = ext_shapeHSM_HigherOrderMomentsSource_12_grid
    table_grid["ext_shapeHSM_HigherOrderMomentsSource_21"] = ext_shapeHSM_HigherOrderMomentsSource_21_grid
    table_grid["ext_shapeHSM_HigherOrderMomentsSource_03"] = ext_shapeHSM_HigherOrderMomentsSource_03_grid
    table_grid["ext_shapeHSM_HigherOrderMomentsSource_40"] = ext_shapeHSM_HigherOrderMomentsSource_40_grid
    table_grid["ext_shapeHSM_HigherOrderMomentsSource_04"] = ext_shapeHSM_HigherOrderMomentsSource_04_grid
    table_grid["ext_shapeHSM_HigherOrderMomentsSource_22"] = ext_shapeHSM_HigherOrderMomentsSource_22_grid

    # Ellipticities
    table_grid["T"] = table_grid["oc_Ixx"] + table_grid["oc_Iyy"]
    table_grid["oc_e1"] = (table_grid["oc_Ixx"] - table_grid["oc_Iyy"]) / table_grid["T"]
    table_grid["oc_e2"] = 2 * table_grid["oc_Ixy"] / table_grid["T"]
    table_grid["e"] = np.hypot(table_grid["oc_e1"], table_grid["oc_e2"])

    # Add metadata
    table_grid.meta["ocRot"] = table.meta["ocRot"]
    table_grid.meta["rotTelPos"] = table.meta["rotTelPos"]

    return table_grid

def statistics_on_moments(table, minflux=-9.):
    """Calculates mean, median and standard deviation of moments.

    Parameters
    ----------
    table : `astropy.table.Table`
        The input table containing the original shapes.
    minflux : `float`
        The flux (in base_CircularApertureFlux_6_0_instFlux)
        below which the stars should be discarded for the calculation of statistics

    Returns
    -------
    statistics : `astropy.table.Table`
        A table with the statistics on moments.
    """

    # Selezione solo su flussi maggiori di una certa quantità
    ind_good = table['base_CircularApertureFlux_6_0_instFlux'] >= minflux

    table = table[ind_good]

    ext_shapeHSM_HigherOrderMomentsSource_30_mean = np.mean(table['ext_shapeHSM_HigherOrderMomentsSource_30'].value)
    ext_shapeHSM_HigherOrderMomentsSource_30_median = np.median(table['ext_shapeHSM_HigherOrderMomentsSource_30'].value)
    ext_shapeHSM_HigherOrderMomentsSource_30_std = np.std(table['ext_shapeHSM_HigherOrderMomentsSource_30'].value)

    ext_shapeHSM_HigherOrderMomentsSource_03_mean = np.mean(table['ext_shapeHSM_HigherOrderMomentsSource_03'].value)
    ext_shapeHSM_HigherOrderMomentsSource_03_median = np.median(table['ext_shapeHSM_HigherOrderMomentsSource_03'].value)
    ext_shapeHSM_HigherOrderMomentsSource_03_std = np.std(table['ext_shapeHSM_HigherOrderMomentsSource_03'].value)

    ext_shapeHSM_HigherOrderMomentsSource_22_mean = np.mean(table['ext_shapeHSM_HigherOrderMomentsSource_22'].value)
    ext_shapeHSM_HigherOrderMomentsSource_22_median = np.median(table['ext_shapeHSM_HigherOrderMomentsSource_22'].value)
    ext_shapeHSM_HigherOrderMomentsSource_22_std = np.std(table['ext_shapeHSM_HigherOrderMomentsSource_22'].value)

    ext_shapeHSM_HigherOrderMomentsSource_21_mean = np.mean(table['ext_shapeHSM_HigherOrderMomentsSource_21'].value)
    ext_shapeHSM_HigherOrderMomentsSource_21_median = np.median(table['ext_shapeHSM_HigherOrderMomentsSource_21'].value)
    ext_shapeHSM_HigherOrderMomentsSource_21_std = np.std(table['ext_shapeHSM_HigherOrderMomentsSource_21'].value)

    ext_shapeHSM_HigherOrderMomentsSource_12_mean = np.mean(table['ext_shapeHSM_HigherOrderMomentsSource_12'].value)
    ext_shapeHSM_HigherOrderMomentsSource_12_median = np.median(table['ext_shapeHSM_HigherOrderMomentsSource_12'].value)
    ext_shapeHSM_HigherOrderMomentsSource_12_std = np.std(table['ext_shapeHSM_HigherOrderMomentsSource_12'].value)

    ext_shapeHSM_HigherOrderMomentsSource_40_mean = np.mean(table['ext_shapeHSM_HigherOrderMomentsSource_40'].value)
    ext_shapeHSM_HigherOrderMomentsSource_40_median = np.median(table['ext_shapeHSM_HigherOrderMomentsSource_40'].value)
    ext_shapeHSM_HigherOrderMomentsSource_40_std = np.std(table['ext_shapeHSM_HigherOrderMomentsSource_40'].value)

    ext_shapeHSM_HigherOrderMomentsSource_04_mean = np.mean(table['ext_shapeHSM_HigherOrderMomentsSource_04'].value)
    ext_shapeHSM_HigherOrderMomentsSource_04_median = np.median(table['ext_shapeHSM_HigherOrderMomentsSource_04'].value)
    ext_shapeHSM_HigherOrderMomentsSource_04_std = np.std(table['ext_shapeHSM_HigherOrderMomentsSource_04'].value)

    oc_ixx_mean = np.mean(table['oc_Ixx'].value)
    oc_ixx_median = np.median(table['oc_Ixx'].value)
    oc_ixx_std = np.std(table['oc_Ixx'].value)

    oc_iyy_mean = np.mean(table['oc_Iyy'].value)
    oc_iyy_median = np.median(table['oc_Iyy'].value)
    oc_iyy_std = np.std(table['oc_Iyy'].value)

    oc_ixy_mean = np.mean(table['oc_Ixy'].value)
    oc_ixy_median = np.median(table['oc_Ixy'].value)
    oc_ixy_std = np.std(table['oc_Ixy'].value)

    oc_e1_mean = np.mean(table['oc_e1'].value)
    oc_e1_median = np.median(table['oc_e1'].value)
    oc_e1_std = np.std(table['oc_e1'].value)

    oc_e2_mean = np.mean(table['oc_e2'].value)
    oc_e2_median = np.median(table['oc_e2'].value)
    oc_e2_std = np.std(table['oc_e2'].value)

    aa_e_mean = np.mean(table['aa_e'].value)
    aa_e_median = np.median(table['aa_e'].value)
    aa_e_std = np.std(table['aa_e'].value)

    coma1_mean = np.mean(table['coma1'].value)
    coma1_median = np.median(table['coma1'].value)
    coma1_std = np.std(table['coma1'].value)

    coma2_mean = np.mean(table['coma2'].value)
    coma2_median = np.median(table['coma2'].value)
    coma2_std = np.std(table['coma2'].value)

    trefoil1_mean = np.mean(table['trefoil1'].value)
    trefoil1_median = np.median(table['trefoil1'].value)
    trefoil1_std = np.std(table['trefoil1'].value)

    trefoil2_mean = np.mean(table['trefoil2'].value)
    trefoil2_median = np.median(table['trefoil2'].value)
    trefoil2_std = np.std(table['trefoil2'].value)

    kurtosis_mean = np.mean(table['kurtosis'].value)
    kurtosis_median = np.median(table['kurtosis'].value)
    kurtosis_std = np.std(table['kurtosis'].value)

    e_mean = np.mean(table['e'].value)
    e_median = np.median(table['e'].value)
    e_std = np.std(table['e'].value)

    T_mean = np.mean(table['T'].value)
    T_median = np.median(table['T'].value)
    T_std = np.std(table['T'].value)

    statistics = Table()
    statistics['oc_ixx_mean'] = [oc_ixx_mean]
    statistics['oc_iyy_mean'] = [oc_iyy_mean]
    statistics['oc_ixy_mean'] = [oc_ixy_mean]
    statistics['oc_e1_mean'] = [oc_e1_mean]
    statistics['oc_e2_mean'] = [oc_e2_mean]
    statistics['aa_e_mean'] = [aa_e_mean]
    statistics['e_mean'] = [e_mean]
    statistics['T_mean'] = [T_mean]
    statistics['oc_ixx_median'] = [oc_ixx_median]
    statistics['oc_iyy_median'] = [oc_iyy_median]
    statistics['oc_ixy_median'] = [oc_ixy_median]
    statistics['oc_e1_median'] = [oc_e1_median]
    statistics['oc_e2_median'] = [oc_e2_median]
    statistics['aa_e_median'] = [aa_e_median]
    statistics['e_median'] = [e_median]
    statistics['T_median'] = [T_median]
    statistics['oc_ixx_std'] = [oc_ixx_std]
    statistics['oc_iyy_std'] = [oc_iyy_std]
    statistics['oc_ixy_std'] = [oc_ixy_std]
    statistics['oc_e1_std'] = [oc_e1_std]
    statistics['oc_e2_std'] = [oc_e2_std]
    statistics['aa_e_std'] = [aa_e_std]
    statistics['e_std'] = [e_std]
    statistics['T_std'] = [T_std]

    statistics['ext_shapeHSM_HigherOrderMomentsSource_30_mean'] = [ext_shapeHSM_HigherOrderMomentsSource_30_mean]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_30_median'] = [ext_shapeHSM_HigherOrderMomentsSource_30_median]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_30_std'] = [ext_shapeHSM_HigherOrderMomentsSource_30_std]

    statistics['ext_shapeHSM_HigherOrderMomentsSource_03_mean'] = [ext_shapeHSM_HigherOrderMomentsSource_03_mean]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_03_median'] = [ext_shapeHSM_HigherOrderMomentsSource_03_median]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_03_std'] = [ext_shapeHSM_HigherOrderMomentsSource_03_std]

    statistics['ext_shapeHSM_HigherOrderMomentsSource_22_mean'] = [ext_shapeHSM_HigherOrderMomentsSource_22_mean]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_22_median'] = [ext_shapeHSM_HigherOrderMomentsSource_22_median]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_22_std'] = [ext_shapeHSM_HigherOrderMomentsSource_22_std]

    statistics['ext_shapeHSM_HigherOrderMomentsSource_21_mean'] = [ext_shapeHSM_HigherOrderMomentsSource_21_mean]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_21_median'] = [ext_shapeHSM_HigherOrderMomentsSource_21_median]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_21_std'] = [ext_shapeHSM_HigherOrderMomentsSource_21_std]

    statistics['ext_shapeHSM_HigherOrderMomentsSource_12_mean'] = [ext_shapeHSM_HigherOrderMomentsSource_12_mean]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_12_median'] = [ext_shapeHSM_HigherOrderMomentsSource_12_median]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_12_std'] = [ext_shapeHSM_HigherOrderMomentsSource_12_std]

    statistics['ext_shapeHSM_HigherOrderMomentsSource_40_mean'] = [ext_shapeHSM_HigherOrderMomentsSource_40_mean]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_40_median'] = [ext_shapeHSM_HigherOrderMomentsSource_40_median]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_40_std'] = [ext_shapeHSM_HigherOrderMomentsSource_40_std]

    statistics['ext_shapeHSM_HigherOrderMomentsSource_04_mean'] = [ext_shapeHSM_HigherOrderMomentsSource_04_mean]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_04_median'] = [ext_shapeHSM_HigherOrderMomentsSource_04_median]
    statistics['ext_shapeHSM_HigherOrderMomentsSource_04_std'] = [ext_shapeHSM_HigherOrderMomentsSource_04_std]

    return statistics

def addHigherMoments(table):
    table["coma1"] = (table["ext_shapeHSM_HigherOrderMomentsSource_30"]
                + table["ext_shapeHSM_HigherOrderMomentsSource_12"])
    table["coma2"] = (table["ext_shapeHSM_HigherOrderMomentsSource_21"]
                + table["ext_shapeHSM_HigherOrderMomentsSource_03"])
    table["trefoil1"] = (table["ext_shapeHSM_HigherOrderMomentsSource_30"]
                - 3 * table["ext_shapeHSM_HigherOrderMomentsSource_12"])
    table["trefoil2"] = (3 * table["ext_shapeHSM_HigherOrderMomentsSource_21"]
                - table["ext_shapeHSM_HigherOrderMomentsSource_03"])
    table["kurtosis"] = (table["ext_shapeHSM_HigherOrderMomentsSource_40"]
                + table["ext_shapeHSM_HigherOrderMomentsSource_04"]
                + 2 * table["ext_shapeHSM_HigherOrderMomentsSource_22"])

    return table

def randomRowsPerDetector(table: Table, maxRowsPerDetector: int) -> Table:
    """Select a random subset of rows for each detector from the given table.

    Parameters
    ----------
    table : `astropy.table.Table`
        The table containing the data to be plotted.
    maxRowsPerDetector : `int`
        The maximum number of rows to select per detector.

    Returns
    -------
    table : `astropy.table.Table`
        The table containing the randomly selected subset of rows.
    """
    keep = np.full(len(table), False)
    for det in np.unique(table["detector"]):
        detrows = np.nonzero(table["detector"] == det)[0]
        if len(detrows) > maxRowsPerDetector:
            rng = np.random.default_rng()
            detrows = rng.choice(detrows, maxRowsPerDetector, replace=False)
        keep[detrows] = True
    return table[keep]

def plotHigherOrderMomentsData(
    fig: plt.Figure, # modded
    axs: np.ndarray[plt.Axes], # modded
    # axs: npt.NDArray[np.object_], # modded
    table: Table,
    prefix: str = "",
    binSpacing: float = 0.1,
    is_grid = False,
    maxPointsPerDetector: int = 5,
):
    """Plot coma, trefoil and kurtosis from the table on the provided axes.

    Parameters
    ----------
    axs : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `Table`
        The table containing the data to plot.
    prefix : `str`, optional
        The prefix to use for the column names in the table.
    binSpacing : `float`, optional
        The spacing between arrows and triangles in the plot.
    maxPointsPerDetector : `int`, optional
        The maximum number of points per detector to plot. If the number of
        points exceeds this value, a random subset will be selected.
    """
    if not is_grid:
        table = randomRowsPerDetector(table, maxPointsPerDetector)

    x = table[prefix + "x"]
    y = table[prefix + "y"]
    kurtosis = table["kurtosis"]

    quiverKwargs = {
        "headlength": 5,
        "headaxislength": 5,
        "scale": 1,
        "pivot": "middle",
        "width": 0.002,
    }

    coords = np.vstack([x, y]).T
    meanComa = {}
    for i in (1, 2):
        binning = meanify(binSpacing)
        binning.add_field(coords, table[f"coma{i}"])
        binning.meanify()
        meanComa[i] = binning.params0

    meanComaAngle = np.arctan2(meanComa[2], meanComa[1])
    meanComaAmplitude = np.hypot(meanComa[2], meanComa[2])
    qComa = axs[0].quiver(
        binning.coords0[:, 0],
        binning.coords0[:, 1],
        meanComaAmplitude * np.cos(meanComaAngle),
        meanComaAmplitude * np.sin(meanComaAngle),
        **quiverKwargs,
    )
    axs[0].quiverkey(qComa, X=0.1, Y=0.88, U=0.05, label="0.05", labelpos="S")

    meanTrefoil = {}
    for i in (1, 2):
        binning = meanify(binSpacing)
        binning.add_field(coords, table[f"trefoil{i}"])
        binning.meanify()
        meanTrefoil[i] = binning.params0

    meanTrefoilAngle = np.arctan2(meanTrefoil[2], meanTrefoil[1]) / 3  # spin-3
    meanTrefoilAmplitude = np.hypot(meanTrefoil[2], meanTrefoil[1])
    scaleTriangle = 500
    # axs[1].scatter(250, 250, s=0.1 * scaleTriangle, marker=(3, 0, 30), lw=0.1, color="black")
    # axs[1].text(270, 250, "0.1")
    axs[1].scatter(0.8, 0.86, s=0.1 * scaleTriangle, marker=(3, 0, 30), lw=0.1, color="black", transform=axs[1].transAxes)
    axs[1].text(0.82, 0.85, "0.1", transform=axs[1].transAxes)
    for idx in range(len(meanTrefoilAmplitude)):
        tVal = meanTrefoilAmplitude[idx]
        trefoilAngleDeg = meanTrefoilAngle[idx] * 180 / np.pi
        xCenter = binning.coords0[idx, 0]
        yCenter = binning.coords0[idx, 1]
        axs[1].scatter(
            xCenter,
            yCenter,
            marker=(3, 0, 30 + trefoilAngleDeg),
            s=tVal * scaleTriangle,
            lw=0.1,
            color="black",
        )

    pos = axs[1].get_position()  # get current position [left, bottom, width, height]
    axs[1].set_position([pos.x0, pos.y0, pos.width * 0.931, pos.height])

    textKwargs = {
        "x": 0.95,
        "y": 0.95,
        "ha": "right",
        "va": "top",
        "fontsize": 9,
        "font": "monospace",
    }

    # Kurtosis hist
    axs[2].hist(kurtosis, bins=int(np.sqrt(len(table))), color="C3")

    kurtosisQuartiles = np.nanpercentile(kurtosis, [25, 50, 75])

    s = "Kurtosis\n"
    s += f"25%: {kurtosisQuartiles[0]:.3f}\n"
    s += f"50%: {kurtosisQuartiles[1]:.3f}\n"
    s += f"75%: {kurtosisQuartiles[2]:.3f}\n"

    # posizione della mediana rispetto all'intervallo dell'asse x
    xmin, xmax = axs[2].get_xlim()
    xmedian = (kurtosisQuartiles[1] - xmin) / (xmax - xmin)

    # se la linea mediana è nella zona occupata dal label,
    # sposta il label a sinistra
    if xmedian > 0.75:
        textKwargs["x"] = 0.05
        textKwargs["ha"] = "left"

    axs[2].text(
        s=s,
        transform=axs[2].transAxes,
        **textKwargs,
    )

    axs[2].axvline(kurtosisQuartiles[0], color="grey", lw=1)
    axs[2].axvline(kurtosisQuartiles[1], color="k", lw=2)
    axs[2].axvline(kurtosisQuartiles[2], color="grey", lw=1)

    # axs[2].set_xlim([min(kurtosis), max(kurtosis)])

    axs[0].text(0.05, 0.92, "coma", transform=axs[0].transAxes, fontsize=10)
    axs[1].text(0.82, 0.92, "trefoil", transform=axs[1].transAxes, fontsize=10)
