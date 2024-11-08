# Import packages  Image Access
import lsst.daf.butler as dafButler
import lsst.geom as geom
import lsst.afw.display as afwDisplay
import lsst.daf.base as dafBase
from lsst.daf.butler import Butler
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.geom import Point2D
import numpy as np
import pandas as pd
from rotation_conversion import rsp_to_rtp
from lsst.afw import cameraGeom
import matplotlib.pyplot as plt
import gc

def remove_figure(fig):
    """
    Remove a figure to reduce memory footprint.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        Figure to be removed.

    Returns
    -------
    None
    """
    # get the axes and clear their images
    for ax in fig.get_axes():
        for im in ax.get_images():
            im.remove()
    fig.clf()       # clear the figure
    plt.close(fig)  # close the figure
    gc.collect()    # call the garbage collector
    
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

def calculate_ellipticity_on_xy(calexp, sources, psf, regular_grid_or_star_positions, 
                                rotation_sticks=1, do_flip=True, n_grid=200, fileout=''):

    #rotation_sticks= 1: rotazione di entità +rottelpos_radians degli sticks
    #rotation_sticks= 0: nessuna rotazione degli sticks
    #rotation_sticks=-1: rotazione di entità -rottelpos_radians degli sticks

    det = calexp.getDetector()
    wcs = calexp.getWcs()
    visit_id = calexp.info.getVisitInfo().getId()
    
    rotskypos = (calexp.info.getVisitInfo().getBoresightRotAngle()).asDegrees()
    rottelpos = rsp_to_rtp(rotskypos, \
            (calexp.info.getVisitInfo().getBoresightRaDec())[0].asDegrees(), \
            (calexp.info.getVisitInfo().getBoresightRaDec())[1].asDegrees(), \
            calexp.info.getVisitInfo().getDate().toAstropy()).deg
    rottelpos_radians = np.radians(rottelpos)

    if rotation_sticks==1:
        rottelpos_radians_for_ellipticitysticks = rottelpos_radians
    elif rotation_sticks==0:
        rottelpos_radians_for_ellipticitysticks = 0.
    elif rotation_sticks==-1:
        rottelpos_radians_for_ellipticitysticks = -rottelpos_radians

    # ------------Get the points on grid/star positions (in CCS)------------
    if regular_grid_or_star_positions == 0:
        # Per la visualizzazione della PSF su griglia regolare
        grid_separation_x = calexp.getDimensions()[0] / n_grid
        grid_separation_y = calexp.getDimensions()[1] / n_grid
        x_array = np.arange(n_grid)*grid_separation_x + grid_separation_x/2.
        y_array = np.arange(n_grid)*grid_separation_y + grid_separation_y/2.
        xx, yy = np.meshgrid(x_array, y_array)
        xx_for_zip = xx.flatten()
        yy_for_zip = yy.flatten()
        xxshape = n_grid*n_grid

    elif regular_grid_or_star_positions == 1:
        # Per la visualizzazione della PSF sulle coordinate delle stelle
        xx = [l.getCentroid()[0] for l in sources]
        yy = [l.getCentroid()[1] for l in sources]
        xx_for_zip = xx
        yy_for_zip = yy
        xxshape = len(xx)
        fluxes = [l.getPsfInstFlux() for l in sources]        

    elif regular_grid_or_star_positions == 2:
        xx_for_zip = [2000.]
        yy_for_zip = [2000.]
        xxshape = len(xx_for_zip)


    # ------------convert CCS into DVCS and extract moments------------
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

    for x, y in zip(xx_for_zip, yy_for_zip):
        point = Point2D(x, y)        
        coo = wcs.pixelToSky(x, y)
        cam_x, cam_y = pixel_to_camera_angle(point[0], point[1], det)
        xx_rot = np.asarray(cam_x[0])*np.cos(rottelpos_radians) - \
                                np.asarray(cam_y[0])*np.sin(rottelpos_radians)
        yy_rot = np.asarray(cam_x[0])*np.sin(rottelpos_radians) + \
                                np.asarray(cam_y[0])*np.cos(rottelpos_radians)

        shape = psf.computeShape(point)
        size.append(shape.getTraceRadius())
        i_xx.append(shape.getIxx())
        i_yy.append(shape.getIyy())
        i_xy.append(shape.getIxy())
        points.append(point)

        if do_flip:
            xx_star_dvcs.append(cam_y[0])
            yy_star_dvcs.append(cam_x[0])
            xx_rot_star_dvcs.append(yy_rot)
            yy_rot_star_dvcs.append(xx_rot)
        else:
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

    # ------------Transform moments into ellipticities------------
    theta = np.arctan2(2. * i_xy, i_xx - i_yy) / 2.

    e1 = (i_xx - i_yy) / (i_xx + i_yy)
    e2 = (2. * i_xy) / (i_xx + i_yy)
    
    theta_alternate = np.arctan2(e2, e1) / 2.
    assert np.allclose(theta, theta_alternate)

    e_star = np.sqrt(e1**2 + e2**2)
    ex = e_star * np.cos(theta)
    ey = e_star * np.sin(theta)

    if do_flip:
    # --- Con rotazione e inversione XY degli stick--- (_try0)
        ey_star_dvcs = ex
        ex_star_dvcs = ey
        ey_rot_star_dvcs = ex*np.cos(rottelpos_radians_for_ellipticitysticks) - ey*np.sin(rottelpos_radians_for_ellipticitysticks)
        ex_rot_star_dvcs = ex*np.sin(rottelpos_radians_for_ellipticitysticks) + ey*np.cos(rottelpos_radians_for_ellipticitysticks)
    else:
    # --- Con rotazione negativa senza inversione XY degli stick--- (_try1)
        ey_star_dvcs = ey
        ex_star_dvcs = ex
        ex_rot_star_dvcs = ex*np.cos(rottelpos_radians_for_ellipticitysticks) - ey*np.sin(rottelpos_radians_for_ellipticitysticks)
        ey_rot_star_dvcs = ex*np.sin(rottelpos_radians_for_ellipticitysticks) + ey*np.cos(rottelpos_radians_for_ellipticitysticks)
    
    theta_star_dvcs = np.arctan2(ey, ex)

    # # --- Senza rotazione e inversione XY degli stick---
    # ey_star_dvcs = ey
    # ex_star_dvcs = ex
    # ex_rot_star_dvcs = ex*np.cos(rottelpos_radians) - ey*np.sin(rottelpos_radians)
    # ey_rot_star_dvcs = ex*np.sin(rottelpos_radians) + ey*np.cos(rottelpos_radians)
    # theta_star_dvcs = np.arctan2(ey, ex)
    
    fwhm = []
    # FWHM
    for point in points:
        sigma = psf.computeShape(point).getDeterminantRadius()
        pixelScale = calexp.getWcs().getPixelScale().asArcseconds()
        fwhm_temp = sigma * pixelScale * 2.355
        fwhm.append(fwhm_temp)
    
    if (regular_grid_or_star_positions == 0) | (regular_grid_or_star_positions == 2):
        if fileout != '':
            df = pd.DataFrame(data={'x_pixel_ccs': xx_for_zip, 'y_pixel_ccs': yy_for_zip, 'e_star': e_star, 
                               'ex_star_dvcs': ex_star_dvcs, 'ey_star_dvcs': ey_star_dvcs, 
                                'ex_rot_star_dvcs': ex_rot_star_dvcs, 'ey_rot_star_dvcs': ey_rot_star_dvcs, 
                                'i_xx': i_xx, 'i_yy': i_yy, 'i_xx': i_xy,  
                                'e1': e1, 'e2': e2, 'theta_star_dvcs': theta_star_dvcs,
                                'xx_star_dvcs': xx_star_dvcs, 'yy_star_dvcs': yy_star_dvcs, 
                                'xx_rot_star_dvcs': xx_rot_star_dvcs, 'yy_rot_star_dvcs': yy_rot_star_dvcs, 
                                'ra_star_dvcs': ra_star_dvcs, 'dec_star_dvcs': dec_star_dvcs,
                               'theta_alternate':theta_alternate, 'fwhm': fwhm, 'detector': [det.getId()] * len(xx_for_zip), 
                               'visit_id':  [visit_id] * len(xx_for_zip)})
            df.to_csv(fileout, index=None)
            # df['theta_rot_star_dvcs'] = np.degrees(np.arctan2(ex_rot_star_dvcs, ey_rot_star_dvcs))
            # df[['xx_rot_star_dvcs', 'yy_rot_star_dvcs', 'e_star', 'theta_rot_star_dvcs']].to_csv(fileout+'_for_ricardo', index=None)
    
        return e_star, ex_star_dvcs, ey_star_dvcs, ex_rot_star_dvcs, ey_rot_star_dvcs, i_xx, i_yy, i_xy, \
            e1, e2, xx_star_dvcs, yy_star_dvcs, theta_star_dvcs, \
            xx_rot_star_dvcs, yy_rot_star_dvcs, ra_star_dvcs, dec_star_dvcs, fwhm, size

    elif regular_grid_or_star_positions == 1:
        if fileout != '':
            df = pd.DataFrame(data={'x_pixel_ccs': xx_for_zip, 'y_pixel_ccs': yy_for_zip, 'e_star': e_star, 
                               'ex_star_dvcs': ex_star_dvcs, 'ey_star_dvcs': ey_star_dvcs, 
                                'ex_rot_star_dvcs': ex_rot_star_dvcs, 'ey_rot_star_dvcs': ey_rot_star_dvcs, 
                                'i_xx': i_xx, 'i_yy': i_yy, 'i_xx': i_xy,  
                                'e1': e1, 'e2': e2, 'theta_star_dvcs': theta_star_dvcs,
                                'xx_star_dvcs': xx_star_dvcs, 'yy_star_dvcs': yy_star_dvcs, 
                                'xx_rot_star_dvcs': xx_rot_star_dvcs, 'yy_rot_star_dvcs': yy_rot_star_dvcs, 
                                'ra_star_dvcs': ra_star_dvcs, 'dec_star_dvcs': dec_star_dvcs,
                               'theta_alternate':theta_alternate, 'fwhm': fwhm, 'fluxes': fluxes, 'detector': [det.getId()] * len(xx_for_zip), 
                               'visit_id':  [visit_id] * len(xx_for_zip)})
            df.to_csv(fileout, index=None)
            # df['theta_rot_star_dvcs'] = np.degrees(np.arctan2(ex_rot_star_dvcs, ey_rot_star_dvcs))
            # df[['xx_rot_star_dvcs', 'yy_rot_star_dvcs', 'e_star', 'theta_rot_star_dvcs']].to_csv(fileout+'_for_ricardo', index=None)
    
        return e_star, ex_star_dvcs, ey_star_dvcs, ex_rot_star_dvcs, ey_rot_star_dvcs, i_xx, i_yy, i_xy, \
            e1, e2, xx_star_dvcs, yy_star_dvcs, theta_star_dvcs, \
            xx_rot_star_dvcs, yy_rot_star_dvcs, ra_star_dvcs, dec_star_dvcs, fwhm, size, fluxes

def plot_ellipticitymap(x, y, ex, ey, e, fileout, figure_size_degrees=.5, clim_min=0., clim_max=1., scale=.5):
        fig = plt.figure(figsize=(10,8))
        plt.quiver(x, y, ex, ey, e, scale=scale, headlength=0., headwidth=1., pivot='mid', linewidths=.01)

        colorbar = plt.colorbar(label='ellipticity')

        if not 'clim_min' in locals():
            clim_min=min(e)
        if not 'clim_max' in locals():
            clim_max=max(e)
        plt.clim(clim_min, clim_max)
        plt.xlim([-figure_size_degrees,figure_size_degrees])
        plt.ylim([-figure_size_degrees,figure_size_degrees])
        plt.xlabel('x [deg]')
        plt.ylabel('y [deg]')
        plt.title('Ellipticity Sticks')
        fig.savefig(fileout)
        remove_figure(fig)
    