# Import packages  Image Access
import lsst.daf.butler as dafButler
import lsst.geom as geom
import lsst.afw.display as afwDisplay
import lsst.daf.base as dafBase
from lsst.daf.butler import Butler
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.afw.geom.ellipses import Quadrupole
from lsst.geom import Point2D, LinearTransform
import numpy as np
import pandas as pd
from rotation_conversion import rsp_to_rtp
from lsst.afw import cameraGeom
import matplotlib.pyplot as plt
import gc
from atm_dispersion_correction_per_vittorio_v2 import compute_atm_dispersion

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
                                rotation_sticks=1, do_flip=True, do_ad_correction=True, 
                                zenith_angle=0., pressure=727, temperature=6.85, n_grid=200, fileout=''):

    #rotation_sticks= 1: rotazione di entità +rottelpos_radians degli sticks
    #rotation_sticks= 0: nessuna rotazione degli sticks
    #rotation_sticks=-1: rotazione di entità -rottelpos_radians degli sticks
    #rotation_sticks=2: rotazione alla Josh con Quadrupole

    det = calexp.getDetector()
    wcs = calexp.getWcs()
    visit_id = calexp.info.getVisitInfo().getId()

    passband = calexp.getInfo().getFilter().bandLabel
    
    rotskypos = (calexp.info.getVisitInfo().getBoresightRotAngle()).asDegrees()
    rottelpos = rsp_to_rtp(rotskypos, \
            (calexp.info.getVisitInfo().getBoresightRaDec())[0].asDegrees(), \
            (calexp.info.getVisitInfo().getBoresightRaDec())[1].asDegrees(), \
            calexp.info.getVisitInfo().getDate().toAstropy()).deg
    rottelpos_radians = np.radians(rottelpos)

    if rotation_sticks>=1:
        rottelpos_radians_for_ellipticitysticks = rottelpos_radians
    elif rotation_sticks==0:
        rottelpos_radians_for_ellipticitysticks = 0.
    elif rotation_sticks==-1:
        rottelpos_radians_for_ellipticitysticks = -rottelpos_radians

    crtp = np.cos(rottelpos_radians)
    srtp = np.sin(rottelpos_radians)

    # aa sta per Alt-Azimuth, trasformazione da codice di Josh:
    # https://github.com/lsst-sitcom/summit_extras/blob/main/python/lsst/summit/extras/plotting/psfPlotting.py
    aaRot = np.array([[crtp, srtp], [-srtp, crtp]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
    transform_for_ell = LinearTransform(aaRot)

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

        # xx_fpposition = sources['base_FPPosition_x']
        # yy_fpposition = sources['base_FPPosition_y']

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
        x0, y0 = np.asarray(cam_x[0]), np.asarray(cam_y[0])
        # Rotazioni mie
        # xx_rot = x0*crtp - y0*srtp
        # yy_rot = x0*srtp + y0*crtp
        # Rotazioni Josh
        xx_rot = aaRot[0, 0] * x0 + aaRot[0, 1] * y0
        yy_rot = aaRot[1, 0] * x0 + aaRot[1, 1] * y0

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

    table_moments = {'Ixx': i_xx, 'Iyy': i_yy, 'Ixy': i_xy}

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
    # --- Con inversione XY degli stick--- OBSOLETE!!!!
        # Rotazioni mie
        ey_star_dvcs = ex
        ex_star_dvcs = ey
        crtp_e = np.cos(rottelpos_radians_for_ellipticitysticks)
        srtp_e = np.sin(rottelpos_radians_for_ellipticitysticks)
        ey_rot_star_dvcs = ex*crtp_e - ey*srtp_e
        ex_rot_star_dvcs = ex*srtp_e + ey*crtp_e
    else:
    # --- Senza inversione XY degli stick--- DEFAULT!!!!
        ey_star_dvcs = ey
        ex_star_dvcs = ex

        # Rotazioni mie
        # ex_rot_star_dvcs = ex*crtp_e - ey*srtp_e
        # ey_rot_star_dvcs = ex*srtp_e + ey*crtp_e

        if rotation_sticks==2:
            # Rotazioni Josh (Quadrupole)
            rot_shapes = []
            for i_xx1, i_yy1, i_xy1 in zip(i_xx, i_yy, i_xy):
                shape = Quadrupole(i_xx1, i_yy1, i_xy1)
                rot_shapes.append(shape.transform(transform_for_ell))
                
            aaIxx = np.asarray([sh.getIxx() for sh in rot_shapes])
            aaIyy = np.asarray([sh.getIyy() for sh in rot_shapes])
            aaIxy = np.asarray([sh.getIxy() for sh in rot_shapes])

            if do_ad_correction:
                aaIyy = aaIyy - compute_atm_dispersion(zenith_angle, passband, pression=pressure, temperature=temperature)
            
            e1_rot_star_dvcs = (aaIxx - aaIyy) / (aaIxx + aaIyy)
            e2_rot_star_dvcs = 2 * aaIxy / (aaIxx + aaIyy)    
            
            theta_josh = np.arctan2(e2_rot_star_dvcs, e1_rot_star_dvcs) / 2.
            e_star = np.sqrt(e1_rot_star_dvcs**2 + e2_rot_star_dvcs**2)
            ex_rot_star_dvcs = e_star * np.cos(theta_josh)
            ey_rot_star_dvcs = e_star * np.sin(theta_josh)
        else:
            # Rotazioni con aaRot, tutte sbagliate???
            ex_rot_star_dvcs = aaRot[0, 0] * ex + aaRot[0, 1] * ey
            ey_rot_star_dvcs = aaRot[1, 0] * ex + aaRot[1, 1] * ey
    
    theta_star_dvcs = np.arctan2(ey, ex)

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
            aaIxx, aaIxy, aaIyy, e1, e2, xx_star_dvcs, yy_star_dvcs, theta_star_dvcs, \
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
            aaIxx, aaIxy, aaIyy, e1, e2, xx_star_dvcs, yy_star_dvcs, theta_star_dvcs, \
            xx_rot_star_dvcs, yy_rot_star_dvcs, ra_star_dvcs, dec_star_dvcs, fwhm, size, fluxes

def plot_ellipticitymap(x, y, ex, ey, e, visitid_complete, fileout, figure_size_degrees=.5, clim_min=0., clim_max=1., scale=.5):
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
        plt.title('Ellipticity Sticks {:13d}'.format(visitid_complete))
        fig.savefig(fileout)
        remove_figure(fig)
