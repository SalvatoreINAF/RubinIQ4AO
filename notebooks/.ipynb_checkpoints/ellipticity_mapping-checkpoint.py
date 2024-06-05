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
    
def calculate_ellipticity_on_xy(calexp, sources, psf, regular_grid_or_star_positions, n_grid=200, fileout=''):

    det = calexp.getDetector()
    wcs = calexp.getWcs()
    visit_id = calexp.info.getVisitInfo().getId()
    
    rotskypos = (calexp.info.getVisitInfo().getBoresightRotAngle()).asDegrees()
    rottelpos = rsp_to_rtp(rotskypos, calexp.getMetadata().getDouble('RA'), 
                           calexp.getMetadata().getDouble('DEC'),
                           calexp.getMetadata().getDouble('MJD')).deg
    rottelpos_radians = np.radians(rottelpos)
    
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

    elif regular_grid_or_star_positions == 2:
        xx_for_zip = [2000.]
        yy_for_zip = [2000.]
        xxshape = len(xx_for_zip)

    print(xxshape)
    
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
        xx_star_dvcs.append(cam_y[0]) #IMPORTANTE INVERTIRE XY TRA CCS E DVCS
        yy_star_dvcs.append(cam_x[0]) #IMPORTANTE INVERTIRE XY TRA CCS E DVCS
        xx_rot_star_dvcs.append(yy_rot) #IMPORTANTE INVERTIRE XY TRA CCS E DVCS
        yy_rot_star_dvcs.append(xx_rot) #IMPORTANTE INVERTIRE XY TRA CCS E DVCS
        ra_star_dvcs.append(coo[1].asDegrees())
        dec_star_dvcs.append(coo[0].asDegrees())

    size = np.reshape(size, xxshape)
    i_xx = np.reshape(i_xx, xxshape)
    i_yy = np.reshape(i_yy, xxshape)
    i_xy = np.reshape(i_xy, xxshape)

    theta = np.arctan2(2. * i_xy, i_xx - i_yy) / 2.

    e1 = (i_xx - i_yy) / (i_xx + i_yy)
    e2 = (2. * i_xy) / (i_xx + i_yy)
    
    theta_alternate = np.arctan2(e2, e1) / 2.
    assert np.allclose(theta, theta_alternate)

    e_star = np.sqrt(e1**2 + e2**2)
    ex = e_star * np.cos(theta)
    ey = e_star * np.sin(theta)
    ey_star_dvcs = ex #IMPORTANTE INVERTIRE XY TRA CCS e DVCS
    ex_star_dvcs = ey #IMPORTANTE INVERTIRE XY TRA CCS e DVCS
    ey_rot_star_dvcs = ex*np.cos(rottelpos_radians) - ey*np.sin(rottelpos_radians) #IMPORTANTE INVERTIRE XY TRA CCS e DVCS
    ex_rot_star_dvcs = ex*np.sin(rottelpos_radians) + ey*np.cos(rottelpos_radians) #IMPORTANTE INVERTIRE XY TRA CCS e DVCS
    theta_star_dvcs = np.arctan2(ex, ey) #IMPORTANTE INVERTIRE XY TRA CCS e DVCS
    
    fwhm = []
    # FWHM
    for point in points:
        sigma = psf.computeShape(point).getDeterminantRadius()
        pixelScale = calexp.getWcs().getPixelScale().asArcseconds()
        fwhm_temp = sigma * pixelScale * 2.355
        fwhm.append(fwhm_temp)
    
    if fileout != '':
        df = pd.DataFrame(data={'x_pixel_ccs': xx_for_zip, 'y_pixel_ccs': yy_for_zip, 'e_star': e_star, 
                               'ex_star_dvcs': ex_star_dvcs, 'ey_star_dvcs': ey_star_dvcs, 
                                'ex_rot_star_dvcs': ex_rot_star_dvcs, 'ey_rot_star_dvcs': ey_rot_star_dvcs, 
                                'e1': e1, 'e2': e2, 'theta_star_dvcs': theta_star_dvcs,
                                'xx_star_dvcs': xx_star_dvcs, 'yy_star_dvcs': yy_star_dvcs, 
                                'xx_rot_star_dvcs': xx_rot_star_dvcs, 'yy_rot_star_dvcs': yy_rot_star_dvcs, 
                                'ra_star_dvcs': ra_star_dvcs, 'dec_star_dvcs': dec_star_dvcs,
                               'theta_alternate':theta_alternate, 'fwhm': fwhm, 'detector': [det] * len(xx_for_zip), 
                               'visit_id':  [visit_id] * len(xx_for_zip)})
        df.to_csv(fileout, index=None)
    
    return e_star, ex_star_dvcs, ey_star_dvcs, ex_rot_star_dvcs, ey_rot_star_dvcs, \
        e1, e2, xx_star_dvcs, yy_star_dvcs, theta_star_dvcs, \
        xx_rot_star_dvcs, yy_rot_star_dvcs, ra_star_dvcs, dec_star_dvcs, fwhm, size