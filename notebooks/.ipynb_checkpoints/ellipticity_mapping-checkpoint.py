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

def calculate_ellipticity_on_xy(calexp, sources, psf, regular_grid_or_star_positions, n_grid, fileout=''):

    rot = (calexp.info.getVisitInfo().getBoresightRotAngle()).asDegrees()
        
    if regular_grid_or_star_positions == 0:
        # Per la visualizzazione della PSF su griglia regolare
        n_grid = n_grid
        x_array = np.arange(0, calexp.getDimensions()[0], n_grid)
        y_array = np.arange(0, calexp.getDimensions()[1], n_grid)
        xx, yy = np.meshgrid(x_array, y_array)
        xx_for_zip = xx.flatten()
        yy_for_zip = yy.flatten()
        xxshape = xx.shape

    elif regular_grid_or_star_positions == 1:
        # Per la visualizzazione della PSF sulle coordinate delle stelle
        xx = [l.getCentroid()[0] for l in sources]
        yy = [l.getCentroid()[1] for l in sources]
        xx_for_zip = xx
        yy_for_zip = yy
        xxshape = len(xx)
        
    size = []
    i_xx = []
    i_yy = []
    i_xy = []
    points = []

    for x, y in zip(xx_for_zip, yy_for_zip):
        point = Point2D(x, y)
        shape = psf.computeShape(point)
        size.append(shape.getTraceRadius())
        i_xx.append(shape.getIxx())
        i_yy.append(shape.getIyy())
        i_xy.append(shape.getIxy())
        points.append(point)
        
    size = np.reshape(size, xxshape)
    i_xx = np.reshape(i_xx, xxshape)
    i_yy = np.reshape(i_yy, xxshape)
    i_xy = np.reshape(i_xy, xxshape)

    theta = np.arctan2(2. * i_xy, i_xx - i_yy) / 2.
    e1 = (i_xx - i_yy) / (i_xx + i_yy)
    e2 = (2. * i_xy) / (i_xx + i_yy)
    
    theta_alternate = np.arctan2(e2, e1) / 2.
    assert np.allclose(theta, theta_alternate)

    e = np.sqrt(e1**2 + e2**2)
    ex = e * np.cos(theta)
    ey = e * np.sin(theta)
    ex_rot = ex*np.cos(np.radians(rot)) - ey*np.sin(np.radians(rot))
    ey_rot = ex*np.sin(np.radians(rot)) + ey*np.cos(np.radians(rot))
    
    fwhm = []
    # FWHM
    for point in points:
        sigma = psf.computeShape(point).getDeterminantRadius()
        pixelScale = calexp.getWcs().getPixelScale().asArcseconds()
        fwhm_temp = sigma * pixelScale * 2.355
        fwhm.append(fwhm_temp)
    
    if fileout != '':
        df = pd.DataFrame(data={'x': xx_for_zip, 'y': yy_for_zip, 'e': e, 
                               'ex': ex, 'ey': ey, 'ex_rot': ex_rot, 'ey_rot': ey_rot, 'e1': e1, 'e2': e2, 'theta': theta,
                               'theta_alternate':theta_alternate, 'fwhm': fwhm})
        df.to_csv(fileout, index=None)
    
    return e, ex, ey, ex_rot, ey_rot, e1, e2, xx, yy, theta, fwhm, size