#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:13:17 2024

@author: zanmar
"""


from fromBatoid import FromBatoid
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import argparse

def regular_grid( radius ):
    
    x = np.linspace(-radius*0.95, radius*0.95, num=20 )
    xx, yy = np.meshgrid( x, x )
    
    ikeep = np.sqrt( xx**2 + yy**2 ) < radius
    
    return xx[ikeep].flatten(), yy[ikeep].flatten()
    
def plot_c4( fb ):
    x1 = np.linspace( -fb.field_radius, fb.field_radius, num = 50 )
    y1 = np.zeros( x1.shape )
    fb.hx, fb. hy = x1, y1
    znkarrX = fb.zernike_coeffs()
    c4arrX = znkarrX[:, 3]
    
    fb.hx, fb. hy = y1, x1      # now along y.
    znkarrY = fb.zernike_coeffs()
    c4arrY = znkarrY[:,3]
    
    fig1, ax = plt.subplots()
    ax.set_title('C4 [defocus]')
    ax.set_xlabel('x or y [m]')
    ax.set_ylabel('c4 [waves]')
    ax.plot( x1, c4arrX, '.' )
    ax.plot( x1, c4arrY, 'x')
    
    return

def plot_one_psf( fb, i ):

    psf_list = fb.psf
    one = psf_list[i]
    psfnorm = one.array / one.array.max()
    
    sl, ss = fb.ellipticity['sl'], fb.ellipticity['ss']
    el, pa = fb.ellipticity['el'], fb.ellipticity['pa']
    xc, yc = fb.ellipticity['xc'], fb.ellipticity['yc']  

      
    print( sl[i], ss[i], pa[i] )
    
    fig, ax = plt.subplots()
    im = ax.pcolormesh(one.coords[:,:,1], one.coords[:,:,0], 
                  psfnorm, 
                  cmap='jet')
    # ax.autoscale(False)
    # ax.axis('equal')
    ax.set_aspect('equal', 'box')
    kk = 0.5e-5
    ax.set_xlim((-2*kk, 2*kk))         # set ROI to -1,1 for both x,y
    ax.set_ylim((-2*kk, 2*kk))
    ax.set_title('PSF batoid (%.1f,%.1f)' %(fb.hx[i],fb.hy[i]) )
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    fig.colorbar(im, ax=ax)
    
    ellip = patches.Ellipse((xc[i],yc[i]), (sl[i]*2), (ss[i]*2), angle=pa[i], linewidth=1, edgecolor='r', facecolor='none' )

    #ax.add_patch(rect)
    ax.add_patch(ellip)
    
    return

def pars2dof():
    parser = argparse.ArgumentParser()

    """
                AOS degrees of freedom.
            0,1,2 are M2 z,x,y in micron
            3,4 are M2 rot around x, y in arcsec
            5,6,7 are camera z,x,y in micron
            8,9 are camera rot around x, y in arcsec
    """

    parser.add_argument('--M2z', help='M2 decenter z [-100, 100] micron', type=int, default=0, choices=range(-100,120,20) )
    parser.add_argument('--M2x', help='M2 decenter x [-3000, 3000] micron', type=int, default=0, choices=range(-3000,3600,600) )
    parser.add_argument('--M2y', help='M2 decenter y [-3000, 3000] micron', type=int, default=0, choices=range(-3000,3600,600) )
    parser.add_argument('--M2rx', help='M2 tilt x [-80, 80 ] arcsec', type=int, default=0, choices=range(-80,90,10) )
    parser.add_argument('--M2ry', help='M2 tilt y [-80, 80 ] arcsec', type=int, default=0, choices=range(-80,90,10) )

    parser.add_argument('--Cz', help='Camera decenter z [-100, 100] micron', type=int, default=0,  choices=range(-100,110,10) )
    parser.add_argument('--Cx', help='Camera decenter x [-3000, 3000] micron', type=int, default=0, choices=range(-3000,3600,600) )
    parser.add_argument('--Cy', help='Camera decenter y [-3000, 3000] micron', type=int, default=0, choices=range(-3000,3600,600) )
    parser.add_argument('--Crx', help='Camera tilt x [-80, 80 ] arcsec', type=int, default=0, choices=range(-80,90,10) )
    parser.add_argument('--Cry', help='Camera tilt y [-80, 80 ] arcsec', type=int, default=0, choices=range(-80,90,10) )


    args = parser.parse_args()

    dof = np.zeros(50)

    dof[0:5] = ( args.M2z, args.M2x, args.M2y, args.M2rx, args.M2ry )
    dof[5:10]= ( args.Cz, args.Cx, args.Cy, args.Crx, args.Cry )

    print( 'dof=', dof[0:10] )
    return dof

    
def plot_ellipticity_map( fb ):
    
    fig, ax = plt.subplots()
    
    U = fb.ellipticity['el'] * np.cos( np.deg2rad( fb.ellipticity['pa'] ) )
    V = fb.ellipticity['el'] * np.sin( np.deg2rad( fb.ellipticity['pa'] ) )
    M = np.hypot(U, V)
    ax.quiver(fb.hx, fb.hy, U, V, M, units='xy', scale=2, headwidth=1,
              headlength=0, headaxislength=0, pivot = 'middle',
              linewidth=0.8)
    ax.scatter( fb.hx, fb.hy, color='black', s=2)
    
    ax.set_xlim(-fb.field_radius, fb.field_radius )
    ax.set_ylim(-fb.field_radius, fb.field_radius )
    ax.set_aspect('equal', 'box')
    ax.set_title('ellipticity' )
    ax.set_xlabel('hx [deg]')
    ax.set_ylabel('hy [deg]')
    plt.show()
    
    
    
    
    

if( __name__ == '__main__'):
    fb = FromBatoid()
    
    x, y = regular_grid( fb.field_radius)
    
    fb.hx, fb.hy = x, y
    fb.wl_index = 1
    #fb.hx, fb.hy = np.array([0.55]), np.array([0.25]) 
    dof = pars2dof()

    fb.dof = dof
    

    fb.psfFFT_compute()
    fb.ellipticity_compute()
    
## PSF
    # plot_one_psf( fb, 100)

## ellipticity map
    plot_ellipticity_map( fb )
    
