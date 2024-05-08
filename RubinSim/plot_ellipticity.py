#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:13:17 2024

@author: zanmar
"""


from fromBatoid import FromBatoid
from util import rayPSF, regular_grid, plot_ellipticity_map
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import argparse
import lsstModel

    
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

    if( fb.psf_type == 'fft' ):
        
        
        sl, ss = fb.ellipticity['sl'], fb.ellipticity['ss']
        el, pa = fb.ellipticity['el'], fb.ellipticity['pa']
        xc, yc = fb.ellipticity['xc'], fb.ellipticity['yc']  
    
          
        print( sl[i], ss[i], pa[i] )
        
        fig, ax = plt.subplots()
        im = ax.pcolormesh(one.coords[:,:,1]/10e-6, one.coords[:,:,0]/10e-6, 
                      psfnorm)
                      # cmap='jet')
        # ax.autoscale(False)
        # ax.axis('equal')
        ax.set_aspect('equal', 'box')
        # kk = 10.0e-6  #size of one pixel
        kk = 1
        ax.set_xlim((-10.5*kk, 10.5*kk))         # set ROI to -1,1 for both x,y
        ax.set_ylim((-10.5*kk, 10.5*kk))
        ax.set_title('PSF fft batoid (%.1f,%.1f)' %(fb.hx[i],fb.hy[i]) )
        ax.set_xlabel('x [pix]')
        ax.set_ylabel('y [pix]')
        fig.colorbar(im, ax=ax)
        
        ellip = patches.Ellipse((xc[i],yc[i]), (sl[i]*2), (ss[i]*2), angle=pa[i], linewidth=1, edgecolor='r', facecolor='none' )
    
        #ax.add_patch(rect)
        ax.add_patch(ellip)
        
        plt.show()
        
        
    elif( fb.psf_type == 'ray' ):
        fig, ax = plt.subplots()
        im = ax.pcolormesh(one.coords[:,:,1]/10e-6, one.coords[:,:,0]/10e-6, 
                            psfnorm )
        # ax.autoscale(False)
        # ax.axis('equal')
        ax.set_aspect('equal', 'box')
        # kk = 0.5e-5
        # ax.set_xlim((-kk, kk))         # set ROI to -1,1 for both x,y
        # ax.set_ylim((-kk, kk))
        ax.set_title('PSF ray batoid (%.1f,%.1f)' %(fb.hx[i],fb.hy[i]))
        ax.set_xlabel('x [pix]')
        ax.set_ylabel('y [pix]')
        fig.colorbar(im, ax=ax)

        plt.show()
        
    return

def pars2dof():
    parser = argparse.ArgumentParser()
    parser.print_help()

    """
                AOS degrees of freedom.
            0,1,2 are M2 z,x,y in micron
            3,4 are M2 rot around x, y in arcsec
            5,6,7 are camera z,x,y in micron
            8,9 are camera rot around x, y in arcsec
    """

    parser.add_argument('--M2z', help='M2 decenter z [-80, 80] micron', type=int, default=0, choices=range(-80,90,10) )
    parser.add_argument('--M2x', help='M2 decenter x [-3000, 3000] micron', type=int, default=0, choices=range(-3000,3600,600) )
    parser.add_argument('--M2y', help='M2 decenter y [-3000, 3000] micron', type=int, default=0, choices=range(-3000,3600,600) )
    parser.add_argument('--M2rx', help='M2 tilt x [-80, 80 ] arcsec', type=int, default=0, choices=range(-80,90,10) )
    parser.add_argument('--M2ry', help='M2 tilt y [-80, 80 ] arcsec', type=int, default=0, choices=range(-80,90,10) )

    parser.add_argument('--Cz', help='Camera decenter z [-80, 80] micron', type=int, default=0,  choices=range(-80,90,10) )
    parser.add_argument('--Cx', help='Camera decenter x [-3000, 3000] micron', type=int, default=0, choices=range(-3000,3600,600) )
    parser.add_argument('--Cy', help='Camera decenter y [-3000, 3000] micron', type=int, default=0, choices=range(-3000,3600,600) )
    parser.add_argument('--Crx', help='Camera tilt x [-80, 80 ] arcsec', type=int, default=0, choices=range(-80,90,10) )
    parser.add_argument('--Cry', help='Camera tilt y [-80, 80 ] arcsec', type=int, default=0, choices=range(-80,90,10) )

    parser.add_argument('--dof10', help='bending mode i=10', type=float, default=0.0 )
    parser.add_argument('--dof11', help='bending mode i=10', type=float, default=0.0 )
    
    parser.add_argument('-m','--model', help="use model, not batoid", action="store_true")
    parser.add_argument('-i','--interactive', help="plot is interactive to inspect PSFs", action="store_true")
    
    args = parser.parse_args()

    dof = np.zeros(50)

    dof[0:5] = ( args.M2z, args.M2x, args.M2y, args.M2rx, args.M2ry )
    dof[5:10]= ( args.Cz, args.Cx, args.Cy, args.Crx, args.Cry )
    
    dof[10], dof[11] = args.dof10, args.dof11

    # print( 'dof=', dof[0:12] )
    return dof, args.model, args.interactive

def save_ellipticity_table( fname, fb ):
    U = fb.ellipticity['el'] * np.cos( np.deg2rad( fb.ellipticity['pa'] ) )
    V = fb.ellipticity['el'] * np.sin( np.deg2rad( fb.ellipticity['pa'] ) )
    M = np.hypot(U, V)
    
    np.savetxt(fname, np.transpose([fb.hx, fb.hy, U, V, M] ),
               delimiter=',', header='hx,hy,ex,ey,e' )
    # np.transpose([x,y*2,z/2]),delimiter=',',header='uno,dos,tres')
    print( 'saved to file: %s' %fname )
    return

class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, fig, ax, ax2, xs, ys, line ):
        self.lastind = 0
        self.fig = fig
        self.ax = ax
        self.ax2 = ax2
        self.xs = xs
        self.ys = ys
        self.line = line
        self.text = ax.text(0.04, 0.96, 'selected: none',
                            transform=ax.transAxes, va='top')
        
        self.selected, = self.ax.plot([self.xs[0]], [self.ys[0]], 'o', ms=12, alpha=0.5,
                                  color='orange', visible=False)

    def on_press(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p'):
            return
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(self.xs) - 1)
        self.update()

    def on_pick(self, event):

        if event.artist != self.line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.xs[event.ind], y - self.ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        self.ax2.clear()
        # ax2.plot(X[dataind])
        
        psf_list = fb.psf
        one = psf_list[dataind]
        psfnorm = one.array / one.array.max()
        
        im = self.ax2.pcolormesh(one.coords[:,:,1]/10e-6, one.coords[:,:,0]/10e-6, 
                            psfnorm )
        # ax.autoscale(False)
        # ax.axis('equal')
        self.ax2.set_aspect('equal', 'box')
        # kk = 0.5e-5
        # ax.set_xlim((-kk, kk))         # set ROI to -1,1 for both x,y
        # ax.set_ylim((-kk, kk))
        self.ax2.set_title('PSF ray batoid')
        self.ax2.set_xlabel('x [pix]')
        self.ax2.set_ylabel('y [pix]')
#        fig.colorbar(im, ax=ax)
        
        
        
        
        
        
        
        
        # annotate eventually

        # ax2.text(0.05, 0.9, f'mu={xs[dataind]:1.3f}\nsigma={ys[dataind]:1.3f}',
        #          transform=ax2.transAxes, va='top')
        # ax2.set_ylim(-0.5, 1.5)
 
        
        self.selected.set_visible(True)
        self.selected.set_data(self.xs[dataind], self.ys[dataind])

        self.text.set_text('(%.3f,%.1f)' %(fb.ellipticity['el'][dataind],
                                           fb.ellipticity['pa'][dataind]))
        self.fig.canvas.draw()

def plot_interactive( fb ):
    
    fig, (ax, ax2) = plt.subplots(1,2)
    
    U = fb.ellipticity['el'] * np.cos( np.deg2rad( fb.ellipticity['pa'] ) )
    V = fb.ellipticity['el'] * np.sin( np.deg2rad( fb.ellipticity['pa'] ) )
    M = np.hypot(U, V)
    
    xs=fb.hx
    ys=fb.hy
    
    
    line = ax.quiver(xs, ys, U, V, M, units='xy', scale=1.5, headwidth=1,
              headlength=0, headaxislength=0, pivot = 'middle',
              linewidth=0.8, picker=True, pickradius=20 )
    # ax.scatter( fb.hx, fb.hy, color='black', s=1)
    
    ax.set_xlim(-fb.field_radius, fb.field_radius )
    ax.set_ylim(-fb.field_radius, fb.field_radius )
    ax.set_aspect('equal', 'box')
    ax.set_title('ellipticity' )
    ax.set_xlabel('hx [deg]')
    ax.set_ylabel('hy [deg]')

    browser = PointBrowser( fig, ax, ax2, xs, ys, line )

    fig.canvas.mpl_connect('pick_event', browser.on_pick)
    fig.canvas.mpl_connect('key_press_event', browser.on_press)

    plt.show()    
    
def grid_from_file( fname ):
    arr = np.loadtxt( fname, delimiter=',', skiprows=1 )
    
    imsim_data={'hx':np.rad2deg( arr[:,2]), 'hy': np.rad2deg(arr[:,3] ),
              'ex':arr[:,5], 'ey':arr[:,6] }
    
    return imsim_data['hx'], imsim_data['hy']
    

if( __name__ == '__main__'):

    manypoints = True
    dof, with_model, interactive = pars2dof()
    # dof = np.zeros(50)
    # dof[0] = -10
    # # dof[1] = 000
    # # dof[2] = 000
    # # dof[3] = 20
    # # dof[4] = 10

    # dof[10] = 0.2
    # dof[11] = -0.05

    print( 'dof=', dof[0:12] )

    if( with_model ):
        # fname = 'znk_batoid_coeffs_wl_2_jmax_22_dbg.hdf5'
        fname = 'znk_batoid_coeffs_wl_6_jmax_22.hdf5'
        # fname = 'znk_batoid_coeffs_wl_2_jmax_37_dbg.hdf5'
        # fname = 'znk_batoid_coeffs_wl_6_jmax_37.hdf5'

        lm = lsstModel.lsstModel( batoid_cube_file=fname, n_fldznk=22)
        lm.wl_index = 1
        lm.dof = dof[0:10]
        
        fov = lm.field_radius
    else:
        fb = FromBatoid()
        fb.wl_index = 1
        fb.dof = dof
        
        fov = fb.field_radius
        



    
    if( manypoints ):
        # fname = 'data/mean_ellipticities_20240301T131538.csv'
        # fname = 'data/ellipticities_focalplane_seqnum0059.csv'  
        # y, x = grid_from_file( fname )                  ###x,y exchanged!
        x, y = regular_grid( fov )
        if( with_model ):
            lm.set_hxhy(x, y)
        else:
            fb.hx, fb.hy = x, y
            
        # fb.hx, fb.hy = np.array([1.]), np.array([0])
    else:
        if( with_model ):
            lm.set_hxhy( np.array([1.0]), np.array([0.5]) )
        else:
            fb.hx, fb.hy = np.array([1.0]), np.array([0.5]) 


    if( with_model ):
        lm.psf_compute()
        lm.ellipticity_compute()
        ellipticity_dic = lm.ellipticity
        
    else:
        fb.psf_compute( psf_type = 'fft', seeing=0.5 )
        fb.ellipticity_compute()
        ellipticity_dic = fb.ellipticity
        ## PSF
       
    if( interactive ):
        if( with_model ):
            print( 'interavtive mode not implemented yet for model')
            pass
            
        else:    
            plot_interactive( fb )
    else:      
        if( manypoints ):
            # ellipticity map
            plot_ellipticity_map( x, y, ellipticity_dic, fov )
            # save_ellipticity_table('data/pipo.txt', fb)
        else:
            plot_one_psf( fb, 0)


    

