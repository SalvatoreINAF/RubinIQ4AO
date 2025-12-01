#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:13:17 2024

@author: zanmar
"""


from fromBatoid import FromBatoid
from util import rayPSF, regular_grid, plot_ellipticity_map, submatsum,makeFocalPlanePlot
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import argparse
import lsstModel
import timeit
from util import addColorbarToAxes

def dof2str( i ):
    # convert from dof number to a string
    if( i == 0 ):
        text = 'M2z'
    elif( i == 1 ):
        text = 'M2x'
    elif( i == 2 ):
        text = 'M2y'
    elif( i == 3 ):
        text = 'M2rx'
    elif( i == 4 ):
        text = 'M2ry'
    elif( i == 5 ):
        text = 'Cz'
    elif( i == 6 ):
        text = 'Cx'
    elif( i == 7 ):
        text = 'Cy'
    elif( i == 8 ):
        text = 'Crx'
    elif( i == 9 ):
        text = 'Cry'
    else:
        text=''
        
    return text
        
    

def get_title( dof ):
    # if only up to 3 dofs is not zero, we create a sensible title
    #   otherwise, we only say ellipticity
    cnt = 0
    title = ''
    for i in range( len( dof ) ):
        if( dof[i] != 0 ):
            cnt += 1
            num = '%d'%dof[i] if( (dof[i] - int( dof[i]) )==0 ) else '%.1f'%dof[i]
            title += dof2str( i ) + '='+num+' '
    if( title != '' ):
        title = title.rstrip(' ')
    
    if( cnt > 3 ):
        title = 'ellipticity'
    return title
        
            
        
    
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

    parser.add_argument('--M2z', help='M2 decenter z [-80, 80] micron', type=float, default=0)
    parser.add_argument('--M2x', help='M2 decenter x [-3000, 3000] micron', type=float, default=0 )
    parser.add_argument('--M2y', help='M2 decenter y [-3000, 3000] micron', type=float, default=0 )
    parser.add_argument('--M2rx', help='M2 tilt x [-80, 80 ] arcsec', type=float, default=0)
    parser.add_argument('--M2ry', help='M2 tilt y [-80, 80 ] arcsec', type=float, default=0 )

    # parser.add_argument('--Cz', help='Camera decenter z [-80, 80] micron', type=int, default=0,  choices=range(-80,90,10) )
    parser.add_argument('--Cz', help='Camera decenter z [-80, 80] micron', type=float, default=0 )
    parser.add_argument('--Cx', help='Camera decenter x [-3000, 3000] micron', type=float, default=0)
    parser.add_argument('--Cy', help='Camera decenter y [-3000, 3000] micron', type=float, default=0)
    parser.add_argument('--Crx', help='Camera tilt x [-80, 80 ] arcsec', type=float, default=0)
    parser.add_argument('--Cry', help='Camera tilt y [-80, 80 ] arcsec', type=float, default=0)

    parser.add_argument("--idxdof", help='cvs string with dof indices 0-49', type=str)
    parser.add_argument("--dof", help='cvs string with corresponding magnitude of idxdof', type=str)

    parser.add_argument('-m','--model', help="use model, not batoid", action="store_true")
    parser.add_argument('-i','--interactive', help="plot is interactive to inspect PSFs", action="store_true")
    parser.add_argument('-s','--saveoutput', help="save results to a file", action="store_true")
    parser.add_argument('-cc','--comcam', help="use ComCam optical model", action="store_true")

    parser.add_argument('--stickscale', help='stick scale e.g. 0.5 - 1.5. *Negative for auto', type=float, default=-1)
    parser.add_argument('--seeing', help='seeing in arcsec for ray type PSF.', type=float, default=0.5)

    parser.add_argument("--psftype", help='chose how to model PSF: [ray/fft]', type=str, default='ray')
    
    parser.add_argument("--coords", help='file with coordinates hx hy. First line skipped.', type=str, default=None)
    
    args = parser.parse_args()

    dof = np.zeros(50)

    dof[0:5] = ( args.M2z, args.M2x, args.M2y, args.M2rx, args.M2ry )
    dof[5:10]= ( args.Cz, args.Cx, args.Cy, args.Crx, args.Cry )


    if( args.idxdof and args.dof ):
        try:
            dof_par = [ float( x ) for x in args.dof.split(",") ]
        except ValueError:
            print( "*** ValueError")
            print( "*** error in %s --dofs0 parameter" %args.dof )
            print( "*** Check your 10 csv")
            # sys.exit( 0 )
        try:
            idxdof_par = [ int( x ) for x in args.idxdof.split(",") ]
        except ValueError:
            print( "*** ValueError")
            print( "*** error in %s --dofs0 parameter" %args.idxdof )
            print( "*** Check your 10 csv")
            # sys.exit( 0 )
        dof_par = np.asarray( dof_par )
        idxdof_par = np.asarray( idxdof_par, dtype=int )
        dof[ idxdof_par ] = dof_par

    return dof, args.model, args.interactive, args.saveoutput, args.comcam, args.stickscale, args.seeing, args.psftype, args.coords

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
                            transform=ax.transAxes, va='top', fontsize='xx-small',
                            fontname='monospace')
        
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
        self.ax2.set_xlim((-10.0, 10.0))         # set ROI to -1,1 for both x,y
        self.ax2.set_ylim((-10.0, 10.0))
        self.ax2.set_title('PSF ray batoid')
        self.ax2.set_xlabel('x [pix]')
        self.ax2.set_ylabel('y [pix]')
        txttemp = 'p = %.2f' %(fb.ellipticity['p'][dataind])
        self.ax2.text( -9.9, 9.9, txttemp,
                            va='top', fontsize='xx-small',
                            fontname='monospace', color='orange')
        txttemp = 'q = %.2f' %(fb.ellipticity['q'][dataind])
        self.ax2.text( -9.9, 9.4, txttemp,
                            va='top', fontsize='xx-small',
                            fontname='monospace', color='orange')
        pa = fb.ellipticity['pa'][dataind]
        
        arrowx, arrowy = 9.5*np.cos(np.deg2rad(pa) ), 9.5*np.sin(np.deg2rad(pa) )
        arrow_dx, arrow_dy = 15*np.cos(np.deg2rad(pa) )-arrowx, 15*np.sin(np.deg2rad(pa) )-arrowy
        
        arrow = patches.Arrow( arrowx, arrowy, arrow_dx, arrow_dy, width=0.2,color='orange'  )
        self.ax2.add_patch(arrow)
        arrow = patches.Arrow( -arrowx, -arrowy, -arrow_dx, -arrow_dy, width=0.2 ,color='orange' )
        self.ax2.add_patch(arrow)
        
        
        
        
        
        
        # annotate eventually

        # ax2.text(0.05, 0.9, f'mu={xs[dataind]:1.3f}\nsigma={ys[dataind]:1.3f}',
        #          transform=ax2.transAxes, va='top')
        # ax2.set_ylim(-0.5, 1.5)
 
        
        self.selected.set_visible(True)
        self.selected.set_data([self.xs[dataind]], [self.ys[dataind]])

        self.text.set_text('el=%.3f\n pa=%.1f' %(fb.ellipticity['el'][dataind],
                                           fb.ellipticity['pa'][dataind]))
        self.fig.canvas.draw()

def plot_interactive( fb ):
    
    fig, (ax, ax2) = plt.subplots(1,2)
    
    U = fb.ellipticity['el'] * np.cos( np.deg2rad( fb.ellipticity['pa'] ) )
    V = fb.ellipticity['el'] * np.sin( np.deg2rad( fb.ellipticity['pa'] ) )
    M = np.hypot(U, V)
    
    try:
        xs=fb.hx
        ys=fb.hy
    except:
        xs=fb._hx
        ys=fb._hy
    
    
    line = ax.quiver(xs, ys, U, V, M, units='xy', scale=None, headwidth=1,
              headlength=0, headaxislength=0, pivot = 'middle',
              linewidth=0.8, picker=True, pickradius=20 )
    # ax.scatter( fb.hx, fb.hy, color='black', s=1)
    xymax = np.max([np.abs(xs),np.abs(ys)] ) * 1.1
    ax.set_xlim(-xymax, xymax )
    ax.set_ylim(-xymax, xymax )
    ax.set_aspect('equal', 'box')
    ax.set_title('ellipticity' )
    ax.set_xlabel('hx [deg]')
    ax.set_ylabel('hy [deg]')

    browser = PointBrowser( fig, ax, ax2, xs, ys, line )

    fig.canvas.mpl_connect('pick_event', browser.on_pick)
    fig.canvas.mpl_connect('key_press_event', browser.on_press)

    plt.show()    
    
def grid_from_file( fname, maxRows=300 ):
    # we return also the original array to append results to original data
    # expected columns: hx, hy, Ixx, Iyy, Ixy
    
    arr = np.loadtxt( fname, delimiter=',', skiprows=1 )
    n = len( arr[:,0] )
    if( n > maxRows ):
        rng = np.random.default_rng()
        indices = rng.choice(n, maxRows, replace=False )
        arr = arr[indices,:]
        
    Ixx, Iyy, Ixy = arr[ :,2 ], arr[ :,3 ], arr[ :,4 ]
    
    
    # imsim_data={'hx':np.rad2deg( arr[:,2]), 'hy': np.rad2deg(arr[:,3] ),
              # 'ex':arr[:,5], 'ey':arr[:,6] }
    imsim_data={'hx': arr[:,0], 'hy': arr[:,1] }
    
    return imsim_data['hx'], imsim_data['hy'], arr, Ixx, Iyy, Ixy

def makeMomentsMap(fig, axes, x, y, d_mxx, d_myy, d_mxy):

    pmin, pmax = np.quantile( d_mxx, [0.02,0.98] )

    cbar = addColorbarToAxes(
        axes[0, 0].scatter(x, y, c=d_mxx, vmin=pmin,vmax=pmax, cmap='bwr', s=16))

    cbar.set_label("mxx")

    pmin, pmax = np.quantile( d_mxy, [0.02,0.98] )
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(x, y, c=d_mxy, vmin=pmin, vmax=pmax, cmap="bwr", s=16)
    )
    cbar.set_label("mxy")

    # cbar = addColorbarToAxes(
    #     axes[1, 1].scatter(x, y, c=d_vy**2, vmin=-emax, vmax=emax, cmap="bwr", s=6)
    # )
    # cbar.set_label("vy")

    cmin, cmax = np.quantile( d_myy , [0.02,0.98])

    cbar = addColorbarToAxes(
        axes[0, 1].scatter(x, y, c=d_myy, vmin=cmin, vmax=cmax, cmap="bwr", s=16)
        # axes[0, 1].scatter(x, y, c=cost, cmap="bwr", s=16)
    )
    cbar.set_label("cost")


    for ax in axes.ravel():
        ax.set_xlabel("Focal Plane x [deg]")
        ax.set_ylabel("Focal Plane y [deg]")
        ax.set_aspect("equal")

    # # Plot camera detector outlines
    # for det in camera:
    #     xs = []
    #     ys = []
    #     for corner in det.getCorners(FOCAL_PLANE):
    #         xs.append(corner.x)
    #         ys.append(corner.y)
    #     xs.append(xs[0])
    #     ys.append(ys[0])
    #     xs = np.array(xs)
    #     ys = np.array(ys)
    #     for ax in axes.ravel():
    #         ax.plot(xs, ys, c="k", lw=1, alpha=0.3)

    fig.tight_layout()
    # if saveAs:
    #     fig.savefig(saveAs)

    return
    

if( __name__ == '__main__'):

    manypoints = True
    dof, with_model, interactive, saveoutput, comcam, stickscale, seeing, psftype, coordfile = pars2dof()
    if stickscale < 0:
        stickscale = None
    print( '############ input pars #############' )
    print( 'dof0:', dof[0:10] )
    print( 'dof1:', dof[10:20] )
    print( 'dof2:', dof[20:30] )
    print( 'dof3:', dof[30:40] )
    print( 'dof4:', dof[40:50] )
    print( 'with_model:', with_model )
    print( 'interactive:', interactive ) 
    print( 'saveoutput:', saveoutput ) 
    print( 'comcam:', comcam )
    print( 'stickscale:', stickscale ) 
    print( 'seeing:', seeing ) 
    print( 'psftype:', psftype ) 
    print( 'coordfile:', coordfile )
    print( '############ ########## #############' )
    
    # with_model = True
    # dof = np.zeros(50)
    # dof[0] = 25
    # dof[1] = 0
    # # dof[2] = 000
    # dof[3] = 22.5
    # dof[4] = 22.5
    # dof[5] = 25
    # dof[6] = 1750
    # dof[7] = 1750
    # dof[8] = 75
    # dof[9] = 75
    # dof[10] = 0.2
    # dof[11] = -0.05

    if( with_model ):
        #fname = 'znk_batoid_coeffs_wl_6_jmax_11.hdf5'
        # fname = 'znk_batoid_coeffs_wl_2_jmax_22_dbg.hdf5'
        # fname = 'znk_batoid_coeffs_wl_6_jmax_22.hdf5'
        # fname = 'znk_batoid_coeffs_wl_2_jmax_37_dbg.hdf5'
        # fname = 'znk_batoid_coeffs_wl_6_jmax_37.hdf5'
        # fname = 'data/test_models/znk_batoid_coeffs_wl_6_jmax_37.hdf5_7BY08R'
        fname = 'data/test_models/znk_batoid_coeffs_wl_6_jmax_37.json'
        #fname = 'data/test_models/znk_batoid_coeffs_wl_6_jmax_37_ComCam.hdf5_Z08IOR'
        lm = lsstModel.lsstModel( model_file=fname, n_fldznk=22)
        lm.wl_index = 2
        lm.dof = dof
        
        fov = lm.field_radius
    else:
        fb = FromBatoid( comcam = comcam )          # default comcam = False

        fb.wl_index = 1
        fb.dof = dof
        
        fov = fb.field_radius
            
    if( manypoints ):
        
        if( coordfile ):
            x, y, data_imsim, Ixx, Iyy, Ixy = grid_from_file( coordfile )
        else:
            x, y = regular_grid( fov, num=19 )
            
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
        tic=timeit.default_timer()
        # lm.psf_compute( with_binning=True )
        # lm.ellipticity_compute()
        
        lm.psf_compute_lattice( N=128, L=6 )
        lm.ellipticity_compute(withcoords=True)
        
        toc=timeit.default_timer()
        print("ellipticity map took %f seconds" %(toc - tic))
        ellipticity_dic = lm.ellipticity
        
    else:
        fb.psf_compute( psf_type = psftype, seeing=seeing )
        fb.ellipticity_compute()
        ellipticity_dic = fb.ellipticity
        ## PSF
       
    if( interactive ):
        if( with_model ):
            fb = lm
            plot_interactive( fb )
            
            
        else:    
            plot_interactive( fb )
    else:      
        if( manypoints ):
            # ellipticity map
            mytitle = get_title( dof )
            if( psftype == 'ray' and not with_model ):
                mytitle += '_s%.1f' %(seeing )
            # plot_ellipticity_map( x, y, ellipticity_dic, fov, scale=stickscale, 
            #                      saveit=True, 
            #                      title=mytitle )
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9), sharex=True, sharey=True)
            makeFocalPlanePlot(fig, axes, x, y, ellipticity_dic )
            plt.show()
            # save_ellipticity_table('data/pipo.txt', fb)
        else:
            plot_one_psf( fb, 0)

    el = np.asarray( ellipticity_dic['el'] )
    pa = np.asarray( ellipticity_dic['pa'] )
    sl = np.asarray( ellipticity_dic['sl'] )
    ss = np.asarray( ellipticity_dic['ss'] )
    mxx = np.asarray( ellipticity_dic['muxx'] ) * 0.2**2 #arcsec^2
    myy = np.asarray( ellipticity_dic['muyy'] ) * 0.2**2
    mxy = np.asarray( ellipticity_dic['muxy'] ) * 0.2**2
    
    if( coordfile ):
        rad = np.hypot( x, y )
        #### plot comparisons between imsim and model/batoid
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 9), sharex=False, sharey=False)
        ax[0,0].plot( rad, Ixx-mxx,'*' )
        # ax[0,0].plot( rad, mxx,'x' )
        # ax[0,0].plot( mxx_ana, p(mxx_ana), '-', label="%fx+%f"%(z[0],z[1]) )
        ax[0,0].set_title('mxx vs mxx')
        ax[0,0].set_xlabel('rad')
        ax[0,0].set_ylabel('mxx')
        ax[0,0].legend(loc="upper left")
        
        ax[1,0].plot( rad, Ixy-mxy,'*' )
        # ax[0,0].plot( rad, mxx,'x' )
        # ax[0,0].plot( mxx_ana, p(mxx_ana), '-', label="%fx+%f"%(z[0],z[1]) )
        ax[1,0].set_title('mxy vs mxy')
        ax[1,0].set_xlabel('rad')
        ax[1,0].set_ylabel('mxx')
        ax[1,0].legend(loc="upper left")
        
        ax[0,1].plot( rad, Iyy-myy,'*' )
        # ax[0,0].plot( rad, mxx,'x' )
        # ax[0,0].plot( mxx_ana, p(mxx_ana), '-', label="%fx+%f"%(z[0],z[1]) )
        ax[0,1].set_title('myy vs myy')
        ax[0,1].set_xlabel('rad')
        ax[0,1].set_ylabel('mxx')
        ax[0,1].legend(loc="upper left")
                
        plt.show( block=False )
        
        ikeep = rad < 3.0
        
        
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 9), sharex=False, sharey=False)
        ax[0,0].plot( mxx[ikeep], Ixx[ikeep],'*' )
        # ax[0,0].plot( rad, mxx,'x' )
        # ax[0,0].plot( mxx_ana, p(mxx_ana), '-', label="%fx+%f"%(z[0],z[1]) )
        ax[0,0].set_title('mxx vs mxx')
        ax[0,0].set_xlabel('model/batoid')
        ax[0,0].set_ylabel('imsim')
        ax[0,0].legend(loc="upper left")
        
        ax[1,0].plot( mxy[ikeep], Ixy[ikeep],'*' )
        # ax[0,0].plot( rad, mxx,'x' )
        # ax[0,0].plot( mxx_ana, p(mxx_ana), '-', label="%fx+%f"%(z[0],z[1]) )
        ax[1,0].set_title('mxy vs mxy')
        ax[1,0].set_xlabel('model/batoid')
        ax[1,0].set_ylabel('imsim')
        ax[1,0].legend(loc="upper left")
        
        ax[0,1].plot( myy[ikeep], Iyy[ikeep],'*' )
        # ax[0,0].plot( rad, mxx,'x' )
        # ax[0,0].plot( mxx_ana, p(mxx_ana), '-', label="%fx+%f"%(z[0],z[1]) )
        ax[0,1].set_title('myy vs myy')
        ax[0,1].set_xlabel('model/batoid')
        ax[0,1].set_ylabel('imsim')
        ax[0,1].legend(loc="upper left")
                
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9), sharex=True, sharey=True)
        
        makeMomentsMap(fig, axes, x, y, Ixx-mxx, Iyy-myy, Ixy-mxy)
        
        
        plt.show( block=True )

    if( saveoutput ):    
        
        Xout = np.vstack( (x,y,el,pa,mxx,myy,mxy) )
        Xout = Xout.T
        if( coordfile ):
            Xout = np.hstack( (data_imsim, Xout ) )
            print( Xout.shape )
            
            
            
            
            

        if( len(mytitle) == 0):
            title = 'nominal'
        else:
            title = mytitle.replace('=','_')
        outfile = 'ellip_'+title+'.csv'
        fullname = '/tmp/'+outfile

        if( coordfile ):
            np.savetxt(fullname, Xout, header='aa_field_x,aa_field_y,aa_Ixx,aa_Iyy,aa_Ixy, hx, hy, mod_el, mod_pa, mod_mxx, mod_myy, mod_mxy',comments='',delimiter=",")
        else:
            np.savetxt(fullname, Xout, header='hx, hy, el, pa, mxx, myy, mxy',comments='#',delimiter=",")

        print('saved %s' %(fullname) )
