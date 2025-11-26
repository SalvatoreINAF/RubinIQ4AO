#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:04:04 2024

@author: zanmar
"""

import numpy as np
from scipy.optimize import minimize
import lsstModel
from util import regular_grid, plot_ellipticity_map, makeAzElPlot, moments_and_dof_from_file
import sys
from fromBatoid import FromBatoid, MisAlignment
from scipy.optimize import Bounds
import timeit
import pickle
import argparse
import matplotlib.pyplot as plt
import pathlib
import json

def parse_inputs():
    parser = argparse.ArgumentParser()
    # parser.print_help()

    parser.add_argument("--idxdof", help='cvs string with dof indices 0-49', type=str)
    parser.add_argument("--dofs0", help='cvs string with corresponding magnitude of idxdof', type=str)
    parser.add_argument("--ifit", help='cvs string with indices of dofs to be fitted', type=str)

    parser.add_argument("--ix0", help='cvs string with index of initial x0 passed to minimize', type=str)
    parser.add_argument("--x0", help='cvs string with x0 passed to minimize', type=str)

    parser.add_argument("--fname", help='file name or mod/ray/fft for fake data', type=str, default='mod')

    parser.add_argument("--oldrun", help='file name of a previous run with solution', type=str )

    parser.add_argument('--ntry', help='number of random tries', type=int, default=2)
    
    parser.add_argument('--noplot', help="don't produce plots", action="store_true")
    parser.add_argument('--noverbose', help="produce minimal CSV output", action="store_true")

    args = parser.parse_args()
    
    
    if( args.ifit ):
        try:
            pars_active = [ int( x ) for x in args.ifit.split(",") ]
        except (AttributeError,ValueError) as e:
            print( e )
            print( "*** error in %s --ifit parameter" %args.ifit )
            print( "*** Check your csv string")
            sys.exit( 1 )
        pars_active = np.asarray( pars_active, dtype=int )
    else:
        print('--ifit empty \n Need at least one parameter to fit for')
        sys.exit( 1 )

    if( args.dofs0 ):
        try:
            dof_par = [ float( x ) for x in args.dofs0.split(",") ]
        except (AttributeError,ValueError) as e:
            print( e )
            print( "*** error in %s --dofs0 parameter" %args.dofs0 )
            print( "*** Check your csv string")
            sys.exit( 1 )
        dof_par = np.asarray( dof_par )
    else:
        dof_par = None
    
    if( args.idxdof ):
        try:
            idxdof_par = [ int( x ) for x in args.idxdof.split(",") ]
        except ValueError:
            print( "*** ValueError")
            print( "*** error in %s --idxdof parameter" %args.idxdof )
            print( "*** Check your csv string")
            sys.exit( 1 )
        idxdof_par = np.asarray( idxdof_par, dtype=int )
        dofs0 = np.zeros(50)
        dofs0[ idxdof_par ] = dof_par
    else:
        idxdof_par = None
        
    if( idxdof_par is None or dof_par is None):
        dofs0 = None
    
    if( args.x0 ):
        try:
            x0_par = [ float( x ) for x in args.x0.split(",") ]
        except (AttributeError,ValueError) as e:
            print( e )
            print( "*** error in %s --x0 parameter" %args.dofs0 )
            print( "*** Check your csv string")
            sys.exit( 1 )
        x0_par = np.asarray( x0_par )
    else:
        x0_par = None
    
    if( args.ix0 ):
        try:
            ix0_par = [ int( x ) for x in args.ix0.split(",") ]
        except ValueError:
            print( "*** ValueError")
            print( "*** error in %s --idxdof parameter" %args.idxdof )
            print( "*** Check your csv string")
            sys.exit( 1 )
        ix0_par = np.asarray( ix0_par, dtype=int )
    else:
        ix0_par = None

    return args.ntry, pars_active, dofs0, args.fname, args.noplot, args.noverbose, args.oldrun, x0_par, ix0_par


def ellip_map( x, lm, pars_active ):
    """
    return an ellipticity map as Nx4 matrix with hx, hy, ex, ey

    Parameters
    ----------
    x : array (float)
        10 dof
    coords : array (float)
        Nx2 matrix with hx, hy coordinates in degrees.
    lm : class
        model

    Returns
    -------
    Nx4 matrix with hx, hy, el, pa [deg]

    """
    
    lm.dof[ pars_active ] = x

    lm.ellipticity_compute_algebra()

    ellip_map = np.zeros( (len( lm.ellipticity['q']), 3 ) )

    ellip_map[:,0 ] = lm.ellipticity['q']
    # ellip_map[:,0 ] = lm.ellipticity['el']
    ellip_map[:,1 ] = (( np.asarray( lm.ellipticity['pa'] ) * 2 + 360 )%360)/2 # twice the angle trick
    ellip_map[:,2 ] = lm.ellipticity['p']

    return ellip_map


def ellipticity_difference( map1, map2 ):
    ### the maps are 2D arrays with magnitude and PA in deg
    # start with the vector difference of ex, ey after doubling the angle
    #   to avoid the 180 angle degeneracy

    q_mod = map2[:,0]
    pa_mod= map2[:,1]
    p_mod = map2[:,2]
    
    q_obs = map1[:,0]
    pa_obs= map1[:,1]
    p_obs = map1[:,2]

    pp_mod = p_mod - np.median( p_mod )
    pp_obs = p_obs - np.median( p_obs )
    
    k0 = 0.183         # empirical scaling constant ARCSEC
    k0 = k0 / 0.2      # PIXEL as q and p
    
    el_mod = k0**2 * np.log( 1 + q_mod/k0**2 )
    el_obs = k0**2 * np.log( 1 + q_obs/k0**2 )
    
    vx_mod = el_mod * np.cos( np.deg2rad( pa_mod ))
    vy_mod = el_mod * np.sin( np.deg2rad( pa_mod ))
    
    vx_obs = el_obs * np.cos( np.deg2rad( pa_obs ))
    vy_obs = el_obs * np.sin( np.deg2rad( pa_obs ))
    
    cost = ( ( pp_mod - pp_obs )**2/4 +
            (vx_mod - vx_obs)**2 +
            (vy_mod - vy_obs)**2 )
    # cost =  (vx_mod - vx_obs)**2 + (vy_mod - vy_obs)**2     # vx and vy

    # ang1 = np.deg2rad( map1[:,1] )
    # ang2 = np.deg2rad( map2[:,1] )
    # ex1 = map1[:,0] * np.cos( ang1 )
    # ey1 = map1[:,0] * np.sin( ang1 )

    # ex2 = map2[:,0] * np.cos( ang2 )
    # ey2 = map2[:,0] * np.sin( ang2 )
    
    # cost = np.sqrt( np.sum( (ex1-ex2)**2 + (ey1-ey2)**2 ) / len( ex1 ) )
    
    return np.sum( cost )*1e6/len( p_mod )

def cost_function( x, data_map, lm, pars_active ):

    current_map = ellip_map( x, lm, pars_active )

    # print( data_map - current_map )

    return ellipticity_difference( data_map, current_map )

    
def setup_model( fname, hx, hy, dof0 ):
    """
    set up the model by calling the lsstModel class. We will initialize the
    coordinates, and DOF

    Parameters
    ----------
    fname : str
        hdf5 file with batoid zernike coefficients.
    hx, hy : array
        Field coordinates
    dof0 : array[50]
        the 50 misalignments, 5 for M2 and 5 for Camera, dz, dx, dy, Rx, Ry
       and bending modes
    Returns
    -------
    The lsstModel.

    """
    # TODO... fldznk is different from jmax in model. Need to investigate the
    #   effect of n_fldznk but perhaps 22 is enough, 37 overkilling.
    if( 'jmax_22' in fname ):
        n_fldznk = 22
    elif( 'jmax_37' in fname):
        n_fldznk = 22
    elif( 'jmax_11' in fname):
        n_fldznk = 22
    else:
        print( '*** Error, unknown fname kind' )
        sys.exit(1)
            
    lm = lsstModel.lsstModel( batoid_cube_file=fname, n_fldznk=n_fldznk )
    lm.wl_index = 2 #red

    # coords =  np.vstack( (hx, hy) ).T
    # print( 'Using %d points' %coords.shape[0] )
    
    lm.set_hxhy( hx, hy )
    
    lm.dof = np.asarray( dof0 )
    
    return lm

def get_random_dofs( N, bounds, seed=None ):
    """
    Generate an array of N x fitted_pars with the random DOFS.

    Parameters
    ----------
    N:       int
        Number of random dofs to be created
    bounds : a list of tuples
        min/max bounds to be used for each of the fitted params.

    Returns
    -------
    an array of N x fitted_pars with each row having a different set of dofs to
       be fitted.
    """

    
    kk_min = np.array( [ x[0] for x in bounds ] )
    kk_max = np.array( [ x[1] for x in bounds ] )
    
    rng = np.random.default_rng( seed = seed )
    
    dofs_fitted_rnd = rng.random( (N, len( bounds ) ) )
    dofs_fitted_rnd = dofs_fitted_rnd * (kk_max-kk_min ) + kk_min
    
    return dofs_fitted_rnd

def run_minimizer_N_times( N_try, lm, data_map, free_pars_index, x0_par=None, ix0_par=None ):
    # a wrapper around scipy.minimize where you can try different random
    #    initial conditions.
    #IN: bounds are the bounds for only the free parameters
    
    # run N times the minimizer with N_tries attempts each.
    
    #We don't expect misalignments outside these bounds, PSF would be a donut.
#    bounds_all =      [(-55.,55.),                #M2     dz
#                       (-1200.,1200.),            #       dx
#                       (-1200.,1200.),            #       dy
#                       (-40., 40.),               #       Rx
#                       (-40., 40.),               #       Ry
#                       (-50.,50.),                #Camera dz
#                       (-4000.,4000.),            #       dx
#                       (-4000.,4000.),            #       dy
#                       (-45., 45.),               #       Rx
#                       (-45., 45) ]               #       Ry

    mal = MisAlignment()
    posamps = mal._max_amplitude / 1
    negamps = -1. * posamps

    bounds_arr = np.vstack( ( negamps, posamps ) ).T
    bounds_all = [ tuple( row ) for row in bounds_arr ]

    # bounds for just the free parameters
    bounds = [ bounds_all[ i ] for i in free_pars_index ]

    results_list = []
    fun_list = []    
    x0_list = []

    # array of different initial guess
    x0arr = get_random_dofs( N_try, bounds )
    print( x0arr )
    if( (x0_par is not None) and (ix0_par is not None) ):
        print( 'Using user provided x0 for indices:', ix0_par )
        for i, ix in enumerate( ix0_par ):
            ireplace = np.where( free_pars_index == ix )[0][0]
            if( ireplace >= 0 and ireplace < x0arr.shape[1] ):
                x0arr[:, ireplace ] = x0_par[ i ]
            print( ix, ireplace, free_pars_index )

        print( x0arr )
    
    for j in range( N_try ):
        # print("=====> attempt %d/%d " %(j+1, N_try ))
        x0 = x0arr[ j ]

        # res = minimize(cost_function, x0, method='nelder-mead',
        #            args=(data_map, lm, free_pars_index ), 
        #            options={'xatol': 1e-8,
        #            # 'fatol': 1e-1,
        #            'disp': False,
        #            'adaptive': True,
        #            'maxfev':10000},
        #            bounds=bounds )

        res = minimize(cost_function, x0, method='powell',
                   args=(data_map, lm, free_pars_index ), 
                   options={
                   'disp': False,
                   'maxfev':10000},
                   bounds=bounds )

        # print("%10d %10e %10d %10d" %( j, res.fun, res.nit, res.status ))

        if( res.success ):
            x0_list.append( x0 )
            fun_list.append( res.fun )
            results_list.append( res )
    
    cost = np.asarray( fun_list )
    imin = np.argmin( cost )
    print( 'old approach min cost:', cost[ imin ], results_list[imin] )
    # now we sort the result list by the minimum cost in fun_list
    fun_list = np.asarray( fun_list )
    isort = np.argsort( fun_list )
    fun_list = fun_list[ isort ]
    x0_list = [ x0_list[i] for i in isort ]
    results_list = [ results_list[i] for i in isort ]
    
    # return results_list[ imin ], x0_list[ imin ]
    return results_list, x0_list

class random_min_tests( object ):
    def __init__( self, N, dof_start, lm, free_pars_index, N_tries ):
        self.N = N                      # N different fits
        self.dof0 = dof_start           # all 10 pars, some of them to be fitted
        self.lm = lm
        self.free_pars_index = free_pars_index
        self.N_tries = N_tries
        self.bounds_all = [(-55.,55.),                #M2     dz
                           (-1200.,1200.),            #       dx
                           (-1200.,1200.),            #       dy
                           (-40., 40.),               #       Rx
                           (-40., 40.),               #       Ry
                           (-50.,50.),                #Camera dz
                           (-4000.,4000.),            #       dx
                           (-4000.,4000.),            #       dy
                           (-45., 45.),               #       Rx
                           (-45., 45) ]               #       Ry
        
        self.bounds = [ self.bounds_all[ i ] for i in free_pars_index ]
        self.dofs_fitted = get_random_dofs(N, self.bounds )
        self.dofs_fitted_initial = None
            
    def run( self ):
        # run N times the minimizer with N_tries attempts each.
        i_exp = []
        j_try = []
        fitted_pars_list=[]
        dofs_fitted_list = []
        x0_list = []
        x_list = []
        fun_list = []
        nit_list = []
        nfev_list = []
        status_list = []
        
        
        
        
        for i in range( self.N ):
            # print("---------- experiment %d/%d ---------" %(i+1, self.N ))
            # fake data
            data_map = ellip_map( self.dofs_fitted[i], self.lm, self.free_pars_index )
            
            # array of different initial guess
            x0arr = get_random_dofs(self.N_tries, self.bounds, (i+1) * 27182 ) 

            for j in range( self.N_tries ):
                # print("=====> attempt %d/%d " %(j+1, self.N_tries ))
                x0 = x0arr[ j ]

                res = minimize(cost_function, x0, method='nelder-mead',
                           args=(data_map, self.lm, self.free_pars_index ), 
                           options={'xatol': 1e-8,
                           # 'fatol': 1e-1,
                           'disp': False,
                           'adaptive': True,
                           'maxfev':10000},
                           bounds=self.bounds )
                # print( 'expected:', self.dofs_fitted[i] )
                # print( 'guess:   ', x0 )
                # if( res.fun <= 0.001 ):
                #     print( 'result:  ', res.x, ' ok' )
                # else:
                #     print( 'result:  ', res.x, ' XX' )
                    
                print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s" %(
                      i, j, 
                      np.asarray( self.free_pars_index ), 
                      self.dofs_fitted[i], 
                      x0, 
                      res.x, 
                      res.fun, 
                      res.nit, 
                      res.nfev, 
                      res.status ))
                i_exp.append( i )
                j_try.append( j )
                fitted_pars_list.append( self.free_pars_index )
                dofs_fitted_list.append( self.dofs_fitted[i] )
                x0_list.append( x0 )
                x_list.append( res.x )
                fun_list.append( res.fun )
                nit_list.append( res.nit )
                nfev_list.append( res.nfev )
                status_list.append( res.status )
                
                
        results={'experiment':i_exp,
                 'try_n':j_try,
                 'fitted_pars': fitted_pars_list,
                 'dofs_fitted': dofs_fitted_list,
                 'x0': x0_list,
                 'x': x_list,
                 'fun': fun_list,
                 'nit': nit_list,
                 'nfev': nfev_list,
                 'status': status_list
                 }
            
        return results

def data_map_batoid( dofs0, fov=1.7, num=26, psf_type='fft', seeing=0.5 ):
    """
    Run batoid to create an ellipticity map.

    Parameters
    ----------
    fov : float
        maximum 
    num : int
        DESCRIPTION. The default is 26.
    psf_type : str [fft/ray]
        DESCRIPTION. The default is 'fft'.
    seeing : float
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    None.

    """
    hx, hy = regular_grid( fov, num=num )

    fb = FromBatoid(jmax=37)
    fb.wl_index = 1
    fb.hx, fb.hy = hx, hy
    fb.dof = np.asarray( dofs0 )

    # fb.psf_compute( psf_type = imsim_file, seeing=0.5 )
    # fb.ellipticity_compute()
    fb.psf_ellip_compute_multiproc( psf_type= psf_type, seeing=0.5 )

    data_map = np.zeros( (len( fb.ellipticity['q']), 3 ) )
    data_map[:,0 ] = fb.ellipticity['q']
    data_map[:,1 ] = (( np.asarray( fb.ellipticity['pa'] ) * 2 + 360 )%360) /2#twice the angle
    data_map[:,2 ] = fb.ellipticity['p']

    data_dic={ 'p': np.asarray( fb.ellipticity['p'] ),
              'e1': np.asarray( fb.ellipticity['e1'] ), 
              'e2': np.asarray( fb.ellipticity['e2'] ), 
              'mxx': np.asarray( fb.ellipticity['muxx'] ),
              'myy': np.asarray( fb.ellipticity['muyy'] ),
              'mxy': np.asarray( fb.ellipticity['muxy'] )}

    del fb
    return hx, hy, data_map, data_dic

def read_ellipticity_map( imsim_file, dofs0=None ):
    """
    The format might change during development, keep an eye.
    
    Right now we expect the following cvs columns with 1 line header
    #x,y,e,theta1,i_xx,i_yy,i_xy    

    Parameters
    ----------
    imsim_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if( imsim_file=='ray' or imsim_file=='fft'):
        # fake data with batoid/rays
        hx, hy, data_map, data_dic = data_map_batoid( dofs0, 1.7, 26, imsim_file )

    elif( imsim_file=='mod' ):
        # fake data with model itself
        # create coordinates or get them from file

        # fname = 'znk_batoid_coeffs_wl_2_jmax_22_dbg.hdf5'
        # fname = 'znk_batoid_coeffs_wl_6_jmax_22.hdf5'
        fname = 'data/test_models/znk_batoid_coeffs_wl_6_jmax_37.hdf5_7BY08R'
        # fname = 'znk_batoid_coeffs_wl_6_jmax_11.hdf5'
        # fname = 'znk_batoid_coeffs_wl_2_jmax_37_dbg.hdf5'
        # fname = 'znk_batoid_coeffs_wl_6_jmax_37.hdf5'

        hx, hy = regular_grid( 1.7*0.9, num=26 )
        lm0 = setup_model(fname, hx, hy, dofs0 )
        ####### test with fake model data
        # pipo=np.asarray( dofs_fake )

        lm0.ellipticity_compute_algebra()

        data_map = np.zeros( (len( lm0.ellipticity['q']), 3 ) )
        data_map[:,0 ] = lm0.ellipticity['q']
        data_map[:,1 ] = (( np.asarray( lm0.ellipticity['pa'] ) * 2 + 360 ) % 360 )/2
        data_map[:,2 ] = lm0.ellipticity['p']

        data_dic={ 'p': np.asarray( lm0.ellipticity['p'] ),
              'e1': np.asarray( lm0.ellipticity['e1'] ), 
              'e2': np.asarray( lm0.ellipticity['e2'] ), 
              'mxx': np.asarray( lm0.ellipticity['muxx'] ),
              'myy': np.asarray( lm0.ellipticity['muyy'] ),
              'mxy': np.asarray( lm0.ellipticity['muxy'] )}

        del lm0
        #######
    else:
        # filename extension now has a meaning.
        myfile = pathlib.Path( imsim_file )
        if( myfile.suffix == '.csv'):
            hx, hy, mxx_ray, myy_ray, mxy_ray, dof_in_file = moments_and_dof_from_file( imsim_file,
                                                                    radmax=1.7 )
            # read moments from imsim file to produce p, q, pa
            # arr = np.loadtxt( imsim_file, delimiter=',', skiprows=1 )
            # hx = arr[:,0]
            # hy = arr[:,1]
    
            p_ray = mxx_ray + myy_ray
            e1 = ( mxx_ray - myy_ray ) / p_ray
            e2 = 2 * mxy_ray / p_ray
            q_ray = np.sqrt( e1**2 + e2**2 ) * p_ray
            pa_ray = np.rad2deg( np.arctan2( e2, e1 ) / 2 )

            pa_ray2 = (pa_ray*2+360)%360
            # e_ray = q_ray/p_ray

            data_map = np.zeros( (len( hx ), 3 ) )
            data_map[:,0 ] = q_ray
            data_map[:,1 ] = pa_ray2 / 2
            data_map[:,2 ] = p_ray
            
            data_dic={ 'p': p_ray, 'e1': e1, 'e2': e2, 
                      'mxx': mxx_ray, 'myy': myy_ray, 'mxy': mxy_ray }
            
        elif( myfile.suffix == '.bat' ):
            # the data map was directly saved, not the moments.
            arr = np.loadtxt( imsim_file, delimiter=',', skiprows=2 )
            hx, hy = arr[:,0], arr[:,1]
            
            data_map = arr[:,2:]
        else:
            print('Unrecognized file extension. Either bat or csv')
            sys.exit(1)



    # filter out points outside FOV of telescope
    # ikeep = np.hypot(hx,hy) <= 1.7
    
    # return hx[ikeep], hy[ikeep], data_map[ikeep,:], data_dic
    return hx, hy, data_map, data_dic

def trim_data( x, y, data_arr, n_Rows = 1000 ):
    n = len( data_arr[:,0] )
    maxRows = n_Rows
    if n > maxRows:
        rng = np.random.default_rng()
        indices = rng.choice(n, maxRows, replace=False)
        data_map = data_arr[indices,:]
        xout = x[indices]
        yout = y[indices]
    else:
        return x, y, data_arr
    
    return xout, yout, data_map

def get_ellip_dic( data_map, hx, hy ):
    # data_map is an array with columns having p, q, pa
    # we need only 3 elements for the plot
    q = data_map[ :, 0 ]
    pa =data_map[ :, 1 ]
    p = data_map[ :, 2 ]
    
    # p = mu_xx + mu_yy                   # pix^2
    # e1 = ( mu_xx - mu_yy ) / p          #
    # e2 = 2 * mu_xy / p                  #
    # q = np.sqrt( e1**2 + e2**2 ) * p    # pix^2
    # sigma_l = np.sqrt( ( p + q ) / 2 )
    # sigma_s = np.sqrt( ( p - q ) / 2 )
    # ellipticity_adimensional = 1 - sigma_s/sigma_l
    # alpha = np.rad2deg( np.arctan2( e2, e1 ) / 2 )
    
    K = np.tan( 2*np.deg2rad( pa ) )
    print('debug plot pa Vs K')
    plt.plot( pa, K, '.' )
    plt.show()
    print( 'debug scatter map hx, hy color is pa' )
    plt.scatter( hx, hy, c= pa, s=5 )
    plt.show()
    print( 'debug plot pa only')
    plt.plot(pa,'.')
    plt.show()
    
    e1 = np.sqrt( q**2 / p**2 / ( 1 + K**2 ) )
    
    ichange = ( ( (pa >  45) & (pa <  90) ) | 
                ( (pa < -45) & (pa > -90) ) | 
                ( (pa >  90) & (pa < 135) ) )
    e1[ichange] = e1[ichange]*-1
    
    e2 = K * e1
    
    
    
    # e = np.hypot( e1, e2 )
    
    dic = {'e1':e1, 'e2':e2,'p':p }
    # dic['e'] = e
    
    print( dic.keys() )
    print( len( dic['p'] ) )
    
    return dic
    
    
def minimize_map( ntry, free_pars_index, dofs_fake, imsim_file, noplot=False, noverbose=False, x0_par=None, ix0_par=None ):
    """
    Find the DOFs that best fits the given ellipticity map.

    Parameters
    ----------
    ntry : int
        Generate ntry different initial conditions.
    free_pars_index : np.array
        array of indices of dof [0-49] to be fitted for
    dofs_fake : np.array of 50 elements
        array of initial state of dofs (to allow some fixed parameters different from 0)
    imsim_file : str
        the file name with ellipticity map. If name is mod/fft/ray we simulate the input data
    noplot : TYPE, optional
        DESCRIPTION. The default is False.
    noverbose : TYPE, optional
        DESCRIPTION. The default is False.
    x0_par : np.array, optional
        array of initial guess for the free parameters.
    ix0_par : np.array, optional
        array of indices of the free parameters for which x0_par has values.

    Returns
    -------
    None.

    """
    if( noverbose ):
        pass
    else:
        print( dofs_fake[0:10] )
        print( dofs_fake[10:20] )
        print( dofs_fake[20:30] )
        print( dofs_fake[30:40] )
        print( dofs_fake[40:50] )

    
    # imsim_file = "data/lsst_imsim/ellipticitymap_grid_visitid5023071800117_forricardo.csv"
    # imsim_file = "data/lsst_imsim/ellipticitymap_grid_visitid5023071800118_forricardo.csv"
    # imsim_file = 'mod'
    hx, hy, data_map, data_dic = read_ellipticity_map( imsim_file, dofs0 = dofs_fake )
    # hx, hy, data_map = trim_data(hx, hy, data_map)
    ### set up model

    # fname = 'znk_batoid_coeffs_wl_2_jmax_22_dbg.hdf5'
    # fname = 'znk_batoid_coeffs_wl_6_jmax_22.hdf5'
    fname = 'data/test_models/znk_batoid_coeffs_wl_6_jmax_37.hdf5_7BY08R'
    # fname = 'znk_batoid_coeffs_wl_6_jmax_11.hdf5'
    # fname = 'znk_batoid_coeffs_wl_2_jmax_37_dbg.hdf5'
    # fname = 'znk_batoid_coeffs_wl_6_jmax_37.hdf5'

    lm = setup_model( fname, hx, hy, dofs_fake.copy() )
    #### FIX THIS ### I need the same ramdom points for both the model
    ####    and the input data. UGLY
    n = len( data_map[:,0] )
    maxRows = 1000
    if n > maxRows:
        rng = np.random.default_rng()
        indices = rng.choice(n, maxRows, replace=False)
        data_map_plot = data_map[indices,:]
        xplt = hx[indices]
        yplt = hy[indices]
    else:
        xplt = hx
        yplt = hy
        data_map_plot = data_map

    if( noplot ):
        pass
    else:
        plot_ellipticity_map( xplt, yplt, data_map_plot, 1.7,
                         scale=None, 
                         saveit=False, 
                         title='data', maxRows=1500 )
    
    # run minimizations
    res_list, x0_list = run_minimizer_N_times(ntry, lm, data_map, free_pars_index, x0_par, ix0_par )
    res = res_list[0]
    x0_winner = x0_list[ 0 ]
    print( 'new approach:', res, x0_winner)
    sols_fname = get_output_fname( imsim_file, free_pars_index, ntry, dofs_fake, 'tries' )
    tries_json = get_tries_json_data( res_list, x0_list, free_pars_index )

    with( open( sols_fname, 'w' ) ) as jf:
        json.dump( tries_json, jf, indent=4 )
    
    dofs_found = np.asarray( dofs_fake.copy() )
    dofs_found[ free_pars_index ] = res.x
    
    #save data and model to a csv file
    fitted_map = ellip_map( res.x, lm, free_pars_index )
    lm.dof[ free_pars_index ] = res.x
    lm.ellipticity_compute_algebra()

    # Concatenate arrays for saving: xplt, yplt, data_map_plot, fitted_map_plot
    output_arr = np.column_stack((
        hx, hy,
        data_map,
        fitted_map,
        np.asarray( data_dic['mxx'] ),
        np.asarray( data_dic['myy'] ),
        np.asarray( data_dic['mxy'] ),
        np.asarray( lm.ellipticity['muxx'] ),
        np.asarray( lm.ellipticity['muyy'] ),
        np.asarray( lm.ellipticity['muxy'] )
    ))
    header = "x[deg], y[deg], q_data[pix^2], pa_data[deg], p_data[pix^2], q_model[pix^2], pa_model[deg], p_model[pix^2], mxx_data[pix^2], myy_data[pix^2], mxy_data[pix^2], mxx_model[pix^2], myy_model[pix^2], mxy_model[pix^2]"
    cvs_fname = get_output_fname( imsim_file, free_pars_index, ntry, dofs_fake, 'map' )
    np.savetxt( cvs_fname, output_arr, delimiter=",", header=header, comments='')

### TODO: 
###   - we need to read a jason result and plot the results without fitting


    if( noverbose ):
        pass
        
    else:
        print( 'result:',res )
        print( '->:     ',res.x )
        print( 'x0 winner:', x0_winner )
        print( 'freepars:', free_pars_index )
        
        for i in range(50):
            if( dofs_found[ i ] != 0 or dofs_fake[ i ] != 0):
                print( "%.2d \t %.3f \t %.3f " %(i, dofs_fake[ i ], dofs_found[ i ] ) )

    if( noplot ):
        pass
    else:
        if n > maxRows:
            fitted_map_plot = fitted_map[indices,:]
        else:
            fitted_map_plot = fitted_map
    
        plot_ellipticity_map( xplt, yplt, fitted_map_plot, 1.7, 
                             scale=None, 
                             saveit=False, 
                             title='model data', maxRows=1500 )
        plt.show()
        
        # new fancy plot for model
        lm.dof[ free_pars_index ] = res.x
        lm.ellipticity_compute_algebra()
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), sharex=False, sharey=False)
        makeAzElPlot(fig, axes, hx, hy, lm.ellipticity )
        plt.show()
        
        # new fancy plot for data... need to reconstruct ellip dictionary
        # ellip_dic_data = get_ellip_dic( data_map, hx, hy )
        # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), sharex=False, sharey=False)
        # makeAzElPlot(fig, axes, hx, hy, ellip_dic_data )
        # plt.show()
        
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), sharex=False, sharey=False)
        makeAzElPlot(fig, axes, hx, hy, data_dic )
        plt.show()
        
        # ### debug misterious angle.
        # ikeep = (( hx > -0.5 ) & ( hx < -0.3 ) & ( hy < 1.5 ) &( hy > 1.3) )
        # print( '***-------- good e1,e2:')
        # print( 'hx:', hx[ikeep])
        # print( 'hy:', hy[ikeep])
        # print( 'e1:', data_dic['e1'][ikeep])
        # print( 'e2:', data_dic['e2'][ikeep])
        # print( 'pa:', np.rad2deg( 0.5* np.arctan2(data_dic['e2'][ikeep], data_dic['e1'][ikeep])) )
        # print( 'e:', np.hypot( data_dic['e1'][ikeep], data_dic['e2'][ikeep] ))
        # print( '***-------- bad e1,e2:')
        # print( 'e1:', ellip_dic_data['e1'][ikeep])
        # print( 'e2:', ellip_dic_data['e2'][ikeep])
        # print( 'pa:', np.rad2deg( 0.5* np.arctan2(ellip_dic_data['e2'][ikeep], ellip_dic_data['e1'][ikeep])) )
        # print( 'e:', np.hypot( ellip_dic_data['e1'][ikeep], ellip_dic_data['e2'][ikeep] ))
        
    if( not noplot ):
        q_mod = data_map[:,0]
        pa_mod= data_map[:,1]
        p_mod = data_map[:,2]
        
        q_obs = fitted_map[:,0]
        pa_obs= fitted_map[:,1]
        p_obs = fitted_map[:,2]

        pp_mod = p_mod - np.median( p_mod )
        pp_obs = p_obs - np.median( p_obs )
        
        k0 = 0.183         # empirical scaling constant ARCSEC
        k0 = k0 / 0.2      # PIXEL as q and p
        
        el_mod = k0**2 * np.log( 1 + q_mod/k0**2 )
        el_obs = k0**2 * np.log( 1 + q_obs/k0**2 )
        
        vx_mod = el_mod * np.cos( np.deg2rad( pa_mod ))
        vy_mod = el_mod * np.sin( np.deg2rad( pa_mod ))
        
        vx_obs = el_obs * np.cos( np.deg2rad( pa_obs ))
        vy_obs = el_obs * np.sin( np.deg2rad( pa_obs ))
        
        cost_arr = ( ( pp_mod - pp_obs )**2/4 +
                (vx_mod - vx_obs)**2 +
                (vy_mod - vy_obs)**2 )
        

    return dofs_found, res

def get_output_fname( imsim_file, index_free_pars, ntry, dofs0, tipo ):
    nfree = len( index_free_pars )
    #avoid overwritting tests with same nfree but different free par indices
    if( nfree <= 10 ):
        imrigid = np.where( index_free_pars < 10 )[0]
        if( len( imrigid ) == nfree ):
            #convert index_free_pars to a string
            indexstr = ''.join( [ '%.1d' %i for i in index_free_pars ] )
            indexstr = '_' + indexstr
        else:
            indexstr = ''
    else:
        indexstr = ''


    # tipo can be: map, result, tries
    if( dofs0 is None ):
        ndofs = 0
    else:
        ndofs = len( np.where( dofs0 != 0 )[0] )

    if( tipo == 'map'):
        myext = '.cvs'
        mystr = '_map'
    elif( tipo == 'result' ):
        myext = '.json'
        mystr = ''
    elif( tipo == 'tries' ):
        myext = '.json'
        mystr = '_sol'
    else:
        print( 'error calling get_output_fname. tipo needs to be: map, result, tries')
        sys.exit(1)

    myfile = pathlib.Path( imsim_file )
    if( myfile.suffix == '.bat' or myfile.suffix == '.csv'):
        base = str(myfile).replace( myfile.suffix, '')
    elif( imsim_file in ['mod','ray','fft'] ):
        base = 'data/simulation/' + imsim_file
    else:
        base = imsim_file

    output_fname = base + '_nfit_%.2d'%nfree + '_nini_' +'%.2d'%ndofs + '_ntry_' + '%.4d' %ntry + indexstr + mystr+ myext
    
    return output_fname

def get_result_json_data( res, ntry, free_pars_index, dofs0, imsim_file ):
    json_data = {}
    json_data['r_message'] = res.message
    json_data['r_success'] = res.success
    json_data['r_status'] = res.status
    json_data['r_x'] = res.x.tolist()
    json_data['r_fun'] = res.fun
    json_data['r_nit'] = res.nit
    json_data['r_nfev'] = res.nfev
    json_data['r_direc'] = res.direc.tolist()
    
    json_data['input_file'] = imsim_file
    json_data['ntry'] = ntry
    json_data['free_pars_index'] = free_pars_index.tolist()
    
    json_data['ndofs'] = len( np.where( dofs0 != 0 )[0] ) if dofs0 is not None else 0
    json_data['index_dofs0'] = np.where( dofs0 != 0 )[0].tolist() if dofs0 is not None else []
    json_data['dofs0'] = dofs0[ np.where(dofs0!=0)[0] ].tolist() if dofs0 is not None else []

    # save all non zero dofs regardless of origin (fitted or fixed) to replicate plots/solution
    dofs_full = np.zeros( 50 )
    if( dofs0 is not None ):
        dofs_full[ np.where( dofs0 != 0 )[0] ] = dofs0[ np.where( dofs0 != 0 )[0] ]
    dofs_full[ free_pars_index ] = res.x

    json_data['index_dofs_full'] = str( np.where( dofs_full != 0 )[0].tolist() )[1:-1]    #to replicate solution
    json_data['dofs_full'] = str( dofs_full[ np.where(dofs_full!=0)[0] ].tolist() )[1:-1]
    
    return json_data

def get_tries_json_data( res_list, x0_list, free_pars_index  ):

    # res is a list of results
    json_data = {}
    nsol = len( res_list )
    json_data['n_solutions'] = nsol
    sol_list = []
    for i in range( nsol ):
        resi = res_list[i]
        sol_dic = {}
        sol_dic['r_message'] = resi.message
        sol_dic['r_success'] = resi.success
        sol_dic['r_status'] = resi.status
        sol_dic['r_x'] = resi.x.tolist()
        sol_dic['r_fun'] = resi.fun
        sol_dic['r_nit'] = resi.nit
        sol_dic['r_nfev'] = resi.nfev
        sol_dic['r_direc'] = resi.direc.tolist()
        sol_list.append( sol_dic )
    json_data['solutions'] = sol_list
    json_data['initial_guesses'] = [ x0.tolist() for x0 in x0_list ]

    json_data['free_pars_index'] = free_pars_index.tolist()
     
    return json_data

def read_from_json( json_file ):
    with open( json_file, 'r' ) as jf:
        json_data = json.load( jf )

    x = np.asarray( json_data['r_x'] )
    imsim_file = json_data['input_file']
    ntry = json_data['ntry']
    free_pars_index = np.asarray( json_data['free_pars_index'] )
    ndofs = json_data['ndofs']
    if( ndofs > 0 ):
        dofs0 = np.zeros( 50 )
        index_dofs0 = np.asarray( json_data['index_dofs0'] )
        dofs0[ index_dofs0 ] = np.asarray( json_data['dofs0'] )
    else:
        dofs0 = None
        
    return imsim_file, ntry, free_pars_index, dofs0, x

if __name__=='__main__':

    ntry, free_pars_index, dofs_fake, imsim_file, noplot, noverbose, oldrun, x0par, ix0par = parse_inputs()

    print( x0par)
    print( ix0par)

    print( oldrun)
    if( oldrun):
        imsim_file, ntry, free_pars_index, dofs0, x = read_from_json( oldrun )
        print( imsim_file, ntry, free_pars_index, dofs0, x )
        sys.exit(0)

    if( dofs_fake is None ):
        dofs0 = np.zeros( 50 )
    else:
        dofs0 = dofs_fake.copy()

    dofs_found, res = minimize_map( ntry, free_pars_index, dofs0, imsim_file, noplot, noverbose, x0par, ix0par)

    json_fname = get_output_fname( imsim_file, free_pars_index, ntry, dofs_fake, 'result' )
    print( json_fname )
    json_data = get_result_json_data( res, ntry, free_pars_index, dofs_fake, imsim_file  )

    with( open( json_fname, 'w' ) ) as jf:
        json.dump( json_data, jf, indent=4 )

    if( pathlib.Path( imsim_file ).suffix == '.bat'):
        dofs_fake = np.loadtxt( imsim_file, delimiter=',', skiprows=0, max_rows=1 )
    elif( pathlib.Path( imsim_file ).suffix == '.csv' ):
        try:
            dofs_fake = np.loadtxt( imsim_file, delimiter=',', skiprows=0, max_rows=1 )
        except ValueError:
            dofs_fake = None
        
        
    if( dofs_fake is None ):
        for i in range(50):
            if( dofs_found[ i ] != 0 ):
                print( "%.2d, %.3f, ?," %(i, dofs_found[ i ] ), end=' ',flush=True )
        print()
    else:
        for i in range(50):
            if( dofs_found[ i ] != 0 or dofs_fake[ i ] != 0):
                print( "%.2d, %.3f, %.3f," %(i, dofs_fake[ i ], dofs_found[ i ] ), end=' ',flush=True )
        print()
    
        

