#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:34:44 2024

@author: zanmar
"""

# run a bunch of random misalignemnt and compare coeffs and WFs of 
#   model and batoid

import sys
import numpy as np
import lsstModel
import fromBatoid
import pickle
import pathlib
import matplotlib.pyplot as plt
# from plot_bat_mod_psf import plot_one_psf
from util import psf_fft_coeffs, plot_one_psf

rng = np.random.default_rng(seed=777)
    
def get_random_dof( amps, all = False ):
    
    if( all ):
        dof = ( rng.random(10) * 2 - 1 ) * amps     # from -1 to 1
    else:
        dof = np.zeros( 10 )
        one = rng.random() * 2 - 1                  # from -1 to 1
        i_rnd = rng.integers(0, 10 )
        dof[ i_rnd ] = one * amps[ i_rnd ]
        
    return dof

def get_random_coord( fov ):
    """
    return a random coordinate in degrees inside the FOV circle

    Parameters
    ----------
    fov : TYPE
        field of view, radius in degrees

    Returns
    -------
    Two numpy arrays of dimension 1.

    """
    hx=hy=fov
    while( np.hypot( hx, hy ) >=fov ):
        hx = ( rng.random() * 2 - 1 ) * fov
        hy = ( rng.random() * 2 - 1 ) * fov
    
    return np.array( [ hx ]), np.array( [hy] )
    
def rms(array):
   return np.sqrt( np.mean( array ** 2) )

def norm( array ):
    # euclidian norm taking into account array's mask
    return np.sqrt( (array*array).sum() )
    
def run_simulations( lm, fb, mal, nsim ):
    # run nsim random simulations, batoid vs model WF.
    # return a dictionary with the results for later stats and plots
    # to save results
    norm_ratios_all = np.zeros( nsim )
    dof_list = []
    hx_arr = np.zeros( nsim )
    hy_arr = np.zeros( nsim )
    coeffs_bat = []
    coeffs_mod = []
    coeffs_rms = np.zeros( nsim )
    
    
    for i in range( nsim ):
        # generate random zz (at first one at a time, later 10 random dof )
        dof = get_random_dof( mal._max_amplitude, all = True )
        dof_list.append( dof )
        
        # generate random hx,hy
        hx, hy = get_random_coord( fb.field_radius )
        hx_arr[ i ] = hx[0]
        hy_arr[ i ] = hy[0]
        
        # generate random wl? or one at a time. 
        iwl = 1
        
        # get coeffs with batoid
        fb.wl_index = iwl
        fb.dof = np.pad( dof, (0,40), 'empty' )
        fb.hx, fb.hy = hx, hy
        bat_coeff = fb.zernike_coeffs()
        coeffs_bat.append( bat_coeff[0] )
        
        # get coeffs with model
        lm.wl_index = iwl
        lm.dof = dof
        lm.set_hxhy(hx, hy)
        mod_coeff = lm.zernike_coeffs()
        coeffs_mod.append( mod_coeff[0] )
        
        myrms = rms( bat_coeff[0,3:] - mod_coeff[0,3:] )
        coeffs_rms[ i ] = myrms
        
        diff, wf1, wf2 = lsstModel.WF_difference(bat_coeff[0,:], mod_coeff[0,:], 
                                                 mode = 2 )
        
        # new metric: norm of diff divided by norm of wf1
        norm_ratio = norm( diff ) / norm( wf1 ) * 100.0
        norm_ratios_all[ i ] = norm_ratio
        
        print( "%6d %7.1e %7.2f %7.2f %7.3f %7.2f" %(i, myrms, hx[0], hy[0], 
                                          norm_ratio, norm( wf1 ) ) )


    
    results = {'dof_list':dof_list, 'hx_arr': hx_arr, 'hy_arr': hy_arr, 
               'norm_ratios': norm_ratios_all, 'rms': coeffs_rms,
               'mod': coeffs_mod, 'bat': coeffs_bat }
    return results

def plot_histogram( array ):
    fig, ax = plt.subplots()
    
    ax.hist( array, bins=50 )
    ax.set_ylabel("N")
    ax.set_xlabel("% difference")
    # plt.xlim(0,0.004)
    plt.show()
    
def plot_coords( hx, hy ):
    fig, ax = plt.subplots()
    
    ax.plot( hx, hy, '.', markersize=0.5  )
    ax.set_ylabel("hx [deg]")
    ax.set_xlabel("hy[ deg]")
    ax.set_aspect('equal', 'box')
    # plt.xlim(0,0.004)
    plt.show()
    
def plot_dofs( lista, maxamp ):
    
    array = np.asarray( lista ) / maxamp #normalized
    n_points = array.shape[0]
    rnd_i = rng.random( n_points ) * 0.6 - 0.3
    
    fig, ax = plt.subplots()
    
    for i in range( maxamp.size ):
        dof_index = rnd_i + i
        ax.plot( dof_index, array[:,i], '.', markersize=0.1  )
        
        
    ax.set_ylabel("normalized misalignment")
    ax.set_xlabel("dof [dz dx dy Rx Ry | dz dx dy Rx Ry]")
    # ax.set_aspect('equal', 'box')
    # plt.xlim(0,0.004)
    plt.show()
    
def plot_diff_vs_dof( diff, lista, maxamp ):
    dofs = np.asarray( lista ) / maxamp #normalized
    
    dof_label = ['M2 dz', 'M2 dx', 'M2 dy', 'M2 Rx', 'M2 Ry',
                 'Cam dz', 'Cam dx', 'Cam dy', 'Cam Rx', 'Cam Ry']
    

    
    for i in range( maxamp.size ):
        fig, ax = plt.subplots()
        ax.plot( dofs[:,i], diff, '.', markersize=0.5  )
        ax.set_xlabel("dof %d: %s" %(i, dof_label[i] ))
        ax.set_ylabel("% difference")
        plt.show()
    
    mag_dof = np.linalg.norm( dofs, ord=2, axis=1 ) #hypotenuse
    
    fig, ax = plt.subplots()
    ax.plot( mag_dof, diff, '.', markersize=0.5  )
    ax.set_xlabel("dof norm" )
    ax.set_ylabel("% difference")
    plt.show()
    
    
    
    
    # ax.set_aspect('equal', 'box')
    # plt.xlim(0,0.004)
    
    
    

if __name__=='__main__':
    nsim = 10000
    
    # fname = 'znk_batoid_coeffs_wl_2_jmax_22_dbg.hdf5'
    # fname = 'znk_batoid_coeffs_wl_6_jmax_22.hdf5'
    # fname = 'znk_batoid_coeffs_wl_2_jmax_37_dbg.hdf5'
    fname = 'znk_batoid_coeffs_wl_6_jmax_37.hdf5'
    
    outpickle = 'res_bat_mod_'+fname.replace('znk_batoid_coeffs_','').replace('.hdf5','')+'.pkl'
    print( outpickle )
    
    mal = fromBatoid.MisAlignment()
    print( mal._max_amplitude )

    if( pathlib.Path( outpickle ).is_file() ):
        res = pickle.load(open(outpickle, "rb"))
    else:
    
        lm = lsstModel.lsstModel( batoid_cube_file=fname, n_fldznk=22)
        print( lm.batoid_cubes[0] )
        
        if( 'wl_2' in fname ):
            debug = True
        else:
            debug = False
        
        fb = fromBatoid.FromBatoid(jmax=lm.batoid_cubes[0].nznkpupil, debug=debug)
        
        res = run_simulations( lm, fb, mal, nsim )
        
    
        
        pickle.dump( res, open( outpickle, 'wb'))
    
    print( 'mean % error:', res['norm_ratios'].mean() )
    # plot worst case WF
    ibad = res['norm_ratios'].argmax()
    print('worst index and value:', ibad, res['norm_ratios'][ibad] )
    lsstModel.plot_WF( res['bat'][ibad], res['mod'][ibad], residual=False, mode=2 )
    
    # plot worst case PSF
    psf_coeffs_ric = psf_fft_coeffs(  res['mod'][ibad], 360 )
    plot_one_psf( psf_coeffs_ric, 'batoid')     # PSF with coeffs
    psf_coeffs_ric = psf_fft_coeffs(  res['mod'][ibad], 360 )
    plot_one_psf( psf_coeffs_ric, 'model')     # PSF with coeffs
    
    
    
    
    plot_histogram( res['norm_ratios'] )

    plot_coords( res['hx_arr'], res['hy_arr'] )

    plot_dofs( res['dof_list'], mal._max_amplitude )
    