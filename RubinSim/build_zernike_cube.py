#!/usr/bin/env python

# we run batoid to get zernike coefficients for a series of
#   wavelenghts
#       zeta (dof, degrees of freedom)
#           nfield points Hx,Hy
# the results are saved to a HDF5 file.

from util import zernike_optimal_sampling, fit_zernike_coefficients
import numpy as np
import h5py
import batoid
from batoid_rubin import builder
import timeit
from pathlib import Path
import sys


# wavelength = np.array( [0.384, 0.481, 0.622, 0.770, 0.895, 0.994]) * 1e-6
wavelength = np.array( [0.384, 0.481]) * 1e-6

wave_files = ["LSST_u.yaml", "LSST_g.yaml", "LSST_r.yaml",
              "LSST_i.yaml", "LSST_z.yaml", "LSST_y.yaml"]
field_radius = 1.708    #degrees.


nmax_zernike = 22
npert = 4           # 4 points for each pertubation
# perturbations, dof. max amplitudes for dof
#                dz     dx      dy     rx     ry    dz    dx      dy      rx    ry
amp = np.array([25.0, 1000.0, 1000.0, 25.0, 25.0, 25.0, 4000.0, 4000.0, 25.0, 25.0])

def get_zeta_arr( amp, nrow ):
    """Build an array with nrow linear distorsions for each distorsion on 
        the amp array.

    Parameters
    ----------
    amp : float array 
        n distorsion elements, e.g. dx, dy, dz, Rx, Ry for M2 and Camera
    nrow: int
        number of linear distorsions. Example, if amp for dx is 1000 and
    nrow is 4, we return for that distorsion [250, 500, 750, 1000].
    
    Returns
    -------
    zeta : array
        nrow x len( amp )  array

    Notes
    -----
    
    """

    zeta = np.zeros( (nrow, len( amp ) ))

    for i in range( len( amp ) ):
        zeta_i = np.linspace( 0, amp[i], num = nrow + 1)[1:]
        zeta[:,i] = zeta_i

    return zeta

def get_zernike_for_hxhy( hx, hy, tel1, wl, nmax ):
    """fits nmax zernike coefficients for each point hx, hy for the
    distorted telescope tel1 for wavelength wl.

    Parameters
    ----------
    hx, hy : array
        field points
    tel1: batoid optic
        the telescope represented with batoid
    wl: float
        wavelength in meters
    nmax: int
        number of zernike coefficients

    
    Returns
    -------
    znk_arr : array
        zernike coefficients. 

    Notes
    -----
    The returned array has nmax+1 elements because in
    galsim convention, the first element of the array is always 0 to avoid
    the 0-index.
    
    """
    znk_arr = []        # collect   nznk x nfield
    for x, y in zip( hx, hy ):
        # for testing, we simulate some numbers
        # znk = np.linspace( 1, nmax, num = nmax )
        # #run batoid with this inputs and get the zernike terms
        znk = fit_zernike_coefficients( tel1, 
                      np.deg2rad(x), np.deg2rad( y ), wl, nmax )
        znk_arr.append( znk )

    znk_arr = np.asarray( znk_arr )
    return znk_arr

def get_zernike_for_zeta_hxhy( zeta, hx, hy, mybuilder, wl, nmax ):
    """fits nmax zernike coefficients for each point hx, hy for the
    distorted telescope tel1 for wavelength wl.

    Parameters
    ----------
    zeta: array
        an 2D array with nrow different distorsions for each of the ncol
    different distorsions (dx, dy, dz, Rx, Ry ...)
    hx, hy : array
        field points
    mybuilder: batoid builder
        a batoid builder that can distort the optical array of a previously
    created batoid optic
    wl: float
        wavelength in meters
    nmax: int
        number of zernike coefficients

    
    Returns
    -------
    cube : 3D array
        zernike coefficients. 

    Notes
    -----
    A cube nfield x nmax x ndistorsions with zernike coefficients.
    
    """

    cube = []                       # collect ndist x nznk x nfield 
    for i in range( zeta.shape[1] ):   #loop over diff. kind of distorsion (10)
        zeta_arr_i = zeta[:,i]

        for dof_i in zeta_arr_i:    # loop over magnitude of distorsion (4)
            # Now disturb telescope
            dof = np.zeros( 50 )
            dof[ i ] = dof_i
            mybuilder = mybuilder.with_aos_dof( dof )
            tel1 = mybuilder.build()

            znk_arr = get_zernike_for_hxhy( hx, hy, tel1, wl, nmax )

            cube.append( znk_arr )

    cube = np.transpose( np.asarray( cube ) )
    return cube


if __name__ == "__main__":

    fname = "test_znk_w1_z11.hdf5"
    mypath = Path( fname )
    if( mypath.is_file() ):
        print( "File %s found." %(fname ) )
        input( "Continue? [ctrl+C] to exit" )
    
    #open output file
    f = h5py.File( fname,'w')

    # disalignments
    zeta = get_zeta_arr( amp, npert )
    # field points
    hx, hy, rho, theta = zernike_optimal_sampling( nmax_zernike )
    hx = hx * field_radius      #scale to cover the whole FOV. Batoid uses rads
    hy = hy * field_radius      #zemax uses normalized field? 0 to 1
    
    ## for debugging, only take a tenth of the field points
    # if( hx.size > 10):
    #     print('*** dbg: slicing field points to a tenth')
    #     hx = hx[0:-1:10]
    #     hy = hy[0:-1:10]
    #     rho= rho[0:-1:10]
    #     theta=theta[0:-1:10]
    
    tic0=timeit.default_timer()
    print( "optimal number of points is %d for %d zernike terms" %(hx.size, nmax_zernike))
    coords = np.vstack( (hx,hy) )
    wl_cnt = 0
    for wl in wavelength:
        print( wl )
        print("%s" %wave_files[ wl_cnt ])

        # create optic for this wavelength
        fiducial = batoid.Optic.fromYaml(wave_files[ wl_cnt ])
        mybuilder = builder.LSSTBuilder(fiducial, "fea_legacy", "bend_legacy" )
        
        # nominal case once per wl
        dof = np.zeros( 50 )                
        mybuilder = mybuilder.with_aos_dof( dof )   
        tel1 = mybuilder.build()    
        
        tic=timeit.default_timer()
        znk_nominal_arr = get_zernike_for_hxhy( hx, hy, tel1, wl, nmax_zernike )
        toc = timeit.default_timer()
        
        print( toc - tic, " seconds for one case" )
        print( "I predict %f seconds total" %((toc-tic)*zeta.size*len(wavelength)) )
        
        grp_current = f.create_group("wl%d" %( wl_cnt + 1) )

        cube = get_zernike_for_zeta_hxhy( zeta, hx, hy, mybuilder, wl, nmax_zernike )
        cube = np.transpose( cube, axes=[1,0,2])

        
        dset_current = grp_current.create_dataset("cube", data = cube )
        dset_current = grp_current.create_dataset("coords", data = coords )
        dset_current = grp_current.create_dataset("nominal", data = znk_nominal_arr )
        dset_current = grp_current.create_dataset("zeta", data = zeta )
        grp_current.attrs['wl'] = wl
        grp_current.attrs['nfield'] = len(hx)
        grp_current.attrs['nzernike'] = nmax_zernike
        grp_current.attrs['n_dof'] = len( amp )
        grp_current.attrs['npert'] = npert
        grp_current.attrs['field_radius'] = field_radius
        wl_cnt += 1

    print( cube.shape )
   
    f.close()

    tac = timeit.default_timer()
    print( tac - tic0, " total seconds" )            




#TODO: cube final transpose in line 198 should be done insde get_zernike_for_zeta_hxhy





