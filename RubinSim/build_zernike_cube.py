#!/usr/bin/env python

# we run batoid to get zernike coefficients for a series of
#   -wavelenghts
#   -zeta (dof, degrees of freedom)
#   -nfield points Hx,Hy
# the results are saved to a HDF5 file.

import numpy as np
import h5py
import timeit
from pathlib import Path
from util import zernike_optimal_sampling
from fromBatoid import FromBatoid, MisAlignment
import argparse

def save_to_hd5( list_nominals, list_cube, fb, mal, debug=False ):
    if( debug ):
        fname = "znk_batoid_coeffs_wl_%d_jmax_%d_dbg.hdf5" %(fb.n_wl, fb.jmax)
    else:
        fname = "znk_batoid_coeffs_wl_%d_jmax_%d.hdf5" %(fb.n_wl, fb.jmax)
        
    mypath = Path( fname )
    if( mypath.is_file() ):
        print( "File %s found." %(fname ) )
        input( "Continue? [ctrl+C] to exit" )
            
    f = h5py.File( fname,'w')
    for wl_cnt in range( fb.n_wl ):
        grp_current = f.create_group("wl%d" %( wl_cnt + 1) )
        
        grp_current.create_dataset("cube", data = list_cube[ wl_cnt ][1] )
        grp_current.create_dataset("coords", data = np.vstack( ( fb.hx, fb.hy ) ) )
        grp_current.create_dataset("nominal", data = list_nominals[ wl_cnt ][1] )
        grp_current.create_dataset("zeta", data = mal.zeta )
        assert wl_cnt == list_cube[ wl_cnt ][0]
        assert wl_cnt == list_nominals[ wl_cnt ][0]        
        grp_current.attrs['wl'] = fb.wl_array[ wl_cnt ]
        grp_current.attrs['nfield'] = len( fb.hx )
        grp_current.attrs['nzernike'] = fb.jmax
        grp_current.attrs['n_dof'] = mal.n_dof
        grp_current.attrs['npert'] = mal.nperturbations
        grp_current.attrs['field_radius'] = fb.field_radius
        
    f.close()
    print( "Saved %s" %(fname) )
    return

def get_pars():
    parser = argparse.ArgumentParser()

    """
       debug == True:    only a reduced number of field points and 2 wavelengths
       jmax:             number of zernike coefficients to represent pupil WF
    """

    parser.add_argument('--jmax', help='jmax number of zernike coefficients', type=int, default=37, choices=[11,22,37] )
    parser.add_argument("-d", "--debug", help="2 wavelengths only and a reduced number of field points",
                    action="store_true")
    
    args = parser.parse_args()
    
    if args.debug:
        debug = True
    else:
        debug = False

    jmax = args.jmax 

    return jmax, debug


if __name__ == "__main__":
    
    jmax, debug = get_pars()
    
    print( 'jmax=', jmax )
    
    fBat = FromBatoid(jmax=jmax, debug=debug)
    
    mal = MisAlignment()
        
    # H
    hx, hy, _, _ = zernike_optimal_sampling( fBat.jmax, debug=debug )
    
    hx_deg, hy_deg = hx * fBat.field_radius, hy * fBat.field_radius
    fBat.hx, fBat.hy = hx_deg, hy_deg
    
    tic=timeit.default_timer()
    list_nominals = fBat.get_list_nominal_znk_coeffs()
    toc = timeit.default_timer()
    
    print("Nominals took %d seconds" %(toc - tic))
    print("Estimated total = %d" %( mal.zeta.size * (toc - tic)  ) )
    
    list_cube = fBat.get_list_cube_znk_coeffs( mal.list )
    
    tac = timeit.default_timer()
    print( "Actual time = %d" %(tac -tic ) )
    
    save_to_hd5( list_nominals, list_cube, fBat, mal, debug=debug )

    