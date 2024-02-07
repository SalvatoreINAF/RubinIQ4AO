#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:02:24 2024

@author: zanmar
"""

from util import zernike_optimal_sampling, fit_zernike_coefficients
from build_zernike_cube import get_zeta_arr
import batoid
from batoid_rubin import builder
import numpy as np
import h5py
from pathlib import Path
import sys
import timeit

class FromBatoid:
    
    def __init__( self, wl_index = 0, jmax = 22 ):
        # default telescope, etc
        self.jmax = jmax                  # zernike pupil terms
        self.field_radius = 1.708       # degrees
        if __debug__ :
            print("*** dbg: only 2 wavelengths")
            self.wl_array = np.array( [0.384, 0.481]) * 1e-6
            self.wave_files = ["LSST_u.yaml", "LSST_g.yaml"]
        else:
            self.wl_array =np.array([0.384,0.481,0.622,0.770,0.895,0.994])*1e-6
            self.wave_files = ["LSST_u.yaml", "LSST_g.yaml", "LSST_r.yaml",
                               "LSST_i.yaml", "LSST_z.yaml", "LSST_y.yaml"]
        self.n_wl = len( self.wl_array )
        self._dof = np.zeros( 50 )               #nominal case
        self.hx = np.array([0.0])                 # in degrees
        self.hy = np.array([0.0])
        
        assert 0 <= wl_index < len( self.wl_array ), "invalid current_wl"
        
        self._wl_index = wl_index
        self._wl = self.wl_array[ self._wl_index ]  # current wavelength
        self._optic = None
        self._builder = None
        self._telescope = None
        
        self.psf = None
        self.ellipticity = None
        self.update_telescope()
        
        
    @property
    def wl_index( self ):
        return self._wl_index
        
    @wl_index.setter
    def wl_index( self, value ):
        assert 0 <= value < len( self.wl_array ), "invalid wl index"
        if( value != self._wl_index ):
            #print( 'got new wl index, need to update optic/builder/telescope')
            self._wl_index = value
            self._wl = self.wl_array[ self._wl_index ]
            self.update_telescope()      
        else:
            #print( 'same wl index as before.')
            pass
        
        # self._wl_index = value
    
    @property
    def dof( self ):
        return self._dof
    
    @dof.setter
    def dof( self, value ):
        assert isinstance(value, np.ndarray ) and value.size == 50, \
            'expected 1D numpy array of size 50'
        if( not np.allclose( value,self._dof ) ):
            print( 'new dof, update builder/telescope' )
            self._dof = value
            self.rebuild_telescope_with_dof()
        """            AOS degrees of freedom.
                    0,1,2 are M2 z,x,y in micron
                    3,4 are M2 rot around x, y in arcsec
                    5,6,7 are camera z,x,y in micron
                    8,9 are camera rot around x, y in arcsec"""
    @property
    def wl( self ):
        return self._wl
    
            
    def update_telescope( self ):
        #print('updating optic, builder, telescope')
        self._optic = batoid.Optic.fromYaml(self.wave_files[ self.wl_index ])
        self._builder = builder.LSSTBuilder(self._optic, "fea_legacy", "bend_legacy")
        self.rebuild_telescope_with_dof()
        
        
    def rebuild_telescope_with_dof( self ):
        self._builder = self._builder.with_aos_dof( self._dof )
        self._telescope = self._builder.build()

    def get_tel( self ):
        return self._telescope
    
        
    def zernike_coeffs( self ):
        #TODO. -> change to use batoid's zernike method.
        
        znk_arr = []        # collect   nznk x nfield
        for x, y in zip( self.hx, self.hy ):
            znk = fit_zernike_coefficients( self._telescope, 
                    np.deg2rad(x), np.deg2rad( y ), self.wl, self.jmax )
            znk_arr.append( znk )

        znk_arr = np.asarray( znk_arr )
        return znk_arr
        
    def zernike_coeffs_batoid( self ):
        # batoid offers its own method, is it different?
        """
        this is giving very different and inconsistent results... why!?
        """
        znk_arr = []
        
        
        for x, y in zip( self.hx, self.hy ):
            znk = batoid.analysis.zernike( self._telescope, 
                    np.deg2rad(x), np.deg2rad( y ), 
                    self.wl, reference='chief', jmax=self.jmax, eps=0.61
                    )
            znk_arr.append( znk )

        znk_arr = np.asarray( znk_arr )
        return znk_arr
    
    def wavefront( self ):
        wf_arr = []
        # mask_arr = []
        
        for x, y in zip( self.hx, self.hy ):
            opd = batoid.wavefront(
                    self._telescope,
                    np.deg2rad( x ), np.deg2rad( y ),
                    self.wl,
                    nx=255
                )
            wf_arr.append( opd.array )
            # mask_arr.append( opd.array.mask )
            
        # wf_arr = np.asarray( wf_arr )
        # mask_arr = 
        return wf_arr
    
    def psfFFT_compute( self ):
        psf_list = []
        
        for x, y in zip( self.hx, self.hy ):
            psf = batoid.analysis.fftPSF(self._telescope, 
                                         np.deg2rad(x), np.deg2rad(y), 
                                         self.wl, nx=64,
                                   pad_factor=8, reference='mean')    
            psf_list.append( psf )
            
        self.psf = psf_list
        
        return
    
    def ellipticity_compute( self ):
        
        sl = []
        ss = []
        el = []
        pa = []
        xc = []
        yc = []
        cnt = 0
        for psf in self.psf:
            psfnorm = psf.array / psf.array.max()
            X = psf.coords[:,:,1]
            Y = psf.coords[:,:,0]
            suma = psfnorm.sum()
        #centroid
            x_cen = ( psfnorm * X ).sum() / suma
            y_cen = ( psfnorm * Y ).sum() / suma
        #moments 2nd
            mu_xx = ((X - x_cen)**2 * psfnorm ).sum() / suma
            mu_yy = ((Y - y_cen)**2 * psfnorm ).sum() / suma
            mu_xy = ((X - x_cen) * ( Y - y_cen ) * psfnorm ).sum() / suma
        #ellipticity
            p = mu_xx + mu_yy
            q = np.sqrt( (mu_xx - mu_yy)**2 + 4 * mu_xy**2 )
            sigma_l = np.sqrt( ( p + q ) / 2 )
            sigma_s = np.sqrt( ( p - q ) / 2 )
            ellipticity_adimensional = 1 - sigma_s/sigma_l
            alpha = np.rad2deg( np.arctan2( 2 * mu_xy, mu_xx - mu_yy ) / 2 )
            
            # print( q * np.cos( alpha ), q * np.sin( alpha ), mu_xx, mu_yy, mu_xy )
            print("%.3d %5.2f %10.2f" %( cnt, ellipticity_adimensional, alpha ) )
            cnt+=1
            
            sl.append( sigma_l)
            ss.append( sigma_s )
            el.append( ellipticity_adimensional )
            pa.append( alpha )          # degrees
            xc.append( x_cen )
            yc.append( y_cen )
            
        self.ellipticity = {'sl': sl, 'ss': ss, 'el': el, 
                         'pa': pa, 'xc': xc, 'yc': yc}
            
        return 
            
            
            
            
            
            
            
            
            
        
        
        

def get_dof_list( amp, npert ):
    lista = []
    for i in range( len( amp ) ):
        zeta_i = np.linspace( 0, amp[i], num = npert + 1)[1:]
        for j in zeta_i:
            dof = np.zeros( 50 )
            dof[ i ] = j
            lista.append( dof )
    return lista
    
def get_list_nominal_znk_coeffs( fb ):
    
    assert np.allclose( fb.dof, np.zeros(50) ), "you want nominal but current dof is not zero"
    res = []
    for i in range( fb.n_wl ):
        fb.wl_index = i
        res.append( (i, fb.zernike_coeffs() ) )
        
    return res

def get_list_cube_znk_coeffs( fb, dof_list ):
    
    list_cube = []
    for i in range( fb.n_wl ):
        cube = []                       # collect ndist x nznk x nfield 
        for dof in dof_list:
            # print( dof[0:10] )
            fb.dof = dof
            cube.append( fb.zernike_coeffs() )
        
        cube = np.transpose( np.asarray( cube ) )   #expected target shape
        list_cube.append( (i, cube ) )
    return list_cube

def save_to_hd5( list_nominals, list_cube, fb, mal ):
    if(__debug__):
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

class MisAlignment():
    def __init__( self ):
        self._max_amplitude = np.array([25.0, 1000.0, 1000.0, 25.0, 25.0, 25.0, 4000.0, 4000.0, 25.0, 25.0])
        self.nperturbations = 4
        self.list = get_dof_list( self._max_amplitude, self.nperturbations )
        self.zeta = get_zeta_arr( self._max_amplitude, self.nperturbations )
        self.n_dof = len( self._max_amplitude )
            
        

if __name__== '__main__':
    
    fBat = FromBatoid()
    
    mal = MisAlignment()
        
    # H
    hx, hy, _, _ = zernike_optimal_sampling( fBat.jmax )
    
    hx_deg, hy_deg = hx * fBat.field_radius, hy * fBat.field_radius
    fBat.hx, fBat.hy = hx_deg, hy_deg
    
    tic=timeit.default_timer()
    list_nominals = get_list_nominal_znk_coeffs( fBat )
    toc = timeit.default_timer()
    
    print("Nominals took %d seconds" %(toc - tic))
    print("Estimated total = %d" %( mal.zeta.size * (toc - tic)  ) )
    
    list_cube = get_list_cube_znk_coeffs( fBat, mal.list )
    
    tac = timeit.default_timer()
    print( "Actual time = %d" %(tac -tic ) )
    
    save_to_hd5( list_nominals, list_cube, fBat, mal )
    
    
    
    
    
    
    
    
    
        
        
        
        
            
        
        
    
