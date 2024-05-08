#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:02:24 2024

@author: zanmar
"""

from util import fit_zernike_coefficients, rayPSF
import batoid
from batoid_rubin import builder
import numpy as np

class FromBatoid:
    
    def __init__( self, wl_index = 0, jmax = 22, debug=False ):
        # default telescope, etc
        self.jmax = jmax                  # zernike pupil terms
        self.field_radius = 1.708       # degrees
        if( debug ):
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
        self.psf_type = 'ray'       #can be 'ray' or 'fft'
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
            # print( 'new dof, update builder/telescope' )
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
        """
        get zernike coefficients for all coordinates.

        Returns
        -------
        znk_arr : ncoords x nznkpupil array
            We remove the first dummy coefficient. The result is not galsim 
               compatible.
        """
        
        znk_arr = []        # collect   nznk x nfield
        for x, y in zip( self.hx, self.hy ):
            znk = batoid.analysis.zernike( self._telescope, 
                    np.deg2rad(x), np.deg2rad( y ), 
                    self.wl, reference='mean', jmax=self.jmax, eps=0.61 )
            znk_arr.append( znk[1:] )              #we remove dummy coefficient

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
                    self.wl, reference='mean', jmax=self.jmax, eps=0.61 )
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
                    nx=255, reference='mean'
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
    
    def psf_compute( self, psf_type = 'ray', seeing=0.5 ):
        if( psf_type == 'ray' or psf_type == 'fft' ):
            self.psf_type = psf_type
        else:
            print('*** invalid psf_type. Using current valid:', self.psf_type )
            
        psf_list = []
        if( self.psf_type == 'ray' ):
            for x, y in zip( self.hx, self.hy ):
                psf = rayPSF( self._telescope,
                              np.deg2rad(x), np.deg2rad(y),
                              self.wl, nphot=10000,
                              seeing=seeing )
                psf_list.append( psf )
        elif( self.psf_type == 'fft' ):
            for x, y in zip( self.hx, self.hy ):
                psf = batoid.analysis.fftPSF(self._telescope,
                                             np.deg2rad(x), np.deg2rad(y),
                                             self.wl, nx=360,
                                       pad_factor=4, reference='mean') 
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
            
            
    def get_list_nominal_znk_coeffs( self ):
        
        assert np.allclose( self.dof, np.zeros(50) ), "you want nominal but current dof is not zero"
        res = []
        for i in range( self.n_wl ):
            self.wl_index = i
            res.append( (i, self.zernike_coeffs() ) )
            
        return res


    def get_list_cube_znk_coeffs( self, dof_list ):
        
        list_cube = []
        for i in range( self.n_wl ):
            self.wl_index = i
            cube = []                       # collect ndist x nznk x nfield 
            cnt = 1
            for dof in dof_list:
                print( dof[0:10], cnt,'/', len(dof_list), i+1,'/', self.n_wl )
                self.dof = dof
                cube.append( self.zernike_coeffs() )
                cnt+=1
            
            cube = np.asarray( cube )
            if( cube.shape[1] == self.jmax ):
                cube = np.transpose( cube, [2, 1, 0 ] )
            elif( cube.shape[1] == self.hx.size ):
                cube = np.transpose( cube, [1, 2, 0])
            else:
                print('*** expect trouble ahead')
                
            list_cube.append( (i, cube ) )
        return list_cube

class MisAlignment():
    def __init__( self ):
        self._max_amplitude = np.array([25.0, 1000.0, 1000.0, 25.0, 25.0, 25.0, 4000.0, 4000.0, 25.0, 25.0])
        self.n_dof = len( self._max_amplitude )
        self.nperturbations = 8     # even number
        self.list = self.get_dof_list()
        self.zeta = self.get_zeta_arr()
            
    def get_dof_list( self ):
        assert ( self.nperturbations % 2 ) == 0, 'expected even number of perturbations'
        lista = []
        for i in range( self.n_dof ):
            zeta_i = np.linspace( 0, self._max_amplitude[i], num = self.nperturbations//2 + 1)[1:]
            zeta_j = np.linspace( -self._max_amplitude[i], 0, num = self.nperturbations//2, endpoint=False )
            zeta_k = np.concatenate( [ zeta_j, zeta_i ] )
            
            for j in zeta_k:
                dof = np.zeros( 50 )
                dof[ i ] = j
                lista.append( dof )
        return lista
    
    def get_zeta_arr( self ):
        """Build an array with nperturbation linear distorsions for each 
        distorsion on the max_amplitude array.

        Parameters
        ----------
        max_amplitude : float array 
            n distorsion elements, e.g. dx, dy, dz, Rx, Ry for M2 and Camera
        nperturbations: int
            number of linear distorsions. Example, if amp for dx is 1000 and
        nrow is 4, we return for that distorsion [250, 500, 750, 1000].
        
        Returns
        -------
        zeta : array
            nrow x len( amp )  array
    
        
        """
        assert ( self.nperturbations % 2 ) == 0, 'expected even number of perturbations'
        zeta = np.zeros( (self.nperturbations, self.n_dof ))
    
        for i in range( self.n_dof ):
            zeta_i = np.linspace( 0, self._max_amplitude[i], num = self.nperturbations//2 + 1)[1:]
            zeta_j = np.linspace( -self._max_amplitude[i], 0, num = self.nperturbations//2, endpoint=False )
            zeta_k = np.concatenate( [zeta_j, zeta_i ] )
            zeta[:,i] = zeta_k
    
        return zeta
