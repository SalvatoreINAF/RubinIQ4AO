#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:17:11 2024

@author: zanmar
"""

import numpy as np

from scipy.fft import ifftshift, ifft2
from scipy.stats import moment
from scipy import ndimage

import galsim
from util import read_h5_coeffs, wf_mesh, submatsum
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool

import mpi4py.MPI as MPI

class Engine(object):
    # an class to be iterated in Pool with inputs pars
    # this does only PSF
    def __init__(self, coeffs, N, Xpmin, Xpmax, pupil_mask, xx, yy ):
        self.coeffs = coeffs
        self.N = N
        self.Xpmin = Xpmin
        self.Xpmax = Xpmax
        self.pupil_mask = pupil_mask
        self.xx = xx
        self.yy = yy

    def __call__(self, cnt ):
        W = wf_mesh( self.coeffs[ cnt, :], self.N, self.Xpmin, self.Xpmax )
        W = np.fft.ifftshift( W )
        E = np.exp( -1j * 2 * np.pi * W ) * self.pupil_mask
        E = np.fft.ifftshift( E )
        PSF = np.abs( np.fft.ifftshift( np.fft.ifft2( E )))**2.
        mylat = my_lattice( PSF, self.xx, self.yy )
        return mylat

class Engine2(object):
    # an class to be iterated in Pool with inputs pars
    # this does both PSF and ellipticity
    def __init__(self, coeffs, N, Xpmin, Xpmax, pupil_mask, xx, yy ):
        self.coeffs = coeffs
        self.N = N
        self.Xpmin = Xpmin
        self.Xpmax = Xpmax
        self.pupil_mask = pupil_mask
        self.xx = xx
        self.yy = yy

    def __call__(self, cnt ):
        W = wf_mesh( self.coeffs[ cnt, :], self.N, self.Xpmin, self.Xpmax )
        W = np.fft.ifftshift( W )
        E = np.exp( -1j * 2 * np.pi * W ) * self.pupil_mask
        E = np.fft.ifftshift( E )
        PSF = np.abs( np.fft.ifftshift( np.fft.ifft2( E )))**2.
        psf = my_lattice( PSF, self.xx, self.yy )

        withcoords = True           # TODO: make it a parameter later

        if( not withcoords ):
            nx, ny = self.psf[0].shape

        if( withcoords ):
            psfnorm = psf.array / psf.array.max()
            X = psf.coords[:,:,1]
            Y = psf.coords[:,:,0]
        else:
            psfnorm = psf / psf.max()
            xs = np.linspace(0, nx-1, nx )        # pixel coordintes
            ys = np.linspace(0, ny-1, ny )
            X, Y = np.meshgrid(xs, ys)
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

        return ( sigma_l, sigma_s, ellipticity_adimensional, alpha, x_cen, y_cen )


class my_lattice:
    def __init__( self, array, coordsX, coordsY ):
        self.array = array[::-1,::-1]
        self.coords = np.stack((np.flip( coordsY, axis=0),
                               np.flip( coordsX, axis=1)), axis=2 )

class lsstModel:
    def __init__( self, n_fldznk = 22, 
                 batoid_cube_file =None, 
                 model_file=None ):
    
        self.wl_index = 0
        self.n_fldznk = n_fldznk
        self.batoid_cubes = None
        self._hx = None
        self._hy = None
        self.znk_basis = None
        self.nom_proj = None            #projected nominal coeffs
        self.nominal = None             #for current coordinates hx, hy
        self.n_wl = None
        self.field_radius = 1.708
        self.f_number = 1.233279
        self.a1 = None                  # polynomial fit a1*x + a2*x^2
        self.a2 = None
        self.wl_array =np.array([0.384,0.481,0.622,0.770,0.895,0.994])*1e-6
        
        self.dof = np.zeros(50)
    
        if( batoid_cube_file ): #fit input batoid coefficients
            self.batoid_cubes = read_h5_coeffs( batoid_cube_file )
            self.n_wl = len( self.batoid_cubes )
            self.field_radius = self.batoid_cubes[0].field_radius
            # when creating model we can't use set_hxhy yet because we need nom_proj
            self._hx = self.batoid_cubes[0].coords[0,:]
            self._hy = self.batoid_cubes[0].coords[1,:]
            
            self.znk_basis = galsim.zernike.zernikeBasis( self.n_fldznk, 
                                            self._hx/self.field_radius, #norm 1
                                            self._hy/self.field_radius)
            znk_inv = np.linalg.pinv( self.znk_basis )
            
            self.nom_proj = [ np.matmul( znk_inv.T, self.batoid_cubes[iwl].nominal ) for iwl in range(self.n_wl ) ]
            self.nominal = [ np.matmul( self.znk_basis.T, nom_proj ) for nom_proj in self.nom_proj ]
            
            self.c_dblznk = get_dbl_znk_coeff( self.batoid_cubes, znk_inv )#(znkpup,znkdbl,zita,wl)
            
            self.a1, self.a2 = fit_misaligned_coeffs( self.batoid_cubes, self.c_dblznk )
        elif( model_file ):     # read a1, a2, nom_proj 
            pass                #  sets n_wl, n_fldznk and znk_basis, hx, hy to default
            
    def zernike_coeffs( self, piston_tilt = None ):
        """
        
        Returns
        -------
        pupil zernike coefficients for each field coordinate hx, hy. It does it
            for current set wavelength only.
        The first 3 coefficients are zeroed.

        """

        modelC = self.a1[:,:,:,self.wl_index] * self.dof + self.a2[:,:,:,self.wl_index] * self.dof**2
        modelC = np.sum( modelC, 2 )
        
        coeffs = np.matmul( self.znk_basis.T, modelC.T ) + self.nominal[self.wl_index]
        # coeffs = np.matmul( self.znk_basis.T, modelC.T ) + self.batoid_cubes[ self.wl_index ].nominal
        # print('x',coeffs.shape)
        
        if( piston_tilt is None ):
            coeffs[:,0:3] = 0.0
        else:
            coeffs[:,0:3] = piston_tilt

        return coeffs
    
    def wavefront( self ):
        wf_list = []
        
        coeffs = self.zernike_coeffs()      # ncoords x nznk
        
        for i in range( coeffs.shape[0] ):
            wf = wf_mesh( coeffs[i], 255 )
            wf_list.append( wf )
            
        return wf_list
    
    def set_hxhy( self, hx, hy ):
        """
        field coordinates in degrees
        When we set the coordinates we also update znk basis and nominals for
            the new coordinates. 

        Parameters
        ----------
        hx : array
            field x coordinates in degrees.
        hy : TYPE
            field y coordinates in degrees.

        Returns
        -------
        None.

        """
        self._hx, self._hy = hx, hy
        self.znk_basis = galsim.zernike.zernikeBasis( self.n_fldznk, 
                                        self._hx/self.field_radius, #norm 1
                                        self._hy/self.field_radius)
        self.nominal = [ np.matmul( self.znk_basis.T, nom_proj ) for nom_proj in self.nom_proj ]
        
        return

    def cube_from_model_fit( self ):
        # we apply the model in the same grid as the fit to compare input cube to
        #  fitted model
        result = []
        for iwl in range( self.n_wl ):
            self.wl_index = iwl
            icnt = 0
            model_cube = np.zeros([ self.batoid_cubes[iwl].nfield, self.batoid_cubes[iwl].nznkpupil, self.batoid_cubes[iwl].zeta.size ])
            for k in range( self.batoid_cubes[iwl].ndof ):              # dof
                # idx = np.arange( k * d[iwl].npert, 
                #                  k * d[iwl].npert + d[iwl].npert )

                x = self.batoid_cubes[iwl].zeta[:,k]            

                for i_dist in range( self.batoid_cubes[iwl].npert ):
                    zz = np.zeros( ( [ self.batoid_cubes[iwl].ndof ] ) )
                    zz[ k ] = x[ i_dist ]
                    
                    self.dof = zz
                    coeffs = self.zernike_coeffs()
                    
#                    modelC = a1sol[:,:,:,iwl] * zz + a2sol[:,:,:,iwl] * zz**2
#                    modelC = np.sum( modelC, 2 )
                    
#                    coeffs = np.matmul( znk_basis.T, modelC.T ) + d[iwl].nominal
                    
                    #compare with original
                    # mymean = np.mean( d[iwl].cube[:,3:,icnt] - coeffs[:,3:] )
                    # mystd = np.std( d[iwl].cube[:,3:,icnt] - coeffs[:,3:] )
                    
                    model_cube[:,:,icnt] = coeffs
                    
                    # print( "%.3d %.5e %.5e" %(icnt, mymean, mystd ) )
                    icnt += 1
                    
            result.append( model_cube )
        
        
        
        return result
    
    
    def psf_compute( self, with_binning = False, seeing=0.0, piston_tilt=None ):
        # compute a psf with FFT for each coordinate
        pad_factor = 4
        nx = 360
        nbin = 23   # TODO: remove, study scale instead
        
        psf_list = []
        
        coeffs = self.zernike_coeffs( piston_tilt )


        cnt = 0
        for x, y in zip( self._hx, self._hy ):
            
            wfarr = wf_mesh( coeffs[ cnt, :], n=nx  )
                        
            # wfarr = wf.array
            pad_size = nx*pad_factor
            expwf = np.zeros((pad_size, pad_size), dtype=np.complex128)
            start = pad_size//2-nx//2
            stop = pad_size//2+nx//2
            expwf[start:stop, start:stop][~wfarr.mask] = \
                np.exp(2j*np.pi*wfarr[~wfarr.mask])
            psf = np.abs(np.fft.fftshift(np.fft.fft2(expwf)))**2
            

            sigma = 10 * seeing / 2.35 / 0.2 * psf.shape[0] / 20
            # print( 'sigma', sigma )
            if( seeing > 0.0 ):
                print('.', end='',flush=True )
                #sigma = 10 * seeing / 2.35 / 0.2 * psf.shape[0] / 20
                psf = gaussian_filter(psf, sigma= sigma )


            if( with_binning ):
                psf = submatsum( psf, nbin, nbin )


            psf_list.append( psf )
            cnt += 1

        self.psf = psf_list
        
        return
    
    def psf_compute_lattice( self, N=256, L = 8 ):
        """A slightly different approach for computing the PSF, this time with
        physical units. As in batoid, we output a lattice.
        
        L is the size in pixels where each pixel is 10um

        Parameters
        ----------
        with_binning : TYPE, optional
            DESCRIPTION. The default is False.
        seeing : TYPE, optional
            DESCRIPTION. The default is 0.0.
        piston_tilt : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        wl = self.wl_array[ self.wl_index ]
        eta = 0.61
        xmax_in = L/2 * 10e-6   # xmax_in is half the size of ROI in focal plane in meters
        dx_in = 2*xmax_in / N
        
        # oversampling grid
        oversampling = 1
        xmax = oversampling * xmax_in
        dx = dx_in / oversampling
        
        x = np.linspace(-xmax,xmax-dx,num= N * oversampling ) #test oversampling not 1
        xx,yy = np.meshgrid(x,x)        # these are the coordinates for the lattice
        
        # pupil plane sampling
        fu = np.linspace(-1/(2*dx), 1/ (2*dx) - ( 1/ (2*xmax) ), endpoint=True, num=N )
        
        #print( 'dx=', dx )
        #print( 'fu:', fu[0], fu[-1], fu.shape )
        Xpmin = -2 * wl * self.f_number * fu[0]
        Xpmax = -2 * wl * self.f_number * fu[-1]
        
        #print( 'wl', wl)
        
        fu = np.fft.fftshift( fu )
        uu, vv = np.meshgrid( fu, fu )
            
        # Normalized pupil coordinates
        Xp = -2 * wl * self.f_number * uu
        Yp = -2 * wl * self.f_number * vv
        
        #print( 'xp min/max', Xp.min(), Xp.max() )
        
        #print( 'x:', -xmax, xmax )
        #print('Xp:', Xpmin, Xpmax )
        
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(Xp, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Xp')
        # fig.colorbar(im, ax=ax)
        
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(Yp, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Yp')
        # fig.colorbar(im, ax=ax)
        
        #Pupil mask
        pupil_mask = uu * 0.0
        pupil_mask[ ( np.sqrt( Xp**2. + Yp**2. ) <= 1 ) ] = 1.
        pupil_mask[( np.sqrt( Xp**2. + Yp**2. ) <= eta )] = 0. #central obscuration
        
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh( pupil_mask, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('pupil mask')
        # fig.colorbar(im, ax=ax)
        
#        xpp = np.fft.ifftshift( Xp[0,:] )
#        ypp = np.fft.ifftshift( Yp[:,0] )
        
        #print( 'criteria: %.1e < %.1e' %( 1/(wl*self.f_number), 1/(2*dx) ) )
        
        if( np.max(np.abs((Xp))) < 1):
            print("Insufficient Pupil plane sampling. Increase number of points")
            print(" of reduce maximum Image plane covering (xmax)")
            sys.exit(1)
            
        Mp, Np = Xp.shape
        #print("Pupil plane grid size: %d x %d" %(Mp, Np))
        
        coeffs = self.zernike_coeffs( )     #for all current field points
                
        psf_list = []


        # try:
        #     pool = Pool() # on 8 processors
        #     eng = Engine(coeffs,N,Xpmin,Xpmax,pupil_mask,xx,yy)
        #     psf_list = pool.map(eng, range(len(self._hx)))
        # finally: # To make sure processes are closed in the end, even if errors happen
        #     pool.close()
        #     pool.join()

        cnt = 0
        for x, y in zip( self._hx, self._hy ):
        
            W = wf_mesh( coeffs[ cnt, :], N, Xpmin, Xpmax )
            
            # fig, ax = plt.subplots()
            # im = ax.pcolormesh( W*wl, cmap='jet')
            # ax.set_aspect('equal', 'box')
            # ax.set_title('wf')
            # fig.colorbar(im, ax=ax)
            
            #->W = np.fft.ifftshift( W )
            W = ifftshift( W )
            
            
            E = np.exp( -1j * 2 * np.pi * W ) * pupil_mask
            
            #->E = np.fft.ifftshift( E )
            E = ifftshift( E )
            
            #->PSF = np.abs( np.fft.ifftshift( np.fft.ifft2( E )))**2.
            PSF = np.abs( ifftshift( ifft2( E )))**2.
            
            # fig, ax = plt.subplots()
            # im = ax.pcolormesh( np.real(E), cmap='jet')
            # ax.set_aspect('equal', 'box')
            # ax.set_title('E')
            # fig.colorbar(im, ax=ax)
            
            # fig, ax = plt.subplots()
            # im = ax.pcolormesh( np.imag(E), cmap='jet')
            # ax.set_aspect('equal', 'box')
            # ax.set_title('E')
            # fig.colorbar(im, ax=ax)
            
            # fig, ax = plt.subplots()
            # im = ax.pcolormesh( xx, yy, PSF, cmap='jet')
            # ax.set_aspect('equal', 'box')
            # ax.set_title('PSF')
            # ax.set_xlim((5.475e-5,-5.475e-5 ))         # set ROI to -1,1 for both x,y
            # ax.set_ylim((5.475e-5,-5.475e-5 ))
            # fig.colorbar(im, ax=ax)        
            
            mylat = my_lattice( PSF, xx, yy ) 
        
            psf_list.append( mylat )
            cnt += 1

        self.psf = psf_list

    def psf_compute_lattice_multiproc( self, N=256, L = 8 ):
        """A slightly different approach for computing the PSF, this time with
        physical units. As in batoid, we output a lattice.

        L is the size in pixels where each pixel is 10um

        Parameters
        ----------
        with_binning : TYPE, optional
            DESCRIPTION. The default is False.
        seeing : TYPE, optional
            DESCRIPTION. The default is 0.0.
        piston_tilt : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        wl = self.wl_array[ self.wl_index ]
        eta = 0.61
        xmax_in = L/2 * 10e-6   # xmax_in is half the size of ROI in focal plane in meters
        dx_in = 2*xmax_in / N

        # oversampling grid
        oversampling = 1
        xmax = oversampling * xmax_in
        dx = dx_in / oversampling

        x = np.linspace(-xmax,xmax-dx,num= N * oversampling ) #test oversampling not 1
        xx,yy = np.meshgrid(x,x)        # these are the coordinates for the lattice

        # pupil plane sampling
        fu = np.linspace(-1/(2*dx), 1/ (2*dx) - ( 1/ (2*xmax) ), endpoint=True, num=N )

        #print( 'dx=', dx )
        #print( 'fu:', fu[0], fu[-1], fu.shape )
        Xpmin = -2 * wl * self.f_number * fu[0]
        Xpmax = -2 * wl * self.f_number * fu[-1]

        #print( 'wl', wl)

        fu = np.fft.fftshift( fu )
        uu, vv = np.meshgrid( fu, fu )

        # Normalized pupil coordinates
        Xp = -2 * wl * self.f_number * uu
        Yp = -2 * wl * self.f_number * vv

        #print( 'xp min/max', Xp.min(), Xp.max() )

        #print( 'x:', -xmax, xmax )
        #print('Xp:', Xpmin, Xpmax )

        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(Xp, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Xp')
        # fig.colorbar(im, ax=ax)

        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(Yp, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Yp')
        # fig.colorbar(im, ax=ax)

        #Pupil mask
        pupil_mask = uu * 0.0
        pupil_mask[ ( np.sqrt( Xp**2. + Yp**2. ) <= 1 ) ] = 1.
        pupil_mask[( np.sqrt( Xp**2. + Yp**2. ) <= eta )] = 0. #central obscuration

        # fig, ax = plt.subplots()
        # im = ax.pcolormesh( pupil_mask, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('pupil mask')
        # fig.colorbar(im, ax=ax)

#        xpp = np.fft.ifftshift( Xp[0,:] )
#        ypp = np.fft.ifftshift( Yp[:,0] )

        #print( 'criteria: %.1e < %.1e' %( 1/(wl*self.f_number), 1/(2*dx) ) )

        if( np.max(np.abs((Xp))) < 1):
            print("Insufficient Pupil plane sampling. Increase number of points")
            print(" of reduce maximum Image plane covering (xmax)")
            sys.exit(1)

        Mp, Np = Xp.shape
        #print("Pupil plane grid size: %d x %d" %(Mp, Np))

        coeffs = self.zernike_coeffs( )     #for all current field points

        psf_list = []


        try:
            pool = Pool() # on 8 processors
            eng = Engine(coeffs,N,Xpmin,Xpmax,pupil_mask,xx,yy)
            psf_list = pool.map(eng, range(len(self._hx)))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        self.psf = psf_list

    def psf_ellip_compute_multiproc( self, N=256, L = 8, withcoords = False, nprocess=6 ):
        """A slightly different approach for computing the PSF, this time with
        physical units. As in batoid, we output a lattice.

        L is the size in pixels where each pixel is 10um

        Parameters
        ----------
        with_binning : TYPE, optional
            DESCRIPTION. The default is False.
        seeing : TYPE, optional
            DESCRIPTION. The default is 0.0.
        piston_tilt : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        wl = self.wl_array[ self.wl_index ]
        eta = 0.61
        xmax_in = L/2 * 10e-6   # xmax_in is half the size of ROI in focal plane in meters
        dx_in = 2*xmax_in / N

        # oversampling grid
        oversampling = 1
        xmax = oversampling * xmax_in
        dx = dx_in / oversampling

        x = np.linspace(-xmax,xmax-dx,num= N * oversampling ) #test oversampling not 1
        xx,yy = np.meshgrid(x,x)        # these are the coordinates for the lattice

        # pupil plane sampling
        fu = np.linspace(-1/(2*dx), 1/ (2*dx) - ( 1/ (2*xmax) ), endpoint=True, num=N )

        #print( 'dx=', dx )
        #print( 'fu:', fu[0], fu[-1], fu.shape )
        Xpmin = -2 * wl * self.f_number * fu[0]
        Xpmax = -2 * wl * self.f_number * fu[-1]

        #print( 'wl', wl)

        fu = np.fft.fftshift( fu )
        uu, vv = np.meshgrid( fu, fu )

        # Normalized pupil coordinates
        Xp = -2 * wl * self.f_number * uu
        Yp = -2 * wl * self.f_number * vv

        #print( 'xp min/max', Xp.min(), Xp.max() )

        #print( 'x:', -xmax, xmax )
        #print('Xp:', Xpmin, Xpmax )

        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(Xp, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Xp')
        # fig.colorbar(im, ax=ax)

        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(Yp, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Yp')
        # fig.colorbar(im, ax=ax)

        #Pupil mask
        pupil_mask = uu * 0.0
        pupil_mask[ ( np.sqrt( Xp**2. + Yp**2. ) <= 1 ) ] = 1.
        pupil_mask[( np.sqrt( Xp**2. + Yp**2. ) <= eta )] = 0. #central obscuration

        # fig, ax = plt.subplots()
        # im = ax.pcolormesh( pupil_mask, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('pupil mask')
        # fig.colorbar(im, ax=ax)

#        xpp = np.fft.ifftshift( Xp[0,:] )
#        ypp = np.fft.ifftshift( Yp[:,0] )

        #print( 'criteria: %.1e < %.1e' %( 1/(wl*self.f_number), 1/(2*dx) ) )

        if( np.max(np.abs((Xp))) < 1):
            print("Insufficient Pupil plane sampling. Increase number of points")
            print(" of reduce maximum Image plane covering (xmax)")
            sys.exit(1)

        Mp, Np = Xp.shape
        #print("Pupil plane grid size: %d x %d" %(Mp, Np))

        coeffs = self.zernike_coeffs( )     #for all current field points

        try:
            pool = Pool(processes=nprocess) # on 8 processors
            eng = Engine2(coeffs,N,Xpmin,Xpmax,pupil_mask,xx,yy)
            ellip_list = pool.map(eng, range(len(self._hx)))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        sl = [ tup[0] for tup in ellip_list ]
        ss = [ tup[1] for tup in ellip_list ]
        el = [ tup[2] for tup in ellip_list ]
        pa = [ tup[3] for tup in ellip_list ]
        xc = [ tup[4] for tup in ellip_list ]
        yc = [ tup[5] for tup in ellip_list ]

        self.ellipticity = {'sl': sl, 'ss': ss, 'el': el, 
                         'pa': pa, 'xc': xc, 'yc': yc}

    def psf_ellip_compute_mpi( self, N=256, L = 8, withcoords=False ):
        """Same as lattice version but we split loops for MPI parallelization
        
        L is the size in pixels where each pixel is 10um

        Parameters
        ----------
        with_binning : TYPE, optional
            DESCRIPTION. The default is False.
        seeing : TYPE, optional
            DESCRIPTION. The default is 0.0.
        piston_tilt : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        wl = self.wl_array[ self.wl_index ]
        eta = 0.61
        xmax_in = L/2 * 10e-6   # xmax_in is half the size of ROI in focal plane in meters
        dx_in = 2*xmax_in / N
        
        # oversampling grid
        oversampling = 1
        xmax = oversampling * xmax_in
        dx = dx_in / oversampling
        
        x = np.linspace(-xmax,xmax-dx,num= N * oversampling ) #test oversampling not 1
        xx,yy = np.meshgrid(x,x)        # these are the coordinates for the lattice
        
        # pupil plane sampling
        fu = np.linspace(-1/(2*dx), 1/ (2*dx) - ( 1/ (2*xmax) ), endpoint=True, num=N )
        
        #print( 'dx=', dx )
        #print( 'fu:', fu[0], fu[-1], fu.shape )
        Xpmin = -2 * wl * self.f_number * fu[0]
        Xpmax = -2 * wl * self.f_number * fu[-1]
        
        #print( 'wl', wl)
        
        fu = np.fft.fftshift( fu )
        uu, vv = np.meshgrid( fu, fu )
            
        # Normalized pupil coordinates
        Xp = -2 * wl * self.f_number * uu
        Yp = -2 * wl * self.f_number * vv
        
        #print( 'xp min/max', Xp.min(), Xp.max() )
        
        #print( 'x:', -xmax, xmax )
        #print('Xp:', Xpmin, Xpmax )
        
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(Xp, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Xp')
        # fig.colorbar(im, ax=ax)
        
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(Yp, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Yp')
        # fig.colorbar(im, ax=ax)
        
        #Pupil mask
        pupil_mask = uu * 0.0
        pupil_mask[ ( np.sqrt( Xp**2. + Yp**2. ) <= 1 ) ] = 1.
        pupil_mask[( np.sqrt( Xp**2. + Yp**2. ) <= eta )] = 0. #central obscuration
        
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh( pupil_mask, cmap='jet')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('pupil mask')
        # fig.colorbar(im, ax=ax)
        
#        xpp = np.fft.ifftshift( Xp[0,:] )
#        ypp = np.fft.ifftshift( Yp[:,0] )
        
        #print( 'criteria: %.1e < %.1e' %( 1/(wl*self.f_number), 1/(2*dx) ) )
        
        if( np.max(np.abs((Xp))) < 1):
            print("Insufficient Pupil plane sampling. Increase number of points")
            print(" of reduce maximum Image plane covering (xmax)")
            sys.exit(1)
            
        Mp, Np = Xp.shape
        #print("Pupil plane grid size: %d x %d" %(Mp, Np))
        
        coeffs = self.zernike_coeffs( )     #for all current field points
                
        psf_list = []


######  MPI  ######

        world_comm = MPI.COMM_WORLD
        world_size = world_comm.Get_size()
        my_rank = world_comm.Get_rank()

        N_field = len( self._hx )   #number of field points
        

        # determine the workload of each rank
        workloads = [ N_field // world_size for i in range(world_size) ]
        for i in range( N_field % world_size ):
            workloads[i] += 1
        my_start = 0
        for i in range( my_rank ):
            my_start += workloads[i]
        my_end = my_start + workloads[my_rank]

        
        ######  MAIN loop to be parallelized  ######
        sl = []
        ss = []
        el = []
        pa = []
        xc = []
        yc = []
        
        for cnt in range( my_start, my_end ):   #main loop to be split by MPI
            x = self._hx
            y = self._hy
        # for x, y in zip( self._hx, self._hy ):

            W = wf_mesh( coeffs[ cnt, :], N, Xpmin, Xpmax )

            # fig, ax = plt.subplots()
            # im = ax.pcolormesh( W*wl, cmap='jet')
            # ax.set_aspect('equal', 'box')
            # ax.set_title('wf')
            # fig.colorbar(im, ax=ax)

            #->W = np.fft.ifftshift( W )
            W = ifftshift( W )


            E = np.exp( -1j * 2 * np.pi * W ) * pupil_mask

            #->E = np.fft.ifftshift( E )
            E = ifftshift( E )

            #->PSF = np.abs( np.fft.ifftshift( np.fft.ifft2( E )))**2.
            PSF = np.abs( ifftshift( ifft2( E )))**2.

            # fig, ax = plt.subplots()
            # im = ax.pcolormesh( np.real(E), cmap='jet')
            # ax.set_aspect('equal', 'box')
            # ax.set_title('E')
            # fig.colorbar(im, ax=ax)

            # fig, ax = plt.subplots()
            # im = ax.pcolormesh( np.imag(E), cmap='jet')
            # ax.set_aspect('equal', 'box')
            # ax.set_title('E')
            # fig.colorbar(im, ax=ax)
            
            # fig, ax = plt.subplots()
            # im = ax.pcolormesh( xx, yy, PSF, cmap='jet')
            # ax.set_aspect('equal', 'box')
            # ax.set_title('PSF')
            # ax.set_xlim((5.475e-5,-5.475e-5 ))         # set ROI to -1,1 for both x,y
            # ax.set_ylim((5.475e-5,-5.475e-5 ))
            # fig.colorbar(im, ax=ax)        
            
            psf = my_lattice( PSF, xx, yy )
            
            
            ### and now we compute the ellipticity for this psf
        
            # psf_list.append( psf )

            if( not withcoords ):
                nx, ny = self.psf[0].shape

            if( withcoords ):
                psfnorm = psf.array / psf.array.max()
                X = psf.coords[:,:,1]
                Y = psf.coords[:,:,0]
            else:
                psfnorm = psf / psf.max()
                xs = np.linspace(0, nx-1, nx )        # pixel coordintes
                ys = np.linspace(0, ny-1, ny )
                X, Y = np.meshgrid(xs, ys)
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
            #print("%.3d %5.2f %10.2f" %( cnt, ellipticity_adimensional, alpha ) )
            
            
            sl.append( sigma_l)
            ss.append( sigma_s )
            el.append( ellipticity_adimensional )
            pa.append( alpha )          # degrees
            xc.append( x_cen )
            yc.append( y_cen )
                
        ### END LOOP, collect results
        
        sl_all = np.array( [] )   #initialize outside
        ss_all = np.array( [] )
        el_all = np.array( [] )
        pa_all = np.array( [] )
        xc_all = np.array( [] )
        yc_all = np.array( [] )
        
        if my_rank == 0:
            sl_all = np.concatenate( (sl_all, sl ))
            ss_all = np.concatenate( (ss_all, ss ))
            el_all = np.concatenate( (el_all, el ))
            pa_all = np.concatenate( (pa_all, pa ))
            xc_all = np.concatenate( (xc_all, xc ))
            yc_all = np.concatenate( (yc_all, yc ))
            
            # print('inside rank0, pa')
            # print( pa_all )
            
            for i in range( 1, world_size ):
                sl_np = np.empty( workloads[i] )
                ss_np = np.empty( workloads[i] )
                el_np = np.empty( workloads[i] )
                pa_np = np.empty( workloads[i] )
                xc_np = np.empty( workloads[i] )
                yc_np = np.empty( workloads[i] )
                
                world_comm.Recv( [sl_np, MPI.DOUBLE], source=i, tag=37 )
                world_comm.Recv( [ss_np, MPI.DOUBLE], source=i, tag=47 )
                world_comm.Recv( [el_np, MPI.DOUBLE], source=i, tag=57 )
                world_comm.Recv( [pa_np, MPI.DOUBLE], source=i, tag=67 )
                world_comm.Recv( [xc_np, MPI.DOUBLE], source=i, tag=77 )
                world_comm.Recv( [yc_np, MPI.DOUBLE], source=i, tag=87 )
                
                sl_all = np.concatenate( (sl_all, sl_np ) )
                ss_all = np.concatenate( (ss_all, ss_np ) )
                el_all = np.concatenate( (el_all, el_np ) )
                pa_all = np.concatenate( (pa_all, pa_np ) )
                xc_all = np.concatenate( (xc_all, xc_np ) )
                yc_all = np.concatenate( (yc_all, yc_np ) )
                
            # print( 'after concat all cpus:' )
            # print( pa_all )

        else:
            sl_np = np.array( sl )
            ss_np = np.array( ss )
            el_np = np.array( el )
            pa_np = np.array( pa )
            xc_np = np.array( xc )
            yc_np = np.array( yc )

            world_comm.Send( [sl_np, MPI.DOUBLE], dest=0, tag=37 )
            world_comm.Send( [ss_np, MPI.DOUBLE], dest=0, tag=47 )
            world_comm.Send( [el_np, MPI.DOUBLE], dest=0, tag=57 )
            world_comm.Send( [pa_np, MPI.DOUBLE], dest=0, tag=67 )
            world_comm.Send( [xc_np, MPI.DOUBLE], dest=0, tag=77 )
            world_comm.Send( [yc_np, MPI.DOUBLE], dest=0, tag=87 )


        # print( 'outside:', my_rank )
        # print( pa_all )
        
        self.ellipticity = {'sl': list( sl_all ), 'ss': list( ss_all ), 
                    'el': list( el_all ), 'pa': list( pa_all ), 
                    'xc': list( xc_all ), 'yc': list( yc_all )}
#        self.psf = psf_list

    def psf_compute_noloop( self, N=256, L = 8 ):
        """A slightly different approach for computing the PSF, this time with
        physical units. As in batoid, we output a lattice.

        L is the size in pixels where each pixel is 10um

        Parameters
        ----------
        with_binning : TYPE, optional
            DESCRIPTION. The default is False.
        seeing : TYPE, optional
            DESCRIPTION. The default is 0.0.
        piston_tilt : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        wl = self.wl_array[ self.wl_index ]
        eta = 0.61
        xmax_in = L/2 * 10e-6   # xmax_in is half the size of ROI in focal plane in meters
        dx_in = 2*xmax_in / N

        # oversampling grid
        oversampling = 1
        xmax = oversampling * xmax_in
        dx = dx_in / oversampling

        x = np.linspace(-xmax,xmax-dx,num= N * oversampling ) #test oversampling not 1
        xx,yy = np.meshgrid(x,x)        # these are the coordinates for the lattice

        # pupil plane sampling
        fu = np.linspace(-1/(2*dx), 1/ (2*dx) - ( 1/ (2*xmax) ), endpoint=True, num=N )

        #print( 'dx=', dx )
        #print( 'fu:', fu[0], fu[-1], fu.shape )
        Xpmin = -2 * wl * self.f_number * fu[0]
        Xpmax = -2 * wl * self.f_number * fu[-1]

        #print( 'wl', wl)

        fu = np.fft.fftshift( fu )
        uu, vv = np.meshgrid( fu, fu )

        # Normalized pupil coordinates
        Xp = -2 * wl * self.f_number * uu
        Yp = -2 * wl * self.f_number * vv


        #Pupil mask
        pupil_mask = uu * 0.0
        pupil_mask[ ( np.sqrt( Xp**2. + Yp**2. ) <= 1 ) ] = 1.
        pupil_mask[( np.sqrt( Xp**2. + Yp**2. ) <= eta )] = 0. #central obscuration


        if( np.max(np.abs((Xp))) < 1):
            print("Insufficient Pupil plane sampling. Increase number of points")
            print(" of reduce maximum Image plane covering (xmax)")
            sys.exit(1)

        Mp, Np = Xp.shape

        coeffs = self.zernike_coeffs( )     #for all current field points

        psf_list = []

        # print( 'coeffs shape:', coeffs.shape )


####

        # test, prepare a cube to do calculations at once with numpy
        Wcube = np.zeros( (len( self._hx ), N, N ) )

        for cnt in range( len( self._hx ) ):
            Wcube[cnt,:,:] = wf_mesh( coeffs[ cnt, :], N, Xpmin, Xpmax )

# step by step OR
#        Wcube = np.fft.ifftshift( Wcube, axes=(1,2) )
#        Ecube = np.fft.ifftshift( np.exp( -1j * 2 * np.pi * Wcube ) * pupil_mask[None,...], axes=(1,2))
#        PSFcube = np.abs( np.fft.ifftshift( np.fft.ifft2( Ecube ),axes=(1,2)))**2.

# all in one go
# v1
        # Ecube = np.fft.ifftshift( np.exp( -1j * 2 * np.pi * np.fft.ifftshift( Wcube, axes=(1,2) ) ) * pupil_mask[None,...], axes=(1,2))
        # PSFcube = np.abs( np.fft.ifftshift( np.fft.ifft2( Ecube ),axes=(1,2)))**2.

    # with scipy
        Ecube = ifftshift( np.exp( -1j * 2 * np.pi * ifftshift( Wcube, axes=(1,2) ) ) * pupil_mask[None,...], axes=(1,2))
        PSFcube = np.abs( ifftshift( ifft2( Ecube ),axes=(1,2)))**2.

# v2 ** no advantage seen and is too ugly

        # PSFcube = np.abs( np.fft.ifftshift( np.fft.ifft2( np.fft.ifftshift( np.exp( -1j * 2 * np.pi * np.fft.ifftshift( Wcube, axes=(1,2) ) ) * pupil_mask[None,...], axes=(1,2)) ),axes=(1,2)))**2.
####
        psf_list2 = []
        for cnt in range( len( self._hx ) ):
            psf_list2.append( my_lattice( PSFcube[cnt,:,:], xx, yy ) )


        # for cnt in range( len( self._hx ) ):
        # # for x, y in zip( self._hx, self._hy ):

        #     W = wf_mesh( coeffs[ cnt, :], N, Xpmin, Xpmax )
        #     W = np.fft.ifftshift( W )

        #     if( np.allclose(W, Wcube[cnt,:,:])):
        #         print('ok ishift')



        #     E = np.exp( -1j * 2 * np.pi * W ) * pupil_mask
        #     E = np.fft.ifftshift( E )
        #     if( np.allclose(E, Ecube[cnt,:,:])):
        #         print('ok E with ishift')
        #     else:
        #         print(' bad E')


        #     PSF = np.abs( np.fft.ifftshift( np.fft.ifft2( E )))**2.

        #     if( np.allclose(PSF, PSFcube[cnt,:,:])):
        #         print('OK PSF')
        #     else:
        #         print(' bad PSF')



        #     mylat = my_lattice( PSF, xx, yy )

        #     psf_list.append( mylat )
        #     # cnt += 1

        self.psf = psf_list2
    
    def ellipticity_compute( self, withcoords=False ):
        if( not withcoords ):
            nx, ny = self.psf[0].shape
        sl = []
        ss = []
        el = []
        pa = []
        xc = []
        yc = []
        mu_xx_list = []
        mu_yy_list = []
        mu_xy_list = []
        p_list = []
        q_list = []
        e1_list = []
        e2_list = []
        cnt = 0
        for psf in self.psf:
            if( withcoords ):
                psfnorm = psf.array / psf.array.max()
                X = psf.coords[:,:,1]
                Y = psf.coords[:,:,0]
            else:
                psfnorm = psf / psf.max()
                xs = np.linspace(0, nx-1, nx )        # FFT grid pixel coordinates
                ys = np.linspace(0, ny-1, ny )        # NOT lsst pixels
                X, Y = np.meshgrid(xs, ys)
            suma = psfnorm.sum()
        #centroid
            x_cen = ( psfnorm * X ).sum() / suma
            y_cen = ( psfnorm * Y ).sum() / suma
        #moments 2nd
            mu_xx = ((X - x_cen)**2 * psfnorm ).sum() / suma
            mu_yy = ((Y - y_cen)**2 * psfnorm ).sum() / suma
            mu_xy = ((X - x_cen) * ( Y - y_cen ) * psfnorm ).sum() / suma
        #ellipticity
            p = mu_xx + mu_yy                  # m^2
            e1 = ( mu_xx - mu_yy ) / p         #
            e2 = 2 * mu_xy / p                 #
            q = np.sqrt( e1**2 + e2**2 ) * p   # m^2
            sigma_l = np.sqrt( ( p + q ) / 2 ) # m
            sigma_s = np.sqrt( ( p - q ) / 2 ) # m
            ellipticity_adimensional = 1 - sigma_s/sigma_l
            alpha = np.rad2deg( np.arctan2( e2, e1 ) / 2 )   # deg
            
            # print( q * np.cos( alpha ), q * np.sin( alpha ), mu_xx, mu_yy, mu_xy )
            #print("%.3d %5.2f %10.2f" %( cnt, ellipticity_adimensional, alpha ) )
            pix_per_m = 1 / 10.0e-6    # 1pix = 10um
            cnt+=1
            
            sl.append( sigma_l * pix_per_m )
            ss.append( sigma_s * pix_per_m )
            el.append( ellipticity_adimensional )
            pa.append( alpha )          # degrees
            xc.append( x_cen * pix_per_m )
            yc.append( y_cen * pix_per_m )
            mu_xx_list.append( mu_xx * pix_per_m**2 )
            mu_yy_list.append( mu_yy * pix_per_m**2 )
            mu_xy_list.append( mu_xy * pix_per_m**2 )
            p_list.append( p * pix_per_m**2 )
            q_list.append( q * pix_per_m**2 )
            e1_list.append( e1 )
            e2_list.append( e2 )
            
        self.ellipticity = {'sl': sl, 'ss': ss, 'el': el, 
                         'pa': pa, 'xc': xc, 'yc': yc,
                         'muxx': mu_xx_list,
                         'muyy': mu_yy_list,
                         'muxy': mu_xy_list,
                         'p': p_list,
                         'q': q_list,
                         'e1': e1_list,
                         'e2': e2_list
                         }
            
        return
    
    def ellipticity_compute_scipy( self, withcoords=False ):
        if( not withcoords ):
            nx, ny = self.psf[0].shape
        sl = []
        ss = []
        el = []
        pa = []
        xc = []
        yc = []
        cnt = 0
        for psf in self.psf:
            if( withcoords ):
                psfnorm = psf.array / psf.array.max()
                X = psf.coords[:,:,1]
                Y = psf.coords[:,:,0]
            else:
                psfnorm = psf / psf.max()
                xs = np.linspace(0, nx-1, nx )        # pixel coordintes
                ys = np.linspace(0, ny-1, ny )
                X, Y = np.meshgrid(xs, ys)
            suma = psfnorm.sum()
        #centroid
            x_cen = ( psfnorm * X ).sum() / suma
            y_cen = ( psfnorm * Y ).sum() / suma
            
            
            nx, ny = self.psf[0].array.shape
            xs = np.linspace(0, nx-1, nx )        # pixel coordintes
            ys = np.linspace(0, ny-1, ny )
            XXX, YYY = np.meshgrid(xs, ys)
            
            x_cen2 = ( psfnorm * XXX ).sum() / suma
            y_cen2 = ( psfnorm * YYY ).sum() / suma
            
            print( 'xcen, ycen old:', x_cen2, y_cen2 )
            xxx = ndimage.center_of_mass( psfnorm )
            print('xxx', xxx )
            
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
            #print("%.3d %5.2f %10.2f" %( cnt, ellipticity_adimensional, alpha ) )
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
    
    def ellipticity_compute_algebra( self, withcoords=False ):
        ### calculate moments algebraically for each field point
        # e.g. field dependence is not included in the algebra.
        # for now return the same dictionary as the psf/ellip version
        # the output units should be in pixels

        #TODO: copy from analytic_moments mu_xx, mu_yy and mu_xy
        znk_coeffs = self.zernike_coeffs( )
        c4  = znk_coeffs[:, 3 ]
        c5  = znk_coeffs[:, 4 ] 
        c6  = znk_coeffs[:, 5 ]
        c7  = znk_coeffs[:, 6 ]
        c8  = znk_coeffs[:, 7 ]
        c9  = znk_coeffs[:, 8 ]
        c10 = znk_coeffs[:, 9 ]
        c11 = znk_coeffs[:, 10 ]

        if( znk_coeffs.shape[1] == 22 or znk_coeffs.shape[1] == 37 ):
            c12 = znk_coeffs[:, 11 ]
            c13 = znk_coeffs[:, 12 ]
            c14 = znk_coeffs[:, 13 ]
            c15 = znk_coeffs[:, 14 ]
            c16 = znk_coeffs[:, 15 ]
            c17 = znk_coeffs[:, 16 ]
            c18 = znk_coeffs[:, 17 ]
            c19 = znk_coeffs[:, 18 ]
            c20 = znk_coeffs[:, 19 ]
            c21 = znk_coeffs[:, 20 ]
            c22 = znk_coeffs[:, 21 ]

        if( znk_coeffs.shape[1] == 37 ):
            c23 = znk_coeffs[:, 22 ]
            c24 = znk_coeffs[:, 23 ]
            c25 = znk_coeffs[:, 24 ]
            c26 = znk_coeffs[:, 25 ]
            c27 = znk_coeffs[:, 26 ]
            c28 = znk_coeffs[:, 27 ]
            c29 = znk_coeffs[:, 28 ]
            c30 = znk_coeffs[:, 29 ]
            c31 = znk_coeffs[:, 30 ]
            c32 = znk_coeffs[:, 31 ]
            c33 = znk_coeffs[:, 32 ]
            c34 = znk_coeffs[:, 33 ]
            c35 = znk_coeffs[:, 34 ]
            c36 = znk_coeffs[:, 35 ]
            c37 = znk_coeffs[:, 36 ]


        if( znk_coeffs.shape[1] == 11):
            c12 = c13 = c14 = c15 = c16 = c17 = c18 = c19 = c20 = c21 = \
                c22 = c23 = c24 = c25 = c26 = c27 = c28 = c29 = c30 =  \
                c31 = c32 = c33 = c34 = c35 = c36 = c37 = 0.0
        elif( znk_coeffs.shape[1] == 22 ):
            c23 = c24 = c25 = c26 = c27 = c28 = c29 = c30 =  \
                c31 = c32 = c33 = c34 = c35 = c36 = c37 = 0.0

        mu_xx = (41.76245727*c4**2 + 30.17329672*c4*c6 + 49.34522499*c4*c11 +
                60.23923033*c4*c12 + 127.5864144*c4*c22 + 41.93214979*c4*c24 +
                66.20356646*c4*c37 + 5.450037513*c5**2 + 21.76136523*c5*c13 +
                15.14794962*c5*c23 + 5.450037513*c6**2 + 17.82591606*c6*c11 +
                21.76136523*c6*c12 + 46.09047206*c6*c22 + 15.14794962*c6*c24 +
                23.91597604*c6*c37 + 24.01523843*c7**2 + 33.38730501*c7*c9 +
                40.63340435*c7*c17 + 50.38665632*c7*c19 + 72.37305702*c7*c29 +
                45.85795805*c7*c31 + 36.55127763*c8**2 + 33.38730501*c8*c10 +
                139.8407978*c8*c16 + 50.38665632*c8*c18 + 104.9814668*c8*c30 +
                45.85795805*c8*c32 + 11.60421683*c9**2 + 28.24539653*c9*c17 +
                35.02515013*c9*c19 + 50.30850175*c9*c29 + 31.87712745*c9*c31 +
                11.60421683*c10**2 + 28.24539653*c10*c16 + 35.02515013*c10*c18 +
                50.30850175*c10*c30 + 31.87712745*c10*c32 + 208.8122863*c11**2 +
                140.3026252*c11*c12 + 226.1282287*c11*c22 + 299.6693439*c11*c24 +
                560.3021601*c11*c37 + 65.04308935*c12**2 + 48.04436987*c12*c14 +
                132.6527868*c12*c22 + 163.1540446*c12*c24 + 62.76208421*c12*c26 +
                188.2357244*c12*c37 + 65.04308935*c13**2 + 48.04436987*c13*c15 +
                163.1540446*c13*c23 + 62.76208421*c13*c25 + 19.75752427*c14**2 +
                48.37159799*c14*c24 + 51.61992573*c14*c26 + 19.75752427*c15**2 +
                48.37159799*c15*c23 + 51.61992573*c15*c25 + 337.417232*c16**2 +
                140.2373507*c16*c18 + 519.0591841*c16*c30 + 231.7589896*c16*c32 +
                113.2280797*c17**2 + 140.2373507*c17*c19 + 163.5731024*c17*c29 +
                231.7589896*c17*c31 + 87.89494365*c18**2 + 66.18103812*c18*c20 +
                127.9331333*c18*c30 + 226.1329287*c18*c32 + 80.01883281*c18*c34 +
                87.89494365*c19**2 + 66.18103812*c19*c21 + 127.9331333*c19*c29 +
                226.1329287*c19*c31 + 80.01883281*c19*c33 + 29.86526998*c20**2 +
                72.17197206*c20*c32 + 72.21959985*c20*c34 + 29.86526998*c21**2 +
                72.17197206*c21*c31 + 72.21959985*c21*c33 + 584.6744017*c22**2 +
                384.3145009*c22*c24 + 606.7657092*c22*c37 + 258.7265639*c23**2 +
                184.1252185*c23*c25 + 258.7265639*c24**2 + 184.1252185*c24*c26 +
                472.7920968*c24*c37 + 118.1835551*c25**2 + 88.35874393*c25*c27 +
                118.1835551*c26**2 + 88.35874393*c26*c28 + 41.92993083*c27**2 +
                41.92993083*c28**2 + 309.2842309*c29**2 + 360.3632011*c29*c31 +
                839.28343*c30**2 + 360.3632011*c30*c32 + 300.6936497*c31**2 +
                232.4337672*c31*c33 + 300.6936497*c32**2 + 232.4337672*c32*c34 +
                155.6806027*c33**2 + 114.8615133*c33*c35 + 155.6806027*c34**2 +
                114.8615133*c34*c36 + 55.96525792*c35**2 + 55.96525792*c36**2 +
                1252.873718*c37**2)

        mu_yy = (41.76245727*c4**2 - 30.17329672*c4*c6 + 49.34522499*c4*c11 -
                60.23923033*c4*c12 + 127.5864144*c4*c22 - 41.93214979*c4*c24 +
                66.20356646*c4*c37 + 5.450037513*c5**2 + 21.76136523*c5*c13 +
                15.14794962*c5*c23 + 5.450037513*c6**2 - 17.82591606*c6*c11 +
                21.76136523*c6*c12 - 46.09047206*c6*c22 + 15.14794962*c6*c24 -
                23.91597604*c6*c37 + 36.55127763*c7**2 - 33.38730501*c7*c9 +
                139.8407978*c7*c17 - 50.38665632*c7*c19 + 104.9814668*c7*c29 -
                45.85795805*c7*c31 + 24.01523843*c8**2 - 33.38730501*c8*c10 +
                40.63340435*c8*c16 - 50.38665632*c8*c18 + 72.37305702*c8*c30 -
                45.85795805*c8*c32 + 11.60421683*c9**2 - 28.24539653*c9*c17 +
                35.02515013*c9*c19 - 50.30850175*c9*c29 + 31.87712745*c9*c31 +
                11.60421683*c10**2 - 28.24539653*c10*c16 + 35.02515013*c10*c18 -
                50.30850175*c10*c30 + 31.87712745*c10*c32 + 208.8122863*c11**2 -
                140.3026252*c11*c12 + 226.1282287*c11*c22 - 299.6693439*c11*c24 +
                560.3021601*c11*c37 + 65.04308935*c12**2 - 48.04436987*c12*c14 -
                132.6527868*c12*c22 + 163.1540446*c12*c24 - 62.76208421*c12*c26 -
                188.2357244*c12*c37 + 65.04308935*c13**2 - 48.04436987*c13*c15 +
                163.1540446*c13*c23 - 62.76208421*c13*c25 + 19.75752427*c14**2 -
                48.37159799*c14*c24 + 51.61992573*c14*c26 + 19.75752427*c15**2 -
                48.37159799*c15*c23 + 51.61992573*c15*c25 + 113.2280797*c16**2 -
                140.2373507*c16*c18 + 163.5731024*c16*c30 - 231.7589896*c16*c32 +
                337.417232*c17**2 - 140.2373507*c17*c19 + 519.0591841*c17*c29 -
                231.7589896*c17*c31 + 87.89494365*c18**2 - 66.18103812*c18*c20 -
                127.9331333*c18*c30 + 226.1329287*c18*c32 - 80.01883281*c18*c34 +
                87.89494365*c19**2 - 66.18103812*c19*c21 - 127.9331333*c19*c29 +
                226.1329287*c19*c31 - 80.01883281*c19*c33 + 29.86526998*c20**2 -
                72.17197206*c20*c32 + 72.21959985*c20*c34 + 29.86526998*c21**2 -
                72.17197206*c21*c31 + 72.21959985*c21*c33 + 584.6744017*c22**2 -
                384.3145009*c22*c24 + 606.7657092*c22*c37 + 258.7265639*c23**2 -
                184.1252185*c23*c25 + 258.7265639*c24**2 - 184.1252185*c24*c26 -
                472.7920968*c24*c37 + 118.1835551*c25**2 - 88.35874393*c25*c27 +
                118.1835551*c26**2 - 88.35874393*c26*c28 + 41.92993083*c27**2 +
                41.92993083*c28**2 + 839.28343*c29**2 - 360.3632011*c29*c31 +
                309.2842309*c30**2 - 360.3632011*c30*c32 + 300.6936497*c31**2 -
                232.4337672*c31*c33 + 300.6936497*c32**2 - 232.4337672*c32*c34 +
                155.6806027*c33**2 - 114.8615133*c33*c35 + 155.6806027*c34**2 -
                114.8615133*c34*c36 + 55.96525792*c35**2 + 55.96525792*c36**2 +
                1252.873718*c37**2 )

        mu_xy = (30.17329672*c4*c5 + 12.5360392*c7*c8 + 17.82591606*c5*c11 +
                60.23923033*c4*c13 - 33.38730501*c7*c10 + 33.38730501*c8*c9 +
                49.60369672*c7*c16 + 140.3026252*c11*c13 - 50.38665632*c7*c18 +
                49.60369672*c8*c17 + 28.24539653*c9*c16 + 41.93214979*c4*c23 +
                46.09047206*c5*c22 + 50.38665632*c8*c19 - 28.24539653*c10*c17 +
                48.04436987*c12*c15 - 48.04436987*c13*c14 + 224.1891523*c16*c17 +
                299.6693439*c11*c23 + 132.6527868*c13*c22 + 140.2373507*c16*c19 -
                140.2373507*c17*c18 + 16.30420491*c7*c30 + 16.30420491*c8*c29 +
                62.76208421*c12*c25 - 48.37159799*c14*c23 - 45.85795805*c7*c32 +
                45.85795805*c8*c31 + 50.30850175*c9*c30 - 50.30850175*c10*c29 -
                62.76208421*c13*c26 + 48.37159799*c15*c24 + 66.18103812*c18*c21 -
                66.18103812*c19*c20 + 23.91597604*c5*c37 + 177.7430408*c16*c29 +
                384.3145009*c22*c23 + 231.7589896*c16*c31 + 177.7430408*c17*c30 -
                127.9331333*c18*c29 - 231.7589896*c17*c32 + 127.9331333*c19*c30 -
                184.1252185*c23*c26 + 184.1252185*c24*c25 + 188.2357244*c13*c37 +
                80.01883281*c18*c33 - 72.17197206*c20*c31 - 80.01883281*c19*c34 +
                72.17197206*c21*c32 - 88.35874393*c25*c28 + 88.35874393*c26*c27 +
                529.9991991*c29*c30 + 472.7920968*c23*c37 - 360.3632011*c29*c32 +
                360.3632011*c30*c31 - 232.4337672*c31*c34 + 232.4337672*c32*c33 -
                114.8615133*c33*c36 + 114.8615133*c34*c35)
        
        # convert rad^2 to pixel^2 where 0.2arcsec = 1 pix (4.2 half diameter of M1)
        convfac = self.wl_array[self.wl_index] ** 2 / 4.2**2 * 206265**2 / 0.2**2
        
        # offset = 0.271298 # for ftt?
        # slope  = 1.18
        # offset = 1.271298   # for rays
        # slope  = 1.38
        
        offset = 0          # to compare against numerical integration
        slope = 1

        mu_xx *= convfac
        mu_yy *= convfac
        mu_xy *= convfac

        mu_xx = mu_xx * slope + offset
        mu_yy = mu_yy * slope + offset
        mu_xy = mu_xy * slope
        
        # mu_xx += offset
        # mu_yy += offset


        
        #ellipticity
        p = mu_xx + mu_yy                   # pix^2
        e1 = ( mu_xx - mu_yy ) / p          #
        e2 = 2 * mu_xy / p                  #
        q = np.sqrt( e1**2 + e2**2 ) * p    # pix^2
        sigma_l = np.sqrt( ( p + q ) / 2 )
        sigma_s = np.sqrt( ( p - q ) / 2 )
        ellipticity_adimensional = 1 - sigma_s/sigma_l
        alpha = np.rad2deg( np.arctan2( e2, e1 ) / 2 )
        
        # print( q * np.cos( alpha ), q * np.sin( alpha ), mu_xx, mu_yy, mu_xy )
        #print("%.3d %5.2f %10.2f" %( cnt, ellipticity_adimensional, alpha ) )
        

        sl = sigma_l.tolist()
        ss = sigma_s.tolist()
        el = ellipticity_adimensional.tolist()
        pa = alpha.tolist()          # degrees
        # xc.append( x_cen )
        # yc.append( y_cen )
        mu_xx_list = mu_xx.tolist()
        mu_yy_list = mu_yy.tolist()
        mu_xy_list = mu_xy.tolist()
        p_list = p.tolist()
        q_list = q.tolist()
        e1_list = e1.tolist()
        e2_list = e2.tolist()


        self.ellipticity = {'sl': sl, 'ss': ss, 'el': el, 
                         'pa': pa, 
                         # 'xc': xc, 'yc': yc,
                         'muxx': mu_xx_list,
                         'muyy': mu_yy_list,
                         'muxy': mu_xy_list,
                         'p': p_list,
                         'q': q_list,
                         'e1': e1_list,
                         'e2': e2_list
                            }
        
        return
        
    def plot_c_dbl( self, iwl, k, cpup, cfld ):
        """
        

        Parameters
        ----------
        iwl : int
            wavelength index
        k : int
            index of DOF [0-9]
        cpup : int
            index of pupil zernike
        cfld : int
            index of field zernike

        Returns
        -------
        None.

        """
        dof_txt=['M2 dz', 'M2 dx', 'M2 dy', 'M2 Rx', 'M2 Ry',
                 'C  dz', 'C  dx', 'C  dy', 'C  Rx', 'C  Ry']
        npert = self.batoid_cubes[iwl].npert
        nznkpup = self.batoid_cubes[iwl].nznkpupil

        y = self.c_dblznk[ cpup, cfld, k*npert:k*npert+npert, iwl] # pup, fld, zeta, wl
        x = self.batoid_cubes[iwl].zeta[:, k]
        
        xmod = np.linspace(x.min(), x.max(), num=50)
        ymod = self.a1[cpup, cfld, k, iwl ] * xmod + self.a2[cpup, cfld, k, iwl ] * xmod**2
        
        fig, ax = plt.subplots()
        
        plt.plot( x, y, '.' )
        plt.plot( xmod, ymod, '-' )
        if( k>9 ):
            ax.set_title("C_%d_%d dof=%d  *nznkpup=%s" %( cpup, cfld, k, nznkpup ))
        else:
            ax.set_title("C_%d_%d dof=%d %s *nznkpup=%s" %( cpup, cfld, k, dof_txt[k], nznkpup ))
        plt.text(0, ymod.max(),'%.1e\n%.1e' %(self.a1[cpup, cfld, k, iwl ],self.a2[cpup, cfld, k, iwl ] ), va='top')
        
        plt.show()

            
def get_dbl_znk_coeff( d, znk_inv ):
    """
    Remove nominal coeffs and project onto zernike base
    ( nfield x npupil x nzeta  ) *  ( nfield x nznkbasis ) -> 
      npupil x nznkbasis x nzeta

    Parameters
    ----------
    d : TYPE
        DESCRIPTION.
    znk_inv : TYPE
        DESCRIPTION.

    Returns
    -------
    double zernike coefficients

    """
    c_dblznk = np.zeros( [d[0].nznkpupil,           # npupil 
                          znk_inv.shape[1],         # nbasis
                          d[0].ndof * d[0].npert,   # n distorsions
                          len(d) ])                 # n wavelength
    
    for iwl in range( len( d ) ):     # do this for each wavelength
        
        for i in range( 0,  d[iwl].nznkpupil ):
            
            plane = ( d[iwl].cube[:,i,:].T - d[iwl].nominal[:,i].T ).T #remove
                                                                       #nominal
            #project on basis            
            c_dblznk[ i, :, :, iwl ] = np.matmul( znk_inv.T, plane )
            
        #TODO: probably the sum over i can be done at once with einsum

    return c_dblznk

def fit_misaligned_coeffs( d, c_dblznk ):
    """
    For each wavelength and for each dblznk coefficient in the list of cubes 
    c_dblznk, fit a simple Ax^2 + Bx function.

    Parameters
    ----------
    d : object
        list of batoid cubes
    c_dblznk : list of numpy 3d arrays
        the dbl_zernike coefficients, one cube for each wavelength

    Returns
    -------
    the fitted coefficients A, B in the function Ax^2 + Bx

    """
    #### TODO:  fit_misaligned_coefficients should be an internal function
    ####   that is not used outside, _fit_mal_coeffs?
    
    nznk_field = c_dblznk.shape[1]
    nznk_pupil = d[0].nznkpupil
    npert = d[0].npert
    ndof =  d[0].ndof
    
    a1sol = np.zeros( [nznk_pupil, nznk_field, ndof, len( d ) ] )
    a2sol = np.zeros( [nznk_pupil, nznk_field, ndof, len( d ) ] )
    
    #fit one coefficient at a time
    for iwl in range( len(d) ):                 # wl
        for i in range( 3, nznk_pupil ):            # c pupil
            for k in range( ndof ):                   # dof
                idx = np.arange( k * npert,             #0123,4567, etc
                                 k * npert + npert )

                x = d[iwl].zeta[:,k]
                y = c_dblznk[ i, 1:, idx, iwl ] #fit all cj's at once
        
                A = np.vstack([ x, x**2 ]).T
                sol = np.linalg.lstsq(A, y, rcond=None)
                
                a1sol[i,1:,k,iwl] = sol[0][0,:]
                a2sol[i,1:,k,iwl] = sol[0][1,:]
                
    return a1sol, a2sol

def WF_difference( coeffs1, coeffs2, mode = 0 ):
    # get the WF for each set of coefficients and return its difference
    #  first we need to ignore the first 3 coeffs which are not fitted for
    # We have 3 modes of doing this:
    #  first 3 coeffs are equalized coeff2[0:3] = coeff1[0:3]
    #  first 1 coeffs are -2    # creates numerical outliers errors
    #  first 3 coeffs are 0

    c_pup1 = np.copy( coeffs1 )
    c_pup2 = np.copy( coeffs2 )
    
    if mode==0:
        c_pup2[0:3] = c_pup1[0:3]
    elif( mode == 1): #only the first equal to a constant
        c_pup1[0:3] = c_pup2[0:3] = 0
        c_pup1[0] =   c_pup2[0] = -2
    elif( mode == 2 ):# all 3 zero
        c_pup1[0:3] = c_pup2[0:3] = 0
        
    wf_1 = wf_mesh( c_pup1 )
    wf_2 = wf_mesh( c_pup2 )
    differ = wf_1 - wf_2 
    
    return differ, wf_1, wf_2
    

def WF_difference_stats( coeff_plane1, coeff_plane2 ):
    """
    return stats from the differences between WF constructued with given coeffs

    Note that the input is a plane with ncoords x pupil coeffs so we have 
        ncoords comparisons.
    Note that the first 3 coeffs are discarded as they are not fitted for.

    Parameters
    ----------
    coeff_plane1,2 : array with pupil coeffs, one for each coordinate.
        

    Returns
    -------
    mean, max

    """
    results = np.zeros( coeff_plane1.shape[0] )
    for i in range( coeff_plane1.shape[0] ):
        differ, wf_1, wf_2 = WF_difference(coeff_plane1[i,:], 
                                           coeff_plane2[i,:],
                                           mode=1 )
        # c_pup1 = np.copy( coeff_plane1[i,:] )
        # c_pup2 = np.copy( coeff_plane2[i,:] )
        
        # c_pup1[0:3] = 0
        # c_pup2[0:3] = 0
        # c_pup1[0] = -2
        # c_pup2[0] = -2
        # # c_pup1[0:3] = c_pup2[0:3]
        
        # wf_1 = wf_mesh( c_pup1 )
        # wf_2 = wf_mesh( c_pup2 )
        differ = np.abs( differ )
        diff_per = ( differ / np.abs( wf_1 ) ) * 100.
        # print( "%.3d %.3d %10.2E %10.2E" %(wl, i, 
        #                                 np.mean(diff_per), diff_per.max()))
        results[i] = diff_per.max()
        # results[i] = differ.max()
            
    # print( 'worst:', results.max(), np.argmax( results ) )
    return results.mean(), results.std(), results.max(), np.argmax( results )
    
    
def plot_WF( coeffs1, coeffs2, residual = True, mode=2 ):
    #gimmy 2 wavefronts and plot them side by side
    # if residual is true, plot first and residual, otherwise plot them
    #   side by side
    
    diff, wf1, wf2 = WF_difference(coeffs1, coeffs2, mode=mode )
    # c1 = np.copy( coeffs1 )
    # c2 = np.copy( coeffs2 )
    
    # c1[0] = -2
    # c2[0] = -2
    # c1[1:3] = 0.0
    # c2[1:3] = 0.0
    
    # c2[0:3] = c1[0:3]
    
    # wf1 = wf_mesh( c1 )
    # wf2 = wf_mesh( c2 )
    
    fig, ax = plt.subplots()
    im = ax.pcolormesh( wf1, 
                  cmap='jet')
    # ax.autoscale(False)
    # ax.axis('equal')
    ax.set_aspect('equal', 'box')
    # kk = 0.5e-5
    # ax.set_xlim((-kk, kk))         # set ROI to -1,1 for both x,y
    # ax.set_ylim((-kk, kk))
    ax.set_title('WF1')
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    fig.colorbar(im, ax=ax)
    

    fig, ax = plt.subplots()
    if( residual ):
        im = ax.pcolormesh( (wf1-wf2), 
                      cmap='jet')
    else:
        im = ax.pcolormesh( wf2, 
                      cmap='jet')
        
    # ax.autoscale(False)
    # ax.axis('equal')
    ax.set_aspect('equal', 'box')
    # kk = 0.5e-5
    # ax.set_xlim((-kk, kk))         # set ROI to -1,1 for both x,y
    # ax.set_ylim((-kk, kk))
    ax.set_title('WF2')
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    fig.colorbar(im, ax=ax)
    
    return    
    
    
if __name__=='__main__':

    # fname = 'znk_batoid_coeffs_wl_2_jmax_22_dbg.hdf5'
    # fname = 'znk_batoid_coeffs_wl_6_jmax_22.hdf5'
    fname = 'znk_batoid_coeffs_wl_2_jmax_37_dbg.hdf5'
    # fname = 'znk_batoid_coeffs_wl_6_jmax_37.hdf5'
    # fname = 'znk_batoid_coeffs_wl_2_jmax_11_dbg.hdf5'

    lm = lsstModel( batoid_cube_file=fname, n_fldznk=22)
    print( lm.batoid_cubes[0] )
    
    for i in range( 10 ):
        lm.plot_c_dbl( 0, i, 3, 1 ) #iwl, idof, c_pup, c_fld
    

    sys.exit(0)
    



    
    
    
    
    
