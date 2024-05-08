#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:17:11 2024

@author: zanmar
"""

import numpy as np

import galsim
from util import read_h5_coeffs, wf_mesh
import sys
import matplotlib.pyplot as plt


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
        self.a1 = None                  # polynomial fit a1*x + a2*x^2
        self.a2 = None
        
        self.dof = np.zeros(10)
    
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
            
    def zernike_coeffs( self ):
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
        
        coeffs[:,0:3] = 0.0

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
    
    
    def psf_compute( self ):
        # compute a psf with FFT for each coordinate
        pad_factor = 4
        nx = 360
        
        psf_list = []
        coeffs = self.zernike_coeffs()
        
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
            
            
            psf_list.append( psf )
            cnt += 1

        self.psf = psf_list
        
        return
    
    def ellipticity_compute( self ):
        
        nx, ny = self.psf[0].shape
        sl = []
        ss = []
        el = []
        pa = []
        xc = []
        yc = []
        cnt = 0
        for psf in self.psf:
            psfnorm = psf / psf.max()
            xs = np.linspace(0, nx-1, nx )        # pixel coordintes
            ys = np.linspace(0, ny-1, ny )
            X, Y = np.meshgrid(xs, ys)
            # X = psf.coords[:,:,1]
            # Y = psf.coords[:,:,0]
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
        ymod = lm.a1[cpup, cfld, k, iwl ] * xmod + lm.a2[cpup, cfld, k, iwl ] * xmod**2
        
        fig, ax = plt.subplots()
        
        plt.plot( x, y, '.' )
        plt.plot( xmod, ymod, '-' )
        ax.set_title("C_%d_%d dof=%d %s *nznkpup=%s" %( cpup, cfld, k, dof_txt[k], nznkpup ))
        plt.text(0, ymod.max(),'%.1e\n%.1e' %(lm.a1[cpup, cfld, k, iwl ],lm.a2[cpup, cfld, k, iwl ] ), va='top')
        
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
    



    
    
    
    
    
