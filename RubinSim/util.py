#!/usr/bin/env python

import numpy as np
import batoid
import galsim
import h5py

def zernike_optimal_sampling( n ):
    """
    % Generate Zernike polynomial optimal sampling grid
    %
    % x,y = zernike_optimal_sampling(n)
    %
    % Formula from the paper:
    % Ramon-Lopez et al., "Optimal sampling patterns for Zernike polynomials"
    % https://arxiv.org/pdf/1511.00449.pdf
    %
    % Inputs:
    % - n : maximum radial order of the Zernike polynomial to sample.
    %
    % Outputs:
    % - x,y : cartesian coordinates of the sampling grid as column vectors.
    % - rho,theta : polar coordinates of the sampling grid as column vectors.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    by SS in matlab
    translated by RzS
    """

    k = n//2 + 1
    j = np.linspace( 1, k, num=k)
    nj = 2*n + 5 - 4*j
    zita_jn = np.cos( np.pi *(2*j - 1 ) / (2*(n+1) ) )
    rj = 1.1565*zita_jn - 0.76535*zita_jn**2. + 0.60517*zita_jn**3.
    
    theta_list=[]
    rho_list = []
    for nn, rho in zip( nj, rj ):
        for sj in range( 1, int( nn ) + 1):
            theta = 2*np.pi*(sj-1)/ int( nn )
            rho_list.append( rho )
            theta_list.append( theta )

    rho = np.asarray( rho_list )
    theta = np.asarray( theta_list )
    if( __debug__):
        print('*** dbg: slicing field points to a tenth')
        rho= rho[0:-1:10]
        theta=theta[0:-1:10]
    
    x = rho * np.cos( theta )
    y = rho * np.sin( theta )
    
    
    
    return x, y, rho, theta

def zernikeStd( p, A ):
    """
    Evaluates zernike terms up to order 37 in Noll indexing convention.    

    Parameters
    ----------
    p : 1D-array
        pupil coordinate rho.
    A : 1D-array
        pupil coordinate theta angle.

    Returns
    -------
    array with 37 terms evaluted at rho/theta to be multiplied by their coeffs

    """
    
    z = np.zeros((p.size, 37))
    
    z[:,0] = np.ones( p.size )
    z[:,1] = 4**(1/2) * (p) * np.cos(A)
    z[:,2] = 4**(1/2) * (p) * np.sin(A)
    z[:,3] = 3**(1/2) * (2*p**2 - 1)
    z[:,4] = 6**(1/2) * (p**2) * np.sin(2*A)
    z[:,5] = 6**(1/2) * (p**2) * np.cos(2*A)
    z[:,6] = 8**(1/2) * (3*p**3 - 2*p) * np.sin(A)
    z[:,7] = 8**(1/2) * (3*p**3 - 2*p) * np.cos(A)
    z[:,8] = 8**(1/2) * (p**3) * np.sin(3*A)
    z[:,9] = 8**(1/2) * (p**3) * np.cos(3*A)
    z[:,10]= 5**(1/2) * (6*p**4 - 6*p**2 + 1)
    z[:,11]= 10**(1/2) * (4*p**4 - 3*p**2) * np.cos(2*A)
    z[:,12]= 10**(1/2) * (4*p**4 - 3*p**2) * np.sin(2*A)
    z[:,13]= 10**(1/2) * (p**4) * np.cos(4*A)
    z[:,14]= 10**(1/2) * (p**4) * np.sin(4*A)
    z[:,15]= 12**(1/2) * (10*p**5 - 12*p**3 + 3*p) * np.cos(A)
    z[:,16]= 12**(1/2) * (10*p**5 - 12*p**3 + 3*p) * np.sin(A)
    z[:,17]= 12**(1/2) * (5*p**5 - 4*p**3) * np.cos(3*A)
    z[:,18]= 12**(1/2) * (5*p**5 - 4*p**3) * np.sin(3*A)
    z[:,19]= 12**(1/2) * (p**5) * np.cos(5*A)
    z[:,20]= 12**(1/2) * (p**5) * np.sin(5*A)
    z[:,21]= 7**(1/2) * (20*p**6 - 30*p**4 + 12*p**2 - 1)
    z[:,22]= 14**(1/2) * (15*p**6 - 20*p**4 + 6*p**2) * np.sin(2*A)
    z[:,23]= 14**(1/2) * (15*p**6 - 20*p**4 + 6*p**2) * np.cos(2*A)
    z[:,24]= 14**(1/2) * (6*p**6 - 5*p**4) * np.sin(4*A)
    z[:,25]= 14**(1/2) * (6*p**6 - 5*p**4) * np.cos(4*A)
    z[:,26]= 14**(1/2) * (p**6) * np.sin(6*A)
    z[:,27]= 14**(1/2) * (p**6) * np.cos(6*A)
    z[:,28]= 16**(1/2) * (35*p**7 - 60*p**5 + 30*p**3 - 4*p) * np.sin(A)
    z[:,29]= 16**(1/2) * (35*p**7 - 60*p**5 + 30*p**3 - 4*p) * np.cos(A)
    z[:,30]= 16**(1/2) * (21*p**7 - 30*p**5 + 10*p**3) * np.sin(3*A)
    z[:,31]= 16**(1/2) * (21*p**7 - 30*p**5 + 10*p**3) * np.cos(3*A)
    z[:,32]= 16**(1/2) * (7*p**7 - 6*p**5) * np.sin(5*A)
    z[:,33]= 16**(1/2) * (7*p**7 - 6*p**5) * np.cos(5*A)
    z[:,34]= 16**(1/2) * (p**7) * np.sin(7*A)
    z[:,35]= 16**(1/2) * (p**7) * np.cos(7*A)
    z[:,36]= 9**(1/2) * (70*p**8 - 140*p**6 + 90*p**4 - 20*p**2 + 1)
    
    return z

def h5database_to_numpy( fname ):
    """
    read datasets into numpy arrays.

    Parameters
    ----------
    fname : str
        input h5 file.

    Returns
    -------
    a list of tuples with required coefficient arrays. Note that between 
    matlab and numpy, arrays need to be transposed to keep the same row/col
    order. The data is placed in a list of tuples with the following order:
        ( nominal, factor, exponent ) for each wavelength.

    """
    
    with h5py.File( fname, 'r' ) as f:
        coeff_list = []
        # grps = list( f.keys() )
        for grp in f:
            nomC = np.transpose( np.asarray(f[grp]['nominalCoeffs'] ))
            ffacC= np.transpose( np.asarray(f[grp]['fieldDependentCoeffs_factor']))
            fexpC= np.transpose( np.asarray(f[grp]['fieldDependentCoeffs_exponent']))
            coeff_list.append(( nomC, ffacC, fexpC ) )
                       # np.asarray(f[grp]['fieldDependentCoeffs_factor']),
                       # np.asarray(f[grp]['fieldDependentCoeffs_exponent'])) )
    return coeff_list            

def LSSTAberrations( perturbation, znk_basis, coeffs_tuple ):
    """
    Applies the model contained in the coeffs_tuple that has 3 terms:
        nominal, factor, exponent coefficients.

    Parameters
    ----------
    perturbation : float array
        Column array with 9 perturbations 
    znk_basis : float 2D array
        nfieldxnzernike array with the basis zernike terms (zernike terms
            evaluated at rho/theta )
    coeffs_tuple : tuple of coefficients for a wavelength
        coefficients read from the h5 file for a single wavelength. The order
        is: (nominal, factor, exponent) as given by h5database_to_numpy()

    Returns
    -------
    A matrix with Zernike pupil coefficients for each field point.

    """
    
    #unpack tuple
    nominalCoeffs, fieldDependentCoeffs_factor, fieldDependentCoeffs_exponent = coeffs_tuple
    
    nfield, nzernike = znk_basis.shape
    ZernikeField = np.transpose( znk_basis )
    NFieldDependentPerturbations = 6
    NZernikePupil, NZernikeField = nominalCoeffs.shape
    
    # The current model includes 6 distorsions
    fieldDependentPerturbation = perturbation[0:NFieldDependentPerturbations]
    
    #create a cube of nzernikePupilxnzernikeFieldx6 for matrix multiplication
    zita3D = np.broadcast_to( np.transpose(fieldDependentPerturbation),
                             [1,1,fieldDependentPerturbation.size])
    perturbationMatrix = np.tile( zita3D, [NZernikePupil,NZernikeField,1] )
        
    # Find Nominal Pupil Zernike terms for these nfield points
    nominalAberrations = np.matmul( nominalCoeffs, ZernikeField )
    print( nominalCoeffs.shape, ZernikeField.shape, nominalAberrations.shape)
    
    # Apply field dependent model: Factor * distorsion ** exponent and sum
    #    along distorsion axis to produce a matrix nzernike_pupilxnzernike_field
    #    Multiply by zernike_basis to obtain zernike_pupil terms for each field
    #     point.
    fieldDependentAberrations = np.matmul( np.sum(fieldDependentCoeffs_factor*perturbationMatrix**fieldDependentCoeffs_exponent,2),ZernikeField)    
    print( fieldDependentAberrations.shape )

    # constant aberrations are copied from original distorsion array    
    constantAberrations = np.zeros( (NZernikePupil, nfield ) )      
    constantAberrations[3,:] = np.tile( perturbation[6],(1,nfield)) #defocus
    constantAberrations[4,:] = np.tile( perturbation[8],(1,nfield)) #astigmatism x
    constantAberrations[5,:] = np.tile( perturbation[7],(1,nfield)) #astigmatism y
    
    print( constantAberrations.shape)
    
    # Misterious crosscorrelation term
    crossCorrelationMatrix = np.zeros( (NZernikePupil, nfield) )
    c4 = 2*fieldDependentCoeffs_factor[3,0,0] * (perturbation[0]*perturbation[2] + perturbation[1]*perturbation[3])
    crossCorrelationC4 = np.tile( c4, (1, nfield ) )
    crossCorrelationMatrix[3,:] = crossCorrelationC4
    
    #sum up all arrays.    
    # aberrationCoefficientsPupil = nominalAberrations+constantAberrations+fieldDependentAberrations
    aberrationCoefficientsPupil = nominalAberrations + constantAberrations + fieldDependentAberrations + crossCorrelationMatrix
    
    
    return aberrationCoefficientsPupil
    

def fit_zernike_coefficients(  tel1, thx, thy, wavelength, n_terms ):
    """Fit a wavefront with zernike polynomials with galsim.

    Parameters
    ----------
    tel1: batoid optic
        the rubin telescope represented with batoid
    thx, thy: float
        field angle in radians
    wavelength: float
        in meters
    n_terms: int 
        number of coefficients for the polynomial.
    
    Returns
    -------
    zeta : array
        nrow x len( amp )  array

    Notes
    -----
    
    """    
    opd = batoid.wavefront(
            tel1,
            np.deg2rad(thx), np.deg2rad(thy),
            wavelength,
            nx=255
        ).array
    
    xs = np.linspace(-1, 1, opd.shape[0])
    ys = np.linspace(-1, 1, opd.shape[1])
    xs, ys = np.meshgrid(xs, ys)
    w = ~opd.mask
    basis = galsim.zernike.zernikeBasis( n_terms, xs[w], ys[w], R_inner=0.61)
    zk, *_ = np.linalg.lstsq(basis.T, opd[w], rcond=None)

    return zk.data[1:]

def coefficients2wavefront( Xp, Yp, Z_abe, wavelength, pupil_mask ):
    """
    

    Parameters
    ----------
    Xp : TYPE
        DESCRIPTION.
    Yp : TYPE
        DESCRIPTION.
    Z_abe : TYPE
        DESCRIPTION.
    wavelength : TYPE
        DESCRIPTION.
    pupil_mask : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    pupil_mask = np.reshape( pupil_mask, pupil_mask.size )
    
    # Pupil plane
    Np = Xp.size
    M, N = Xp.shape
    
    RHOp = np.sqrt( Xp**2 + Yp**2 )
    THETAp = np.arctan2( Yp, Xp )
    
    RHOp = np.reshape( RHOp, Np )
    THETAp = np.reshape( THETAp, Np )
    
    Zernike_p = zernikeStd( RHOp, THETAp)
    
    # Compute Wavefront
    
    W = np.matmul( Zernike_p, Z_abe ) * pupil_mask / 1e9
    
    # Coherent (complex) transfer function
    E = np.exp( -1j * 2 * np.pi * W / wavelength ) * pupil_mask
    
    # restore input matrix shape
    W = np.fft.ifftshift( np.reshape( W, (M, N) ) )
    E = np.fft.ifftshift( np.reshape( E, (M, N) ) )
    
    PSF = np.abs( np.fft.ifftshift( np.fft.ifft2( E )))**2.
    
    return W, E, PSF
    
    
    