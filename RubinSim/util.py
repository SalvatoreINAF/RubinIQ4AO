#!/usr/bin/env python

import numpy as np
import numpy.ma as ma
import batoid
import galsim
import h5py
import matplotlib.pyplot as plt

def zernike_optimal_sampling( n, debug=False ):
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
    if(  debug ):
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
    

def fit_zernike_coefficients(  tel1, thx, thy, wavelength, n_terms, reference ):
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
    reference: str
        'chief' or 'mean'
    
    Returns
    -------
    zeta : array
        nrow x len( amp )  array

    Notes
    -----
    
    """
    opd = batoid.wavefront(
            tel1,
            thx, thy,
            wavelength,
            nx=255, reference=reference
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

def rayPSF( telescope, theta_x, theta_y, wavelength, nphot=1000,
           seeing=0.5):
    """
    Compute PSF with a random number of rays
    

    Parameters
    ----------
    optic :             (batoid.Optic)  – Optical system
    theta_x, theta_y :  (float)         - Field angle in radians
    wavelength :        (float)         – Wavelength in meters
    nphot :             (int, optional) – Number of uniformly sampled rays 
                                            from annular region.
        
    seeing :            (float)         - FWHM of seeing in arcsec.

    Returns
    -------
    (batoid.Lattice) – A 21x21 pixel batoid.Lattice object containing the 
    relative PSF values and the primitive lattice vectors of the focal 
    plane grid. Returned coordinates are in um.

    """
    
    rng = np.random.default_rng()
    
    bins = 21   #box of 21x21 pixels
    bo2 = bins//2
    myrange = [[-bo2*10e-6, bo2*10e-6], [-bo2*10e-6, bo2*10e-6]]
    
    
    rv = batoid.RayVector.asPolar(
        optic=telescope, wavelength=wavelength,
        nrandom=nphot, rng=rng,
        theta_x=theta_x, theta_y=theta_y )
    telescope.trace( rv )
    rv.x[:] -= np.mean(rv.x[~rv.vignetted])
    rv.y[:] -= np.mean(rv.y[~rv.vignetted])
    scale = 10e-6 * seeing/2.35/0.2     #sigma in um, with seeing as FWHM
    rv.x[:] += rng.normal(scale=scale, size=len(rv))
    rv.y[:] += rng.normal(scale=scale, size=len(rv))
    # Bin rays
    psf, _, _ = np.histogram2d(
        rv.y[~rv.vignetted], rv.x[~rv.vignetted], bins=bins,
        range=myrange )
    dx=10e-6
    dy=10e-6
    primitive_vectors = [[dx,0],[0,dy]]
    return batoid.Lattice( psf, primitive_vectors)

class batoid_coeffs_cube( object ):
    def __init__(self, grp):
        self.wl             = grp.attrs['wl']
        self.nfield         = grp.attrs['nfield']
        self.nznkpupil      = grp.attrs['nzernike']
        self.ndof           = grp.attrs['n_dof']    #10
        self.npert          = grp.attrs['npert']    #4 different amplitudes
        self.field_radius   = grp.attrs['field_radius']
        
        self.coords         = np.asarray(grp['coords'] ) #2 x nfield
        self.cube           = np.asarray(grp['cube']) #nfield x ncoeffs x ndist
        self.nominal        = np.asarray( grp['nominal'] ) #nfield x ncoeffs
        self.zeta           = np.asarray( grp['zeta'] ) #npert x ndof array
        
    def __str__( self ):
        print("_"*40)
        print("%-15s %-20s" %( "# array", "shape" ) )
        print("%-15s %-20s" %( 'cube', self.cube.shape ) )
        print("%-15s %-20s" %( 'nominal', self.nominal.shape ) )
        print("%-15s %-20s" %( 'coords', self.coords.shape ) )
        print("%-15s %-20s" %( 'zeta', self.zeta.shape ) )
        print("%-15s %-20s" %( "\n# attributes", "value" ) )
        print("%-15s %-20s" %( 'wl', self.wl ) )
        print("%-15s %-20s" %( 'nfield', self.nfield ) )
        print("%-15s %-20s" %( 'nznkpupil', self.nznkpupil ) )
        print("%-15s %-20s" %( 'ndof', self.ndof ) )
        print("%-15s %-20s" %( 'npert', self.npert ) )
        print("%-15s %-20s" %( 'field_radius', self.field_radius ) )
        
        
        return ""

def read_h5_coeffs( fname ):
    """
    Reads the file previously created in python with batoid coefficients to be
       fitted for lsst model
    ----------
    fname : str
        input h5 file name with data and metadata.

    Returns
    -------
    A list of batoid cubes with (nfield x znkpupil x zeta) and meta data
    """
    with h5py.File( fname, 'r' ) as f:
        lista = []
        for grp in f:
            lista.append( batoid_coeffs_cube( f[grp] ))   
            
    return lista

def wf_mesh( zk_coeff, n=256 ):
    """
    returns a wavefront in a n x n mesh based on the input zernike coefficients

    Parameters
    ----------
    zk_coeff : array
        zernike coefficients, the first coefficient is NOT a dummy as in galsim
    n : integer, optional
        DESCRIPTION. The default is 256.

    Returns
    -------
    wf : masked array
        a masked array where the wavefront is valid, e.g. a donut Rinner=0.61
        fractional up to Router=1.0

    """

    # We need to add the previously removed dummy coefficient 0 to use galsim
    zk = np.pad( zk_coeff, (1,0), 'empty' )

    xs = np.linspace(-1, 1, n )
    ys = np.linspace(-1, 1, n)
    xs, ys = np.meshgrid(xs, ys)

    wf = galsim.zernike.Zernike(zk, R_inner=0.61)(xs, ys)
    mask = ( ( np.sqrt(xs**2+ys**2) >= 1. ) | ( np.sqrt(xs**2+ys**2) <= 0.61 ) )

    wf = ma.array( wf, mask=mask )

    return wf

def plot_wf( wfarr, title ):
    """
    Plot a 2D mesh array.

    Parameters
    ----------
    wfarr : 2d array
        a WF error
    title : str

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    im = ax.pcolormesh( wfarr, cmap='jet' )
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    ax.set_title('WF. %s' %title )
    fig.colorbar(im, ax=ax)

    plt.show()
    return

def plot_one_psf( one, title ):
    """
    plot the array representing a PSF.

    Parameters
    ----------
    one : array
        a psf, tipically the result of a FFT.
    title : str

    Returns
    -------
    None.

    """

    psfnorm = one / one.max()

    fig, ax = plt.subplots()
    im = ax.pcolormesh( psfnorm, cmap='jet' )
                  # cmap='jet')
    # ax.autoscale(False)
    # ax.axis('equal')
    ax.set_aspect('equal', 'box')
    # kk = 10.0e-6  #size of one pixel
    # kk = 1

    xc = yc = one.shape[0]/2
    boxsize = 200
    ax.set_xlim(( xc-boxsize/2, xc+boxsize/2))         # set ROI to -1,1 for both x,y
    ax.set_ylim((yc-boxsize/2, yc+boxsize/2))
    ax.set_title('PSF fft. %s' %title )
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    fig.colorbar(im, ax=ax)

    plt.show()
    return

def regular_grid( radius, num=20 ):
    """
    Return x, y field angle coordinates filling regularly a grid.

    Parameters
    ----------
    radius : float
        Field of view in degrees

    Returns
    -------
    array
        hx coordinates [deg]
    array
        hy coordinates [deg]

    """

    x = np.linspace(-radius*0.95, radius*0.95, num=num )
    xx, yy = np.meshgrid( x, x )

    ikeep = np.sqrt( xx**2 + yy**2 ) < radius   ##test=prova

    return xx[ikeep].flatten(), yy[ikeep].flatten()

def plot_ellipticity_map( x, y, ellipticity, fov ):
    """
    plot a map of ellipticy sticks at field coordinates [deg] x, y

    Parameters
    ----------
    x, y : arrays
        field coordinates in degrees.
    ellipticity : dictionary containing several lists with the following keys:
        'sl':           sl,
        'ss':           ss,
        'el':           magnitude
        'pa':           position angle
        'xc', 'yc':     centroid
    fov: float
        maximum field of view to plot

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots()

    U = ellipticity['el'] * np.cos( np.deg2rad( ellipticity['pa'] ) )
    V = ellipticity['el'] * np.sin( np.deg2rad( ellipticity['pa'] ) )
    M = np.hypot(U, V)
    ax.quiver( x, y, U, V, M, units='xy', scale=1.5, headwidth=1,
              headlength=0, headaxislength=0, pivot = 'middle',
              linewidth=0.8)
    # ax.scatter( x, y, color='black', s=1)

    # plt.colorbar(label='e')
    # ax.clim(0.01, 0.12)

    ax.set_xlim(-fov, fov )
    ax.set_ylim(-fov, fov )
    ax.set_aspect('equal', 'box')
    ax.set_title('ellipticity' )
    ax.set_xlabel('hx [deg]')
    ax.set_ylabel('hy [deg]')

    plt.show()
    
def psf_fft_coeffs( coeffs, nx = 360 ):
    pad_factor = 4
    # nx = 360
        
    wfarr = wf_mesh( coeffs, n=nx  )
                
    # wfarr = wf.array
    pad_size = nx*pad_factor
    expwf = np.zeros((pad_size, pad_size), dtype=np.complex128)
    start = pad_size//2-nx//2
    stop = pad_size//2+nx//2
    expwf[start:stop, start:stop][~wfarr.mask] = \
        np.exp(2j*np.pi*wfarr[~wfarr.mask])
    psf = np.abs(np.fft.fftshift(np.fft.fft2(expwf)))**2
    
    return psf