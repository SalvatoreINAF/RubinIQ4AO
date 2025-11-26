#!/usr/bin/env python

import numpy as np
import numpy.ma as ma
import batoid
import galsim
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable


def zernike_optimal_sampling( n, step=1 ):
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
    # - step: for testing reduce the number of points. array[::step]
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
    
    rho= rho[0::step]
    theta=theta[0::step]

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

def wf_mesh( zk_coeff, n=256, xmin=-1, xmax=1 ):
    """
    returns a wavefront in a n x n mesh based on the input zernike coefficients
    use xmin/xmax > 1 to get the desired physical scale.

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

    xs = np.linspace(xmin, xmax, n )
    #ys = np.linspace(xmin, xmax, n)
    xs, ys = np.meshgrid(xs, xs)

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

def plot_one_psf( one, title, boxsize = -1 ):
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

    if( boxsize > 0 ):
        xc = yc = one.shape[0]/2
        ax.set_xlim(( xc-boxsize/2, xc+boxsize/2))         # set ROI to -1,1 for both x,y
        ax.set_ylim((yc-boxsize/2, yc+boxsize/2))

    ax.set_title('PSF fft. %s' %title )
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    fig.colorbar(im, ax=ax)

    plt.show()
    return

def regular_grid( radius, num=21 ):
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
    
    x = np.linspace(-radius, radius, num=num ) #max used to be 0.85*radius
    xx, yy = np.meshgrid( x, x )

    ikeep = np.sqrt( xx**2 + yy**2 ) <= radius   ##test=prova

    return xx[ikeep].flatten(), yy[ikeep].flatten()

def plot_ellipticity_map( x, y, ellipticity, fov, scale=1.5, title='ellipticity',saveit=False, redbox=False, maxRows=500 ):
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

    if isinstance(ellipticity, dict):
        U = ellipticity['el'] * np.cos( np.deg2rad( ellipticity['pa'] ) )
        V = ellipticity['el'] * np.sin( np.deg2rad( ellipticity['pa'] ) )
    else:
        # ellipticity can now be an array of (Npoints,2) with the first
        #  column 'el' and second 'PA'
        n = len( ellipticity[:,0] )
        if n > maxRows:
            rng = np.random.default_rng()
            indices = rng.choice(n, maxRows, replace=False)
            ellipticity = ellipticity[indices,:]
            x = x[indices]
            y = y[indices]
        U = ellipticity[:,0] * np.cos( np.deg2rad( ellipticity[:,1] ) )
        V = ellipticity[:,0] * np.sin( np.deg2rad( ellipticity[:,1] ) )


    M = np.hypot(U, V)
    q = ax.quiver( x, y, U, V, M, units='xy', scale=scale, headwidth=1,
              headlength=0, headaxislength=0, pivot = 'middle',
              linewidth=0.8)
    if( M.max() > 0.1):
        ulabel = 0.1
    else:
        ulabel = 0.1
    ax.quiverkey(q, X=0.075, Y=0.95, U=ulabel,
             label='$\\epsilon$=%.2f' %ulabel, labelpos='S', color='r', fontproperties={"size":8} )
    # reference ellipticity bar
    # ax.quiver( [-1.5], [1.5], [0.1], [0], [0.1], units='xy', scale=scale, headwidth=1,
    #           headlength=0, headaxislength=0, pivot = 'middle',
    #           linewidth=0.1, color='r')

    print( 'plotting min/max:', M.min(), M.max() )

    # ax.scatter( x, y, color='black', s=1)

    # plt.colorbar(label='e')
    # ax.clim(0.01, 0.12)

    ax.set_xlim(-fov, fov )
    ax.set_ylim(-fov, fov )
    ax.set_aspect('equal', 'box')
    ax.set_title( title )
    ax.set_xlabel('hx [deg]')
    ax.set_ylabel('hy [deg]')
    
    # ellip = patches.Ellipse((0,0), (.75), (.3), angle=30, linewidth=1, edgecolor='r', facecolor='none' )
    if( redbox ):
        ellip = patches.Rectangle((-0.35,-0.35), (.7), (.7), angle=0, linewidth=1, edgecolor='r', facecolor='none' )

        #ax.add_patch(rect)
        ax.add_patch(ellip)

    if( saveit ):
        title = title.replace('=','_')
        outfile = 'ellip_'+title+'.png'
        fullname = '/tmp/'+outfile
        print( 'saved: ',fullname )
        fig.savefig(fullname, format='png', dpi=300)

    # plt.show()
    plt.show(block=False)
    plt.pause(0.001)

def makeFocalPlanePlot( 
    fig,
    axes,
    x, y,
    ellip_dic,
    maxPoints=1000,
    saveAs=None,
):
    """Plot the PSFs in focal plane (detector) coordinates i.e. the raw shapes.

    Top left:
        A scatter plot of the T values in square arcseconds.
    Top right:
        A quiver plot of e1 and e2
    Bottom left:
        A scatter plot of e1
    Bottom right:
        A scatter plot of e2

    This function plots the data from the ``table`` on the provided ``fig`` and
    ``axes`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `numpy.ndarray`
        The table containing the data to be plotted.
    camera : `list`
        The list of camera detector objects.
    maxPoints : `int`, optional
        The maximum number of points to plot. If the number of points in the
        table is greater than this value, a random subset of points will be
        plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    # table = randomRows(table, maxPoints)
    p = np.asarray( ellip_dic['p' ] ) * 0.2**2
    e1= np.asarray( ellip_dic['e1'] )
    e2= np.asarray( ellip_dic['e2'] )
    e = np.asarray( ellip_dic['q' ] ) * 0.2**2


    cbar = addColorbarToAxes(axes[0, 0].scatter(x, y, c=p, s=5))
    # cbar = addColorbarToAxes(axes[0, 0].scatter(x, y, p))
    cbar.set_label("T [arcsec$^2$]")

    emax = np.quantile(np.abs(np.concatenate((e1, e2))), 0.98)
    cbar = addColorbarToAxes(
        axes[1, 0].scatter(x, y, c=e1, vmin=-emax, vmax=emax, cmap="bwr", s=5)
    )
    cbar.set_label("e1")

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(x, y, c=e2, vmin=-emax, vmax=emax, cmap="bwr", s=5)
    )
    cbar.set_label("e2")

    Q = axes[0, 1].quiver(
        x,
        y,
        e * np.cos(0.5 * np.arctan2(e2, e1)),
        e * np.sin(0.5 * np.arctan2(e2, e1)),
        headlength=0,
        headaxislength=0,
        scale=None,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.05, label="0.05", labelpos="S")

    for ax in axes.ravel():
        ax.set_xlabel("Focal Plane x [deg]")
        ax.set_ylabel("Focal Plane y [deg]")
        ax.set_aspect("equal")

    # # Plot camera detector outlines
    # for det in camera:
    #     xs = []
    #     ys = []
    #     for corner in det.getCorners(FOCAL_PLANE):
    #         xs.append(corner.x)
    #         ys.append(corner.y)
    #     xs.append(xs[0])
    #     ys.append(ys[0])
    #     xs = np.array(xs)
    #     ys = np.array(ys)
    #     for ax in axes.ravel():
    #         ax.plot(xs, ys, c="k", lw=1, alpha=0.3)

    fig.tight_layout()
    if saveAs:
        fig.savefig(saveAs)

def addColorbarToAxes(mappable):
    """Add a colorbar to the given axes.

    Parameters
    ----------
    mappable : `matplotlib.cm.ScalarMappable`
        The mappable object to which the colorbar will be added.

    Returns
    -------
    cbar : `matplotlib.colorbar.Colorbar`
        The colorbar object that was added to the axes.
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    return cbar



    
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

def submatsum(data,n,m):
    """
    A way to decrease the resolution of an FFT produced psf image.
       We can call this when we produce the PSF in the model via FFT.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # return a matrix of shape (n,m)
    bs = data.shape[0]//n,data.shape[1]//m  # blocksize averaged over
    return np.reshape(np.array([np.sum(data[k1*bs[0]:(k1+1)*bs[0],k2*bs[1]:(k2+1)*bs[1]]) for k1 in range(n) for k2 in range(m)]),(n,m))

def zernike_mode_name( i ):
    """returns the name of the zernike term, Noll and starting with 0"""

    classical_name=['Piston',                               #0
                    'Horizontal Tilt',                      #1
                    'Vertical Tilt',                        #2
                    'Defocus',                              #3
                    'Oblique Astigmatism',                  #4
                    'Vertical Astigmatism',                 #5
                    'Vertical Coma',                        #6
                    'Horizontal Coma',                      #7
                    'Vertical Trefoil',                     #8
                    'Oblique Trefoil',                      #9
                    'Primary Spherical',                    #10
                    'Vertical Secondary Astigmatism',       #11
                    'Oblique Secondary Astigmatism',        #12
                    'Vertical Quadrifoil',                  #13
                    'Oblique Quadrifoil'                    #14
                    ]

    try:
        result = classical_name[ i ]
    except:
        result = str( i )

    return result

def mxx_analytic( znk_coeffs ):
    #testing matlab formula. kk is the inner radius. 
    # noll zernike coefficients start with 1: c1, c2, c3... c11
    kk = 0.61
    
    # using annular zernike
    
    # c1  = znk_coeffs[ 0 ]
    # c2  = znk_coeffs[ 1 ]
    # c3  = znk_coeffs[ 2 ]
    c4  = znk_coeffs[:, 3 ]
    c5  = znk_coeffs[:, 4 ] 
    c6  = znk_coeffs[:, 5 ]
    c7  = znk_coeffs[:, 6 ]
    c8  = znk_coeffs[:, 7 ]
    c9  = znk_coeffs[:, 8 ]
    c10 = znk_coeffs[:, 9 ]
    c11 = znk_coeffs[:, 10 ]
    
    if( znk_coeffs.shape[1] == 11):
        return (0.31830989*(95.620311*c4**2*kk**4 - 95.620311*c4**2 + 69.085495*c4*c6*kk**4 - 69.085495*c4*c6 + 1572.8019*c4*c11*kk**6 - 1618.5311*c4*c11*kk**4 + 45.729216*c4*c11 + 12.478535*c5**2*kk**4 - 12.478535*c5**2 + 12.478535*c6**2*kk**4 - 12.478535*c6**2 + 568.17321*c6*c11*kk**6 - 584.69285*c6*c11*kk**4 + 16.519636*c6*c11 + 49.945832*c7**2*kk**6 - 49.945832*c7**2 + 69.437442*c7*c9*kk**6 - 69.437442*c7*c9 + 149.8375*c8**2*kk**6 - 299.67499*c8**2*kk**4 + 299.67499*c8**2*kk**2 - 149.8375*c8**2 + 69.437442*c8*c10*kk**6 - 69.437442*c8*c10 + 24.133938*c9**2*kk**6 - 24.133938*c9**2 + 24.133938*c10**2*kk**6 - 24.133938*c10**2 + 7275.9623*c11**2*kk**8 - 13311.131*c11**2*kk**6 + 6849.0758*c11**2*kk**4 - 813.90761*c11**2))/(kk**2 - 1.0)    
        
    elif( znk_coeffs.shape[1] == 22 ):
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
        return -(0.31830989*(- 95.620311*c4**2*kk**4 + 95.620311*c4**2 - 69.085495*c4*c6*kk**4 + 69.085495*c4*c6 - 1572.8019*c4*c11*kk**6 + 1618.5311*c4*c11*kk**4 - 45.729216*c4*c11 - 423.95502*c4*c12*kk**6 + 328.81109*c4*c12*kk**4 + 95.14393*c4*c12 - 11114.216*c4*c22*kk**8 + 20333.088*c4*c22*kk**6 - 10023.949*c4*c22*kk**4 + 805.0771*c4*c22 - 12.478535*c5**2*kk**4 + 12.478535*c5**2 - 153.15335*c5*c13*kk**6 + 118.7827*c5*c13*kk**4 + 34.370655*c5*c13 - 12.478535*c6**2*kk**4 + 12.478535*c6**2 - 568.17321*c6*c11*kk**6 + 584.69285*c6*c11*kk**4 - 16.519636*c6*c11 - 153.15335*c6*c12*kk**6 + 118.7827*c6*c12*kk**4 + 34.370655*c6*c12 - 4015.0001*c6*c22*kk**8 + 7345.3088*c6*c22*kk**6 - 3621.1421*c6*c22*kk**4 + 290.83335*c6*c22 - 49.945832*c7**2*kk**6 + 49.945832*c7**2 - 69.437442*c7*c9*kk**6 + 69.437442*c7*c9 - 919.40475*c7*c17*kk**8 + 866.25518*c7*c17*kk**6 + 53.149568*c7*c17 - 467.21883*c7*c19*kk**8 + 378.36225*c7*c19*kk**6 + 88.856584*c7*c19 - 149.8375*c8**2*kk**6 + 299.67499*c8**2*kk**4 - 299.67499*c8**2*kk**2 + 149.8375*c8**2 - 69.437442*c8*c10*kk**6 + 69.437442*c8*c10 - 2758.2143*c8*c16*kk**8 + 6276.3846*c8*c16*kk**6 - 5197.5311*c8*c16*kk**4 + 1519.9121*c8*c16*kk**2 + 159.44871*c8*c16 - 467.21883*c8*c18*kk**8 + 378.36225*c8*c18*kk**6 + 88.856584*c8*c18 - 24.133938*c9**2*kk**6 + 24.133938*c9**2 - 639.10352*c9*c17*kk**8 + 602.15779*c9*c17*kk**6 + 36.945726*c9*c17 - 324.77666*c9*c19*kk**8 + 263.01*c9*c19*kk**6 + 61.766654*c9*c19 - 24.133938*c10**2*kk**6 + 24.133938*c10**2 - 639.10352*c10*c16*kk**8 + 602.15779*c10*c16*kk**6 + 36.945726*c10*c16 - 324.77666*c10*c18*kk**8 + 263.01*c10*c18*kk**6 + 61.766654*c10*c18 - 7275.9623*c11**2*kk**8 + 13311.131*c11**2*kk**6 - 6849.0758*c11**2*kk**4 + 813.90761*c11**2 - 3922.5292*c11*c12*kk**8 + 6292.2777*c11*c12*kk**6 - 2782.8344*c11*c12*kk**4 + 413.08588*c11*c12 - 109686.7*c11*c22*kk**10 + 282189.6*c11*c22*kk**8 - 254524.49*c11*c22*kk**6 + 84835.916*c11*c22*kk**4 - 2814.3258*c11*c22 - 587.40742*c12**2*kk**8 + 728.93043*c12**2*kk**6 - 282.67198*c12**2*kk**4 + 141.14896*c12**2 - 96.624985*c12*c14*kk**8 + 96.624985*c12*c14 - 29566.486*c12*c22*kk**10 + 69819.535*c12*c22*kk**8 - 57181.617*c12*c22*kk**6 + 17234.756*c12*c22*kk**4 - 306.18804*c12*c22 - 587.40742*c13**2*kk**8 + 728.93043*c13**2*kk**6 - 282.67198*c13**2*kk**4 + 141.14896*c13**2 - 96.624985*c13*c15*kk**8 + 96.624985*c13*c15 - 39.735571*c14**2*kk**8 + 39.735571*c14**2 - 39.735571*c15**2*kk**8 + 39.735571*c15**2 - 13539.55*c16**2*kk**10 + 35202.045*c16**2*kk**8 - 31877.336*c16**2*kk**6 + 11253.398*c16**2*kk**4 - 1927.1985*c16**2*kk**2 + 888.64096*c16**2 - 4586.977*c16*c18*kk**10 + 7534.1501*c16*c18*kk**8 - 3281.1373*c16*c18*kk**6 + 333.96417*c16*c18 - 4513.1832*c17**2*kk**10 + 7973.029*c17**2*kk**8 - 3756.0594*c17**2*kk**6 + 296.21365*c17**2 - 4586.977*c17*c19*kk**10 + 7534.1501*c17*c19*kk**8 - 3281.1373*c17*c19*kk**6 + 333.96417*c17*c19 - 1238.3377*c18**2*kk**10 + 1769.6969*c18**2*kk**8 - 716.56626*c18**2*kk**6 + 185.20707*c18**2 - 131.48707*c18*c20*kk**10 + 131.48707*c18*c20 - 1238.3377*c19**2*kk**10 + 1769.6969*c19**2*kk**8 - 716.56626*c19**2*kk**6 + 185.20707*c19**2 - 131.48707*c19*c21*kk**10 + 131.48707*c19*c21 - 59.335678*c20**2*kk**10 + 59.335678*c20**2 - 59.335678*c21**2*kk**10 + 59.335678*c21**2 - 430612.13*c22**2*kk**12 + 1418023.0*c22**2*kk**10 - 1798599.1*c22**2*kk**8 + 1065766.4*c22**2*kk**6 - 262704.52*c22**2*kk**4 + 8126.3864*c22**2))/(kk**2 - 1.0)
    elif( znk_coeffs.shape[1] == 37 ):
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
        return (41.76245727*c4**2 + 30.17329672*c4*c6 + 49.34522499*c4*c11 + 
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

    

def myy_analytic( znk_coeffs ):
    # using annular zernike
    #testing matlab formula. kk is the inner radius. 
    # noll zernike coefficients start with 1: c1, c2, c3... c11
    kk = 0.61

    
    # c1  = znk_coeffs[ 0 ]
    # c2  = znk_coeffs[ 1 ]
    # c3  = znk_coeffs[ 2 ]
    c4  = znk_coeffs[:, 3 ]
    c5  = znk_coeffs[:, 4 ] 
    c6  = znk_coeffs[:, 5 ]
    c7  = znk_coeffs[:, 6 ]
    c8  = znk_coeffs[:, 7 ]
    c9  = znk_coeffs[:, 8 ]
    c10 = znk_coeffs[:, 9 ]
    c11 = znk_coeffs[:, 10 ]
    
    

    if( znk_coeffs.shape[1] == 11):
        return (0.31830989*(95.620311*c4**2*kk**4 - 95.620311*c4**2 - 69.085495*c4*c6*kk**4 + 69.085495*c4*c6 + 1572.8019*c4*c11*kk**6 - 1618.5311*c4*c11*kk**4 + 45.729216*c4*c11 + 12.478535*c5**2*kk**4 - 12.478535*c5**2 + 12.478535*c6**2*kk**4 - 12.478535*c6**2 - 568.17321*c6*c11*kk**6 + 584.69285*c6*c11*kk**4 - 16.519636*c6*c11 + 149.8375*c7**2*kk**6 - 299.67499*c7**2*kk**4 + 299.67499*c7**2*kk**2 - 149.8375*c7**2 - 69.437442*c7*c9*kk**6 + 69.437442*c7*c9 + 49.945832*c8**2*kk**6 - 49.945832*c8**2 - 69.437442*c8*c10*kk**6 + 69.437442*c8*c10 + 24.133938*c9**2*kk**6 - 24.133938*c9**2 + 24.133938*c10**2*kk**6 - 24.133938*c10**2 + 7275.9623*c11**2*kk**8 - 13311.131*c11**2*kk**6 + 6849.0758*c11**2*kk**4 - 813.90761*c11**2))/(kk**2 - 1.0)
    elif( znk_coeffs.shape[1] == 22 ):
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
        return -(0.31830989*(- 95.620311*c4**2*kk**4 + 95.620311*c4**2 + 69.085495*c4*c6*kk**4 - 69.085495*c4*c6 - 1572.8019*c4*c11*kk**6 + 1618.5311*c4*c11*kk**4 - 45.729216*c4*c11 + 423.95502*c4*c12*kk**6 - 328.81109*c4*c12*kk**4 - 95.14393*c4*c12 - 11114.216*c4*c22*kk**8 + 20333.088*c4*c22*kk**6 - 10023.949*c4*c22*kk**4 + 805.0771*c4*c22 - 12.478535*c5**2*kk**4 + 12.478535*c5**2 - 153.15335*c5*c13*kk**6 + 118.7827*c5*c13*kk**4 + 34.370655*c5*c13 - 12.478535*c6**2*kk**4 + 12.478535*c6**2 + 568.17321*c6*c11*kk**6 - 584.69285*c6*c11*kk**4 + 16.519636*c6*c11 - 153.15335*c6*c12*kk**6 + 118.7827*c6*c12*kk**4 + 34.370655*c6*c12 + 4015.0001*c6*c22*kk**8 - 7345.3088*c6*c22*kk**6 + 3621.1421*c6*c22*kk**4 - 290.83335*c6*c22 - 149.8375*c7**2*kk**6 + 299.67499*c7**2*kk**4 - 299.67499*c7**2*kk**2 + 149.8375*c7**2 + 69.437442*c7*c9*kk**6 - 69.437442*c7*c9 - 2758.2143*c7*c17*kk**8 + 6276.3846*c7*c17*kk**6 - 5197.5311*c7*c17*kk**4 + 1519.9121*c7*c17*kk**2 + 159.44871*c7*c17 + 467.21883*c7*c19*kk**8 - 378.36225*c7*c19*kk**6 - 88.856584*c7*c19 - 49.945832*c8**2*kk**6 + 49.945832*c8**2 + 69.437442*c8*c10*kk**6 - 69.437442*c8*c10 - 919.40475*c8*c16*kk**8 + 866.25518*c8*c16*kk**6 + 53.149568*c8*c16 + 467.21883*c8*c18*kk**8 - 378.36225*c8*c18*kk**6 - 88.856584*c8*c18 - 24.133938*c9**2*kk**6 + 24.133938*c9**2 + 639.10352*c9*c17*kk**8 - 602.15779*c9*c17*kk**6 - 36.945726*c9*c17 - 324.77666*c9*c19*kk**8 + 263.01*c9*c19*kk**6 + 61.766654*c9*c19 - 24.133938*c10**2*kk**6 + 24.133938*c10**2 + 639.10352*c10*c16*kk**8 - 602.15779*c10*c16*kk**6 - 36.945726*c10*c16 - 324.77666*c10*c18*kk**8 + 263.01*c10*c18*kk**6 + 61.766654*c10*c18 - 7275.9623*c11**2*kk**8 + 13311.131*c11**2*kk**6 - 6849.0758*c11**2*kk**4 + 813.90761*c11**2 + 3922.5292*c11*c12*kk**8 - 6292.2777*c11*c12*kk**6 + 2782.8344*c11*c12*kk**4 - 413.08588*c11*c12 - 109686.7*c11*c22*kk**10 + 282189.6*c11*c22*kk**8 - 254524.49*c11*c22*kk**6 + 84835.916*c11*c22*kk**4 - 2814.3258*c11*c22 - 587.40742*c12**2*kk**8 + 728.93043*c12**2*kk**6 - 282.67198*c12**2*kk**4 + 141.14896*c12**2 + 96.624985*c12*c14*kk**8 - 96.624985*c12*c14 + 29566.486*c12*c22*kk**10 - 69819.535*c12*c22*kk**8 + 57181.617*c12*c22*kk**6 - 17234.756*c12*c22*kk**4 + 306.18804*c12*c22 - 587.40742*c13**2*kk**8 + 728.93043*c13**2*kk**6 - 282.67198*c13**2*kk**4 + 141.14896*c13**2 + 96.624985*c13*c15*kk**8 - 96.624985*c13*c15 - 39.735571*c14**2*kk**8 + 39.735571*c14**2 - 39.735571*c15**2*kk**8 + 39.735571*c15**2 - 4513.1832*c16**2*kk**10 + 7973.029*c16**2*kk**8 - 3756.0594*c16**2*kk**6 + 296.21365*c16**2 + 4586.977*c16*c18*kk**10 - 7534.1501*c16*c18*kk**8 + 3281.1373*c16*c18*kk**6 - 333.96417*c16*c18 - 13539.55*c17**2*kk**10 + 35202.045*c17**2*kk**8 - 31877.336*c17**2*kk**6 + 11253.398*c17**2*kk**4 - 1927.1985*c17**2*kk**2 + 888.64096*c17**2 + 4586.977*c17*c19*kk**10 - 7534.1501*c17*c19*kk**8 + 3281.1373*c17*c19*kk**6 - 333.96417*c17*c19 - 1238.3377*c18**2*kk**10 + 1769.6969*c18**2*kk**8 - 716.56626*c18**2*kk**6 + 185.20707*c18**2 + 131.48707*c18*c20*kk**10 - 131.48707*c18*c20 - 1238.3377*c19**2*kk**10 + 1769.6969*c19**2*kk**8 - 716.56626*c19**2*kk**6 + 185.20707*c19**2 + 131.48707*c19*c21*kk**10 - 131.48707*c19*c21 - 59.335678*c20**2*kk**10 + 59.335678*c20**2 - 59.335678*c21**2*kk**10 + 59.335678*c21**2 - 430612.13*c22**2*kk**12 + 1418023.0*c22**2*kk**10 - 1798599.1*c22**2*kk**8 + 1065766.4*c22**2*kk**6 - 262704.52*c22**2*kk**4 + 8126.3864*c22**2))/(kk**2 - 1.0)
    elif( znk_coeffs.shape[1] == 37 ):
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
        return (41.76245727*c4**2 - 30.17329672*c4*c6 + 49.34522499*c4*c11 - 
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

    
  

def mxy_analytic( znk_coeffs ):
    # using annular zernike
    #testing matlab formula. kk is the inner radius. 
    # noll zernike coefficients start with 1: c1, c2, c3... c11
    kk = 0.61
    
    # c1  = znk_coeffs[ 0 ]
    # c2  = znk_coeffs[ 1 ]
    # c3  = znk_coeffs[ 2 ]
    c4  = znk_coeffs[:, 3 ]
    c5  = znk_coeffs[:, 4 ] 
    c6  = znk_coeffs[:, 5 ]
    c7  = znk_coeffs[:, 6 ]
    c8  = znk_coeffs[:, 7 ]
    c9  = znk_coeffs[:, 8 ]
    c10 = znk_coeffs[:, 9 ]
    c11 = znk_coeffs[:, 10 ]
    
    if( znk_coeffs.shape[1] == 11):
        return (0.31830989*(16.519636*c5*c11 - 99.891664*c7*c8 - 69.085495*c4*c5 + 69.437442*c7*c10 - 69.437442*c8*c9 + 69.085495*c4*c5*kk**4 + 299.67499*c7*c8*kk**2 - 299.67499*c7*c8*kk**4 - 584.69285*c5*c11*kk**4 + 99.891664*c7*c8*kk**6 + 568.17321*c5*c11*kk**6 - 69.437442*c7*c10*kk**6 + 69.437442*c8*c9*kk**6))/(kk**2 - 1.0) 
    elif( znk_coeffs.shape[1] == 22 ):
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
        return -(0.31830989*(69.085495*c4*c5 + 99.891664*c7*c8 - 16.519636*c5*c11 + 95.14393*c4*c13 - 69.437442*c7*c10 + 69.437442*c8*c9 + 53.149568*c7*c16 + 413.08588*c11*c13 - 88.856584*c7*c18 + 53.149568*c8*c17 + 36.945726*c9*c16 + 290.83335*c5*c22 + 88.856584*c8*c19 - 36.945726*c10*c17 + 96.624985*c12*c15 - 96.624985*c13*c14 + 592.42731*c16*c17 - 306.18804*c13*c22 + 333.96417*c16*c19 - 333.96417*c17*c18 + 131.48707*c18*c21 - 131.48707*c19*c20 - 69.085495*c4*c5*kk**4 - 299.67499*c7*c8*kk**2 + 299.67499*c7*c8*kk**4 + 584.69285*c5*c11*kk**4 + 328.81109*c4*c13*kk**4 - 99.891664*c7*c8*kk**6 - 568.17321*c5*c11*kk**6 - 423.95502*c4*c13*kk**6 + 69.437442*c7*c10*kk**6 - 69.437442*c8*c9*kk**6 + 759.95605*c7*c16*kk**2 - 2598.7655*c7*c16*kk**4 + 759.95605*c8*c17*kk**2 - 2782.8344*c11*c13*kk**4 + 2705.0647*c7*c16*kk**6 - 2598.7655*c8*c17*kk**4 + 6292.2777*c11*c13*kk**6 - 3621.1421*c5*c22*kk**4 - 919.40475*c7*c16*kk**8 - 378.36225*c7*c18*kk**6 + 2705.0647*c8*c17*kk**6 + 602.15779*c9*c16*kk**6 - 3922.5292*c11*c13*kk**8 + 7345.3088*c5*c22*kk**6 + 467.21883*c7*c18*kk**8 - 919.40475*c8*c17*kk**8 + 378.36225*c8*c19*kk**6 - 639.10352*c9*c16*kk**8 - 602.15779*c10*c17*kk**6 - 4015.0001*c5*c22*kk**8 - 467.21883*c8*c19*kk**8 + 639.10352*c10*c17*kk**8 - 96.624985*c12*c15*kk**8 + 96.624985*c13*c14*kk**8 - 1927.1985*c16*c17*kk**2 + 11253.398*c16*c17*kk**4 + 17234.756*c13*c22*kk**4 - 28121.276*c16*c17*kk**6 - 57181.617*c13*c22*kk**6 + 27229.016*c16*c17*kk**8 - 3281.1373*c16*c19*kk**6 + 3281.1373*c17*c18*kk**6 + 69819.535*c13*c22*kk**8 - 9026.3665*c16*c17*kk**10 + 7534.1501*c16*c19*kk**8 - 7534.1501*c17*c18*kk**8 - 29566.486*c13*c22*kk**10 - 4586.977*c16*c19*kk**10 + 4586.977*c17*c18*kk**10 - 131.48707*c18*c21*kk**10 + 131.48707*c19*c20*kk**10))/(kk**2 - 1.0)
    elif( znk_coeffs.shape[1] == 37 ):
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
        return (30.17329672*c4*c5 + 12.5360392*c7*c8 + 17.82591606*c5*c11 + 
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
 
# def makeAzElPlot(
#     fig: Figure,
#     axes: np.ndarray[Axes],
#     table: Table,
#     camera: Camera,
#     maxPointsPerDetector: int = 5,
#     saveAs: str = "",
# ) -> None:
def makeAzElPlot(
    fig,
    axes,
    hx, hy,
    ellip_dic,
    saveAs: str = "",
) -> None:
    """Plot the PSFs on the focal plane, rotated to az/el coordinates.

    Top left:
        A scatter plot of the T values in square arcseconds.
    Top right:
        A quiver plot of e1 and e2
    Bottom left:
        A scatter plot of e1
    Bottom right:
        A scatter plot of e2

    This function plots the data from the ``table`` on the provided ``fig`` and
    ``axes`` objects. It also plots the camera detector outlines on the focal
    plane plot, respecting the camera rotation for the exposure.

    If ``saveAs`` is provided, the figure will be saved at the specified file
    path.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object to plot on.
    axes : `numpy.ndarray`
        The array of axes objects to plot on.
    table : `numpy.ndarray`
        The table containing the data to be plotted.
    camera : `list`
        The list of camera detector objects.
    maxPointsPerDetector : `int`, optional
        The maximum number of points per detector to plot. If the number of
        points in the table is greater than this value, a random subset of
        points will be plotted.
    saveAs : `str`, optional
        The file path to save the figure.
    """
    
    p = np.asarray( ellip_dic['p' ] ) * 0.2**2          # aka T
    e1= np.asarray( ellip_dic['e1'] )
    e2= np.asarray( ellip_dic['e2'] )
    e = np.hypot( e1, e2 )              # dimensionless
    
    mm_to_deg = 100 * 0.2 / 3600
    
    fullCameraFactor = 4.5
    quiverScale = 5
    plotLimit = 2.25

    FWHM = np.sqrt( p / 2 * np.log(256))

    cbar = addColorbarToAxes(
        axes[0, 1].scatter(hx, hy, c=FWHM, s=5)
    )
    cbar.set_label("FWHM [arcsec]")

    emax = np.quantile(np.abs(np.concatenate([e1, e2])), 0.98)
    axes[1, 0].scatter(
        hx,
        hy,
        c=e1,
        vmin=-emax,
        vmax=emax,
        cmap="bwr",
        s=5,
    )
    axes[1, 0].text(0.05, 0.92, "e1", transform=axes[1, 0].transAxes, fontsize=10)

    cbar = addColorbarToAxes(
        axes[1, 1].scatter(
            hx,
            hy,
            c=e2,
            vmin=-emax,
            vmax=emax,
            cmap="bwr",
            s=5,
        )
    )
    cbar.set_label("e")
    axes[1, 1].text(0.05, 0.92, "e2", transform=axes[1, 1].transAxes, fontsize=10)


    Q = axes[0, 0].quiver(
        hx,
        hy,
        e * np.cos(0.5 * np.arctan2(e2, e1)),
        e * np.sin(0.5 * np.arctan2(e2, e1)),
        headlength=0,
        headaxislength=0,
        scale=quiverScale,
        pivot="middle",
    )
    axes[0, 1].quiverkey(Q, X=0.08, Y=0.95, U=0.2, label="0.2", labelpos="S")

    for ax in axes[:2, :2].ravel():
        ax.set_aspect("equal")
        ax.set_xlim(-plotLimit, plotLimit)
        ax.set_ylim(-plotLimit, plotLimit)
    for ax in axes[1, :2]:
        ax.set_xlabel("$\\Delta$ Azimuth [deg]")
    for ax in axes[:2, 0]:
        ax.set_ylabel("$\\Delta$ Elevation [deg]")

    # Plot camera detector outlines for single-raft cameras


    # Add histograms
    fwhm_percentile = np.nanpercentile(FWHM, [25, 50, 75])
    e_percentile = np.nanpercentile(e, [25, 50, 75])
    axes[0, 2].hist(FWHM, bins=int(np.sqrt(len(FWHM))), color="C0")
    axes[1, 2].hist(e, bins=int(np.sqrt(len(FWHM))), color="C1")
    text_kwargs = {
        "transform": axes[0, 2].transAxes,
        "fontsize": 10,
        "horizontalalignment": "right",
        "verticalalignment": "top",
    }
    axes[0, 2].text(0.95, 0.95, "FWHM", **text_kwargs)
    axes[0, 2].text(0.95, 0.89, "[arcsec]", **text_kwargs)
    axes[0, 2].text(0.95, 0.83, f"median: {fwhm_percentile[1]:.2f}", **text_kwargs)
    axes[0, 2].text(0.95, 0.77, f"IQR: {fwhm_percentile[2] - fwhm_percentile[0]:.2f}", **text_kwargs)
    axes[0, 2].axvline(fwhm_percentile[1], color="k", lw=2)
    axes[0, 2].axvline(fwhm_percentile[0], color="grey", lw=1)
    axes[0, 2].axvline(fwhm_percentile[2], color="grey", lw=1)

    axes[1, 2].text(
        0.95,
        0.95,
        "e",
        transform=axes[1, 2].transAxes,
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="top",
    )
    axes[1, 2].axvline(e_percentile[1], color="k", lw=2)
    axes[1, 2].axvline(e_percentile[0], color="grey", lw=1)
    axes[1, 2].axvline(e_percentile[2], color="grey", lw=1)
    text_kwargs["transform"] = axes[1, 2].transAxes
    axes[1, 2].text(0.95, 0.89, f"median: {e_percentile[1]:.2f}", **text_kwargs)
    axes[1, 2].text(0.95, 0.83, f"IQR: {e_percentile[2] - e_percentile[0]:.2f}", **text_kwargs)

    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    if saveAs:
        fig.savefig(saveAs)

def moments_and_dof_from_file( fname, radmax = 3 ):
    # we return also the original array to append results to original data
    # expected columns: hx, hy, Ixx, Iyy, Ixy
    # the first line are 10 CSV numbers with DOFs

    try:
        dofs = np.loadtxt( fname, delimiter=',', skiprows=0, max_rows=1 )
    except ValueError:
        dofs = None
        
    if( dofs is None ):
        arr = np.loadtxt( fname, delimiter=',', skiprows=1 )
    else:
        arr = np.loadtxt( fname, delimiter=',', skiprows=2 )
    

    hx, hy = arr[:,0], arr[:,1]

    ikeep = ( np.hypot( hx, hy ) < radmax )

    if( '_vit' in fname ): #vittorio's
        # moments from imsim are in arcsec^2 so convert to pixels.
        mxx, myy, mxy = arr[:,4], arr[:,5], arr[:,6]
    else: #my notebook
        # moments from imsim are in arcsec^2 so convert to pixels.
        mxx, myy, mxy = arr[:,2]/0.2**2, arr[:,3]/0.2**2, arr[:,4]/0.2**2

    hx = hx[ikeep]
    hy = hy[ikeep]
    mxx = mxx[ikeep]
    myy = myy[ikeep]
    mxy = mxy[ikeep]

    return hx, hy, mxx, myy, mxy, dofs