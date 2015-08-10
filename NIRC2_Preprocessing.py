# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:57:40 2015

@author: Olivier
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from vip.fits import open_fits as open_fits_vip
from vip.fits import display_array_ds9

#import vip.var.pp_subplots

import warnings
warnings.filterwarnings('ignore')


__all__ = ['open_fits',
           'listing',
           'masterFlat',
           'applyFlat',
           'create_cube_from_frames',
           'plot_surface',
           'moffat',
           'cone',
           'gauss2d',
           'gauss2d_sym',
           'chisquare',
           'vortex_center',
           'vortex_center_routine',
           'timeExtract',
           'cube_crop_frames_optimized',
           'registration']

###############################################################################
###############################################################################
###############################################################################
   
def open_fits(filename, header=False, verbose=False):
    """
    Load a fits file as numpy array.
    
    Parameters
    ----------
    filename : string
        Name of the fits file.

    header : boolean (optional)
        If True, the header is returned along with the data.
        
    Returns
    -------
    out : numpy.array, dict (optional)
        The fits image as a numpy.array and (optional) the header.
    
    Note
    ----
    With non-standard header fits file, several "UserWarning" such as:
    "The following header keyword is invalid or follows an unrecognized 
    non-standard convention" can be returned at the first file opening but are
    ignored after.
    
    """
    try:
        if header:
            return open_fits_vip(filename, header=True, verbose=verbose)
        else:
            return open_fits_vip(filename, header=False, verbose=verbose)
    except IOError: # If a *missing END card* error is raised 
        try:
            import pyfits
        except ImportError:
            print 'Due to a possible missing END card error when opening the fits file {}, the missing pyfits package is required.'.format(filename)
            print 'Download instructions can be found here: '
            print 'http://www.stsci.edu/institute/software_hardware/pyfits/Download'
            if header:
                return (None,None)
            else:
                return None
            
        hdulist = pyfits.open(filename,ignore_missing_end=True)
        image = hdulist[0].data
        if header:
            header = hdulist[0].header
            hdulist.close()
            if verbose:
                print ''
                print 'Fits HDU:0 data and header successfully loaded. Data shape: [{},{}]'.format(image.shape[0],image.shape[1])
            return (image,header)
        else:
            hdulist.close()
            if verbose:
                print ''
                print 'Fits HDU:0 data successfully loaded. Data shape: [{},{}]'.format(image.shape[0],image.shape[1])
            return image


# -----------------------------------------------------------------------------

def create_header(h):
    """
    Create a valid fits header which can be used with write_fits().
    
    Parameters
    ----------
    
    h : dict
        Header formatted as a dict.
        
    Return
    ------
    
    out : astropy header object
    
    """
    from astropy.io.fits import Header
    header_valid = Header(h)
    for h_key in header_valid:
        try:
            header_valid[h_key] = h[h_key]
        except ValueError:
            continue
    return header_valid
    

# -----------------------------------------------------------------------------

def timeExtract(date,time):
    """
    Convert a list of date-time into a datetime object.
    
    Ex: 
    >> date = ['2015-07-10']
    >> time = ['08:06:34.123']    
    >> t = timeExtract(date,time)
    >> print t
        [datetime.datetime(2015,7,10,8,6,34)]

    Parameters
    ----------
    date : list
    
    
    time : list
    
    
    """
    from datetime import datetime
    
    if not isinstance(date,list):
        date = [date]

    if not isinstance(time,list):
        time = [time]        
        
    l = len(date)
    
    return [datetime(int(date[j].split('-')[0]),
                     int(date[j].split('-')[1]),
                     int(date[j].split('-')[2]),
                     int(time[j].split(':')[0]),
                     int(time[j].split(':')[1]),
                     int(time[j].split(':')[2].split('.')[0])) for j in range(l)]
  

# ----------------------------------------------------------------------------- 

def listing(repository, selection=False, ext = 'fits'):
    """
    List all fits files contained in 'repository'. 
    
    Parameters
    ----------
    repository : str
        Path to the repository which contains files to list
        
    selection : boolean (optional)
        If True, each image is opened with DS9 and you are asked to keep or 
        discard it.
        
    ext : str (optional)
        The file extension filter.
        
    Returns
    -------
    out : list of str
        A list with all (or selected) filenames.
        
    """
    from os import listdir
    from os.path import isfile, join    
    
    fileList = [f for f in listdir(repository) if isfile(join(repository,f)) if f.endswith('.'+ext)]

    dim = len(fileList)
    choice = np.ones(dim)    

    if selection:
        for k,f in enumerate(fileList):
            w = open_fits(repository+f)
            display_array_ds9(w)
            choice[k] = int(raw_input('File {}/{} --> {}: keep [1] or discard [0] ? '.format(k+1,dim,repository+f)))
      
        print ''
        print 'DONE !'
    
    return [repository+fileList[j] for j in range(dim) if choice[j] == 1]    
    

# ----------------------------------------------------------------------------- 

def masterFlat(fileList, header=False, norm=True, display=False, save=False):
    """
    Create a master flat (median) from a set of single flat images.
    
    Parameters
    ----------
    fileList : list
        A list of all single flat image paths.
        
    header : boolean (optional)
        If True, the headers of each single flat files are returned into a dict
        file.
        
    norm : boolean (optional)
        If True, the master flat is normalized.
        
    display : boolean (optional)
        If True, the master flat is opened with DS9.
        
    save : boolean (optional)
        If True, the master flat is saved as a fits file in the same folder as 
        the single flat images.
        
    Returns
    -------
    out : numpy.array
        The master flat as a numpy array with the same dimension as the single
        flat images.
        
        If header is True, a dict is also returned as a second output object. 
        
    """
    # TODO: si header est True, ecrire le header dans le masterflat.
    
    # Shape and number of files
    l, c = open_fits(fileList[0]).shape    
    n_image = len(fileList)
    
    # Initializate variables
    flats = np.zeros([l,c,n_image])
    headers = []

    # Loop: open all images and concatenate them into a bigger array
    for j in range(n_image):
        if header:
            flats[:,:,j], h = open_fits(fileList[j], header=True)
            headers.append(h)
        else:
            flats[:,:,j] = open_fits(fileList[j], header=False)
    
    # Create the master flat -> median
    mflat = np.median(flats, axis=2)

    # Normalization
    if norm:
        mflat = mflat/np.median(mflat)

    # Display
    if display:
        display_array_ds9(mflat)

    # Save    
    if save:
        ## Save > Import
        from vip.fits import write_fits
        
        ## Save > Determine the path in which the files will be stored
        index = [k for k,letter in enumerate(fileList[0]) if letter == '/'] 
        if len(index) == 0:
            path = ''
        else:
            path = fileList[0][:index[-1]+1]
            
        ## Save > Write the fits    
        write_fits(path+'mflat.fits',mflat)
    
    # Return output(s)
    if header:    
        return mflat, headers
    else:
        return mflat
    

# ----------------------------------------------------------------------------- 

def applyFlat(fileList, path_mflat, header=False, display=False, save=False, verbose=False):
    """
    Divide all images in the fileList by the master flat. 
    
    Parameters
    ----------
    fileList : list
        A list of all image paths.
        
    path_mflat : str
        Path to the master flat.
        
    display : boolean (optional)
        If True, the master flat is opened with DS9.
        
    save : boolean (optional)
        If True, the processed images are saved as fits files in a new 
        repository ([fileList_path]/processed/).
        
    verbose : boolean (optional)
        If True, additional informations are displayed in the shell.
        
    Returns
    -------
    out : dict
        Dictionary which contains the processed images. Each key corresponds to
        the original file path.
        
    """
    # Open the master flat
    mflat = open_fits(path_mflat, header=False)
    
    # Check if *fileList* is a list and raise an error if not
    if isinstance(fileList, str):
        if fileList.endswith('/'): # If True, fileList is a repository ... 
            fileList = listing(fileList)
        else: # ... otherwise, it's a file path.
            fileList = [fileList]
    elif not isinstance(fileList, list):
        raise TypeError('fileList must be a list or a str, {} given'.format(type(fileList)))
    
    # Initializate few parameters
    processed_all = dict()
    headers = dict()
    
    if save:
        from vip.fits import write_fits
        import os
        
    # Loop: process and handle each file 
    for i, filepath in enumerate(fileList):
        ## Loop > Open file and retrieve the header if needed
        if header:
            raw, headers[fileList[i]] = open_fits(filepath, header=True)
        else:
            raw = open_fits(filepath, header=False)
        
        ## Loop > Process the image and store it 
        processed = raw/mflat
        processed_all[filepath] = processed
        
        ## Loop > display
        if display:
            display_array_ds9(processed)
        
        ## Loop > save
        if save:      
            ### Loop > save > Determine the last / in the filepath to deduce  
            ###               the file path and the last . to deduce the file 
            ###               name.
            index_0 = [k for k,letter in enumerate(filepath) if letter == '/']
            index_1 = [j for j,letter in enumerate(filepath) if letter =='.'] 
            
            if len(index_0) == 0:
                path = ''                
            else:
                path = fileList[0][:index_0[-1]+1]

            filename = filepath[index_0[-1]+1:index_1[-1]]
            
            ### Loop > save > If doest not exist, create the path/flatted/
            ###               repository to store the processed images.
            if not os.path.exists(path+'flatted/'):
                os.makedirs(path+'flatted/')
            
            ### Loop > save > Create valid header
            if header:
                header_valid = create_header(headers[fileList[i]])
            else:
                header_valid = None
            
            ### Loop > save > Write the fits
            output = path+'flatted/'+filename+'_flatted.fits'   
            write_fits(output, processed, header=header_valid, verbose=False)
            
            if verbose:
                print '{} successfully saved'.format(output)

    # Return the output(s)
    if header:
        return processed_all, headers
    else:
        return processed_all




# -----------------------------------------------------------------------------

def create_cube_from_frames(files, header=False, verbose=False, save=False):
    """
    Create a cube (3d numpy.darray) from several fits images. The cube size is
    N x l x c where N is the total number of frames, l x c the size of each 
    frame in pixels.
    
    Parameters
    ----------
    files : list or str
        If list, it contains all the fits image filenames.
        If str, it roots the the repository which contains all fits images.
        
    header : boolean (optional)
        If True, the function returns a list of all fits image headers.
        
    verbose : boolean (optional)
        If True, additional informations are displayed in the shell.

    save : boolean (optional)
        If True, the cube is saved as fits files in the same repository as the 
        single fits images.        
        
    Returns
    -------
    out : numpy.array
        The N x l x c cube.
        If header is True, is also returns a list of all headers.
        
    """
    if isinstance(files,str): # If True, *files* is the path to a directory 
                              # which contains all the fits image to combine.
        file_list = listing(files, selection=False)
        path = files
    elif isinstance(files,list): # If True, *files* is already the list of all
                                 # file names.
        file_list = files
        path = None
    else:
        print '''*files* must be either the list of fits image filenames or the 
                directory which contains all the fits images'''
        return None
   
    if header:
        first, h = open_fits(file_list[0], header=True)
        headers = [create_header(h)]
    else:
        first = open_fits(file_list[0])
        
    l, c = first.shape
    
    cube = np.zeros([len(file_list), l, c])    
    cube[0,:,:] = first    
    
    if verbose:
        print 'Frame {} is added to the cube'.format(file_list[0])
    
    for k,filename in enumerate(file_list[1:]):
        if header:
            temp, h_temp = open_fits(filename, header=True)
        else:
            temp = open_fits(filename, header=False)
            
        if temp.shape != (l,c):
            print 'Each frame must have the same dimension as the first one ({},{}), {} given for {}'.format(l,c,temp.shape,filename)
            continue
        else:
            cube[k+1,:,:] = temp
        
        headers.append(h_temp)
        if verbose:
            print 'Frame {} is added to the cube'.format(filename)
                
    if save:
        from vip.fits import write_fits        
        if path is None:
            index_0 = [k for k,letter in enumerate(files[0]) if letter == '/']
            if len(index_0) == 0:
                path = ''                
            else:
                path = files[0][:index_0[-1]+1]        
        
        write_fits(path+'cube.fits', cube, header=headers[0], verbose=False)
                
        if verbose:
            print ''
            print 'The cube is successfully saved'
    
    if header:        
        return cube, headers
    else:
        return cube
    
    
# -----------------------------------------------------------------------------            

def plot_surface(image, center=None, size=None, output=False, ds9_indexing=True, **kwargs):
    """
    Create a surface plot from image.
    
    By default, the whole image is plotted. The 'center' and 'size' attributs 
    allow to crop the image.
        
    Parameters
    ----------
    image : numpy.array
        The image as a numpy.array.
        
    center : tuple of 2 int (optional, default=None)
        If None, the whole image will be plotted. Otherwise, it locates the
        center of a square in the image.
        
    size : int (optional, default=None)
        If None, the whole image will be plotted. Otherwise, it corresponds to
        the size of a square in the image.

    kwargs:
        Additional attributs are passed to the matplotlib figure() and 
        plot_surface() method.        
    
    Returns
    -------
    out : tuple of 3 numpy.array
        x and y for the grid, and the intensity
        
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if center is None or size is None:
        # If one of them is None, we just plot the whole image
        center = (image.shape[0]//2,image.shape[1]//2)
        size = image.shape[0]
    elif ds9_indexing:
        center = (center[0]-1,center[1]-1)
                
    plt.figure(figsize=kwargs.pop('figsize',(5,5)))
    ax = plt.axes(projection='3d')
    
    x = np.outer(np.arange(0,size,1), np.ones(size))
    y = x.copy().T    
    cy,cx = center
    z = image[cx-size//2:cx+size//2,cy-size//2:cy+size//2]
    
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, **kwargs) 
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$I(x,y)$')
    ax.set_title('Data')
    plt.show()
    
    if output:
        return (x,y,z)
 
 
# -----------------------------------------------------------------------------   

def moffat(x, y, x0, y0, i0, bkg, alpha, beta):
    """
    2-D Moffat profile.
    
    The analytical expression is given by:
    I(r) = bkg + I_0 * [1 + (r/\alpha)^2]^{-\beta}, 
    where bkg is the background value, r = (x^2 + y^2)^(1/2), \alpha is a scale 
    factor and \beta determines the overall shape of the profile.
    
    Parameters
    ----------
    x, y: numpy.array
        The grid where the Moffat profile will be define.
        
    x0, y0 : float
        The position of the Moffat profile maximum intensity.
        
    alpha : float
        The scale factor.
    
    beta : float
        The parameter which determines the overall shape of the profile.
        
    i0 : float
        The maximum intensity.
        
    bkg : float (optional)
        An additive constant to take account of the background.
        
    Returns
    -------
    out : numpy.array
        The 2-D Moffat profile.
    """
    return bkg + i0*(1 + (np.sqrt((x-x0)**2+(y-y0)**2)/alpha)**2)**(-beta)   


# -----------------------------------------------------------------------------
    
def cone(x, y, x0, y0, i0, bkg, radius):
    """
    2-D cone profile.
    
    The analytical expression is given by:
    I(r) = bkg + (1 / tan(alpha)) * (radius-r),        
    where r = (x^2 + y^2)^(1/2) and alpha the cone aperture. 
    
    Parameters
    ----------
    x, y: numpy.array
        The grid where the Moffat profile will be define.
        
    x0, y0 : float
        The position of the Moffat profile maximum intensity.
        
    radius : float
        The scale factor.
    
    alpha : float
        The parameter which determines the overall shape of the profile.
        
    i0 : float
        The maximum intensity.
        
    bkg : float (optional)
        An additive constant to take account of the background.
        
    Returns
    -------
    out : numpy.array
        The 2-D Moffat profile.
    """ 
        
    alpha = np.arctan2(radius,i0)        
            
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    z = bkg + (1/np.tan(alpha))*(radius-r)
    z[r > radius] = bkg

    return z


# -----------------------------------------------------------------------------

def gauss2d(x, y, x0, y0, i0, bkg, sigma_x, sigma_y):
    """
    2-D Gaussian profile.
    
    The analytical expression is given by:
    I(r) = bkg + I_0 * exp[...] 
    where r = (x^2 + y^2)^(1/2) ...
    
    Parameters
    ----------
    x, y: numpy.array
        The grid where the Moffat profile will be define.
        
    x0, y0 : float
        The position of the Moffat profile maximum intensity.
        
    sigma_x : float
        
    
    sigma_y : float
        
        
    i0 : float
        The maximum intensity.
        
    bkg : float (optional)
        An additive constant to take account of the background.
        
    Returns
    -------
    out : numpy.array
        The 2-D Gaussian profile.
    """    
    return bkg + i0 * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2))) 
    
# -----------------------------------------------------------------------------

def gauss2d_sym(x, y, x0, y0, i0, bkg, sigma):
    """
    2-D Gaussian symetrical profile.
    
    The analytical expression is given by:
    I(r) = bkg + I_0 * exp[...] 
    where r = (x^2 + y^2)^(1/2) ...
    
    Parameters
    ----------
    x, y: numpy.array
        The grid where the Moffat profile will be define.
        
    x0, y0 : float
        The position of the Moffat profile maximum intensity.
        
    sigma_x : float
        
    
    sigma_y : float
        
        
    i0 : float
        The maximum intensity.
        
    bkg : float (optional)
        An additive constant to take account of the background.
        
    Returns
    -------
    out : numpy.array
        The 2-D Gaussian profile.    
    """    
    return gauss2d(x, y, x0, y0, i0, bkg, sigma, sigma)    
   
# -----------------------------------------------------------------------------     

def chisquare(model_parameters, x, y, data, fun, n=None):
    """
    Function of merit.
    
    One adopts the reduced chi2 function where the errors/pixel are simply the sqrt of 
    the intensity of the pixel.
    
    Parameters
    ----------
    model_parameters: tuple of float
        The parameters defined by the adopted model. Here, the model is 2-D
        Moffat profile.
        
    x, y : numpy.array
        The grid where the function of merit will be evaluated.
        
    data : numpy.array
        The matrix of intensity. 
        
    fun : callable
        VORTEX signature model. The Vortex_Preprocessing module already include:
        + gauss2d
        + gauss2d_sym
        + moffat
        + cone
        
    n: int (optional, default=None)
        The number of vertices in the grid (i.e. number of pixels in the image).
        If None, n is automatically determined. However, this can be done 
        outside the function when the latter need to be called many times.
    
    Returns
    -------
    out : float
        The reduced chi2.
        
    Note
    ----
    One must have: x.shape == y.shape == data.shape
    
    """
    if x.shape != y.shape:
        print 'x and y must have the same dimension.'
        return np.inf
            
    if n is None:
        n = x.size #np.array([j for j in x.shape]).prod() # Total number of array elements
    
    n_mp = len(model_parameters)      
    z = fun(x,y,*model_parameters)
    
    #print 'chi2 = {}'.format(((z-data)**2/data).sum()/(n-n_mp))
    return ((z-data)**2/data).sum()/(n-n_mp)

# -----------------------------------------------------------------------------

def vortex_center(image, center, size, p_initial, fun, ds9_indexing=True, display=False, verbose=True, savefig=False, **kwargs):
    """
    Determine the VORTEX center in the fits image.
    
    Parameters
    ----------
    image : numpy.array
        The image as an array.
        
    center : tuple
        The center of the square box.
        
    size : int
        The size of the box, in pixels.
        
    p_initial : list
        The initial parameters associated to the adopted model. 
        
    fun : callable
        VORTEX signature model. The Vortex_Preprocessing module already include:
        + gauss2d
        + gauss2d_sym
        + moffat
        + cone        
    
    verbose : boolean (optional)
        If True, additional informations are displayed in the shell.
        
    kwargs
        Additional parameters are passed to scipy.optimize.minimize()
           
    # Moffat: x0, y0, alpha, beta, i0, bkg
    # Cone: x0, y0, radius, i0, bkg
    # Gaussian: x0, y0, sigma_x, sigma_y, i0, bkg
    # Gaussian_sym: x0, y0, sigma_x, i0, bkg
    """
    # Import
    from scipy.optimize import minimize
    
    # Create the grid of pixels
    x = np.outer(np.arange(0,size,1), np.ones(size))
    y = x.copy().T

    # Initializate variables    
    cy, cx = center
    
    if ds9_indexing:
        cx = cx - 1 # To match DS9 (1 -> 1024) and python (0 -> 1023) pixel indexing
        cy = cy - 1 # To match DS9 (1 -> 1024) and python (0 -> 1023) pixel indexing
    
    data = image[cx-size//2:cx+size//2,cy-size//2:cy+size//2]
    n = data.shape[0]*data.shape[1]
    
    # Display
    if display:
        plot_surface(data,figsize=kwargs.pop('figsize',(8,5)), cmap=kwargs.pop('cmap','jet'))
    
    # Start the minimization
    solu = minimize(chisquare,
                    p_initial,
                    args=(x,y,data,fun,n),
                    method = kwargs.pop('method','Nelder-Mead'),
                    options=kwargs.pop('options',None),
                    **kwargs)    
 
    # Determine the absolute position of the VORTEX center when the coordinate
    # of the center of the pixel image[0,0] is [1,1] (as it is for DS9)
    if ds9_indexing:                 
        center_vortex = [cy+1-size/2.-1/2.+solu.x[1],cx+1-size/2.-1/2.+solu.x[0]]  
    else:
        center_vortex = [cy-size/2.+solu.x[1],cx-size/2.+solu.x[0]]

    # Display
    if display:            
        z_best = fun(x,y,*solu.x)
                                           
        labels = ['Data','Model']                       
        toplot = [data,z_best]
        absolute_vmin = np.min(np.concatenate((toplot[0],toplot[1])))                               
        absolute_vmax = np.max(np.concatenate((toplot[0],toplot[1])))
        fig_size = (10,10)        
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
        for k,ax in enumerate(axes.flat):
            ax.plot(solu.x[0],solu.x[1],'+g',markersize=10, markeredgewidth=1)
            im = ax.imshow(toplot[k], vmin=absolute_vmin, vmax=absolute_vmax, interpolation='nearest')            
            ax.set_title(labels[k])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85,0.36,0.035,0.31])
        fig.colorbar(im, cax=cbar_ax)
        if savefig:
            fig.savefig('data_model.png')
        
        fig = plt.figure(figsize=kwargs.pop('figsize',(6,5)))
        plt.hold('on')
        plt.title('Residuals')
        fig2 = plt.contourf(x,y,z_best-data,40,cmap=plt.cm.cool)
        plt.colorbar(fig2)
        
        if savefig:
            fig.savefig('residuals.png')
        
        plt.show()
                              
        
    # Verbose
    if verbose:        
        print ''  
        if solu.success:                            
            print 'Optimization terminated successfully !'
        else:
            print 'The minimization has NOT converged but here is the latest best solution.'
        print ''
        print 'VORTEX center'
        print '-------------'
        print 'Position of the VORTEX center: [{:.3f},{:.3f}]'.format(center_vortex[0],center_vortex[1]) 
        print 'Relative position of the VORTEX center in the box: [{:.3f},{:.3f}]'.format(solu.x[0],solu.x[1])
        
        print ''
        print 'Optimized parameters'
        print '--------------------'
        labels = ['I_0','bkg']
        for j,p in enumerate(solu.x[2:4]):
            print '{}: {:.3f}'.format(labels[j],p)
        
        for k,p in enumerate(solu.x[4:]):
            print 'Additional parameter {}: {:.3f}'.format(k,p)  
        
        print ''
        print 'Function of merit'
        print '-----------------'
        print 'The reduced chi-squared equals {}'.format(solu.fun)
        print ''                 

    return (center_vortex, solu, (x,y))
    
# -----------------------------------------------------------------------------

def vortex_center_routine(path_files, center, size, fun, preprocess=False, path_mflat=None, additional_parameters=None, cards=None, verbose=False, **kwargs):
    """
    Do the same as vortex_center() but for a set of raw or preprocessed images.
    
    For all files in the 'path_files' repository, preprocessing is optionally 
    applied (if preprocess is True), then  the VORTEX center is determined.
    
    Parameters
    ----------
    path_files : str or list
        If str: the path which contains all the files to process.
        If list: all the individual paths for each files.
        
    center : tuple
        The center of the square box.
        
    size : int
        The size of the box, in pixels.
        
    preprocess : boolean (optional)
        If True, preprocess (dividing by the master flat) will by apply to each 
        file.    
        
    path_mflat : str (optional)
        If preprocess is True, path_mflat roots to the master flat fits file.        
        
    additional_parameters: list (optional)
        Additional parameters, depending on which model we've adopted:
        - Moffat: [alpha, beta]
        - Cone: [radius]
        - Gaussian: [sigma_x, sigma_y]
        - Gaussian_sym: [sigma]
    
    verbose : boolean (optional)
        If True or 1, few informations are displayed in the shell. 
        If 2, more informations are displayed.
        
    kwargs
        Additional parameters are passed to scipy.optimize.minimize()
        
    """
    # TODO: p_initial pour chaque image à traiter devrait pouvoir être passé à
    # la fonction
    
    if preprocess and path_mflat is None:
        raise ValueError('If "preprocess" is True, the path to the master flat (path_mflat) must be passed to the function.')
                
    # If required, listing all files in the main repository
    if isinstance(path_files,str):
        if path_files.endswith('/'): # If True, fileList is a repository ... 
            file_list = listing(path_files)
        else:
            file_list = [path_files]
    else:
        file_list = path_files
        
    n = len(file_list)
    
    # First guess (relative center position in the box)
    x_ini, y_ini = (size//2,size//2)
    
    # Outputs initialization
    center_all = np.zeros([n,2])
    success_all = np.zeros(n)
    
    if isinstance(center,tuple):
        center = [center for i in range(n)]
    elif isinstance(center,list):
        if len(center) != n:
            center = [center[0] for i in range(n)]
            print 'If "center" is a list of tuple, it must have the same dimension as "path_files".'
    
    if not isinstance(additional_parameters,list):
        additional_parameters = [additional_parameters]        
    
    # Level of verbose
    if verbose > 1:
        vc_verbose = True
    else:
        vc_verbose = False
    
    # card(s) to extract ?
    if cards is not None:
        cards_all =  {key: [] for key in cards}
    else:
        cards_all = None
    
    # Let's go !
    for k,filename in enumerate(file_list):
        if verbose > 1:
            print '###################################'
            if k+1 < 10:
                print 'Step {}/{}                         #'.format(k+1,n)
            elif k+1 >= 10:
                print 'Step {}/{}                        #'.format(k+1,n)
            print '###################################'
        elif verbose:
            print 'Step {}/{}'.format(k+1,n)
        
        # Load and preprocess images
        if preprocess:
            preprocessed = applyFlat(file_list[k],path_mflat, header=True, display=False, save=False)
            image_flatted, header = preprocessed[0][filename],preprocessed[1][filename]
        else:
            image_flatted, header = open_fits(file_list[k], header=True)
        
        # Background and I0 rought estimation
        bkg_ini = np.median(image_flatted)
        i0_ini = np.max(image_flatted[center[k][0]-size//2:center[k][0]+size//2,center[k][1]-size//2:center[k][1]+size//2]) - bkg_ini
        
        # Initial parameters
        p_initial = np.array([x_ini,y_ini,i0_ini,bkg_ini]+additional_parameters)

        # Extract the user header card(s)
        if cards is not None:
            for card in cards:
                try:
                    cards_all[card].append(header[card])
                except:
                    if verbose > 1:
                        print '{}: invalid header card or not in the header.'.format(card)
                    cards_all[card].append(None)
        
        # Minimization 
        result = vortex_center(image_flatted, 
                                  center[k], 
                                  size, 
                                  p_initial, 
                                  fun,
                                  display= False, 
                                  verbose=vc_verbose,
                                  method = 'Nelder-Mead', 
                                  options = kwargs.pop('options',{'xtol':1e-04, 'maxiter':1e+05,' maxfev':1e+05}),
                                  **kwargs)
        
        center_all[k,:] = result[0]
        success_all[k] = result[1].success

    if verbose:
        print ''
        print 'DONE !'

    if cards is None:    
        return center_all, success_all, file_list
    else:
        return center_all, success_all, file_list, cards_all        


# -----------------------------------------------------------------------------

def registration(fileList, initial_position, final_position, ds9_indexing=True, header=False, verbose=False, display=False, save=False):
    """
    Register (translation, no rotation) a set of fits images. 
    
    Parameters
    ----------
    fileList : str or list
        str: a file path or a repository
        list: list of file paths
        
    initial_position : numpy.array
        Array, shape N x 2 where N = len(fileList). 
        It contains the position of registration. 
        
    final_position : numpy.array
        Array, shape 1 x 2. 
        Position where all images are registered.
        
    ds9_indexing : boolean
        If True, match 1-indexing with Python 0-indexing. Furthermore, pixel 
        coordinates in DS9 is inverted in comparison with the corresponding 
        Python array entry.

    header : boolean
        If True, all the file headers are returned.
        
    verbose : boolean
        If True, informations are displayed in the shell.
        
    display : boolean
        If True, the cropped cube is displayed with DS9.
        
    save : boolean
        If True, the cropped cube is saved.
        
    Return
    ------
    out : if header is False: numpy.array
        Cube of all registered images
        
         if header is True: numpy.array, list
        Cube of all registered images and their headers
            
    """
    from vip.calib import frame_shift
    
    # Check if *fileList* is a list and raise an error if not
    if isinstance(fileList, str):
        if fileList.endswith('/'): # If True, fileList is a repository ... 
            fileList = listing(fileList)
        else: # ... otherwise, it's a file path.        
            fileList = [fileList]
            initial_position = np.array([initial_position])
    elif not isinstance(fileList, list):
        raise TypeError('fileList must be a list or a str, {} given'.format(type(fileList)))

    # Shape and number of files
    l, c = open_fits(fileList[0]).shape    
    n_image = len(fileList)
    
    # If required, convert the DS9 center into Python array center
    if ds9_indexing:
        final_position = final_position[::-1]-1
        initial_position = np.array([initial_position[j][::-1] for j in range(n_image)])-1

    # Initializate variables
    reg = np.zeros([n_image,l,c])    
    headers = [] 

    if save:
        from vip.fits import write_fits
        import os
        
    # Loop: process and handle each file 
    for i, filepath in enumerate(fileList):
        ## Loop > Open file and retrieve the header if needed
        if header:
            raw, h = open_fits(filepath, header=True)
            headers.append(h)
        else:
            raw = open_fits(filepath, header=False)
        
        ## Loop > shift the frame
        shift =  final_position - initial_position[i,:] 
        reg[i,:,:] = frame_shift(raw,shift[0],shift[1])

        if verbose:
            print '{}/{}: frame successfully registred'.format(i+1,n_image)        
    
        ## Loop > save
        if save:      
            ### Loop > save > Determine the last / in the filepath to deduce  
            ###               the file path and the last . to deduce the file 
            ###               name.
            index_0 = [k for k,letter in enumerate(filepath) if letter == '/']
            index_1 = [j for j,letter in enumerate(filepath) if letter =='.'] 
            
            if len(index_0) == 0:
                path = ''                
            else:
                path = fileList[0][:index_0[-1]+1]

            filename = filepath[index_0[-1]+1:index_1[-1]]
            
            ### Loop > save > If doest not exist, create the path/reg/
            ###               repository to store the processed images.
            if not os.path.exists(path+'reg/'):
                os.makedirs(path+'reg/')
            
            ### Loop > save > Create valid header
            if header:
                header_valid = create_header(headers[i])
            else:
                header_valid = None
            
            ### Loop > save > Write the fits
            output = path+'reg/'+filename+'_reg.fits' 
            write_fits(output, reg[i,:,:], header=header_valid, verbose=False)
                       
            if verbose:
                print '       {} successfully saved'.format(output) 
                print ''
    
    if save:
        output_cube = path+'reg/'+'cube'+'_reg.fits'
        write_fits(output_cube, reg, header=header_valid, verbose=False)
        
    if verbose:
        print ''
        print '{} successfully saved'.format(output_cube)
        
    if display:
        display_array_ds9(reg)
       
    if header:         
        return reg, headers
    else:
        return reg
        
# -----------------------------------------------------------------------------

def cube_crop_frames_optimized(cube, ceny, cenx, ds9_indexing=True, verbose=True, display=False, save=False, **kwargs):
    """
    Determine the optimized size of the croppable cube of frames and crop it.
    
    Parameters
    ----------
    cube : numpy.array
        The cube of frames to crop.    
    
    ceny :
        The y-coordinate of the center of the cropped cube.
    
    cenx :
        The x-coordinate of the center of the cropped cube.
        
    ds9_indexing : boolean
        If True, match 1-indexing with Python 0-indexing. Furthermore, pixel 
        coordinates in DS9 is inverted in comparison with the corresponding 
        Python array entry.
        
    verbose : boolean
        If True, informations are displayed in the shell.
        
    display : boolean
        If True, the cropped cube is displayed with DS9.
        
    save : boolean
        If True, the cropped cube is saved.
        
    
    """
    from vip.calib import cube_crop_frames
    
    crop_center = np.array([cenx,ceny])
    
    # Convert the DS9 center into Python array center
    if ds9_indexing:
        crop_center = crop_center[::-1]-1
        
    # Determine the size of the cropped cube
    n_frames = cube.shape[0]
    size_all = np.zeros(n_frames)
    
        
    # Loop: optimize the size of the frames
    for j in range(n_frames):
        cube_frame = cube[j,:,:]
    
        q = np.where(cube_frame==0)
        qr = np.sqrt((q[0]-crop_center[0])**2 + (q[1]-crop_center[1])**2)
        try:
            q_zero = qr.min()
        except ValueError:
            q_zero = np.inf    
    
        q_edge_x = np.min([np.abs(crop_center[0]-cube_frame.shape[0]),crop_center[0]])
        q_edge_y = np.min([np.abs(crop_center[1]-cube_frame.shape[1]),crop_center[1]])
        side = np.min([q_zero,q_edge_x,q_edge_y])
    
        size = 2*(side)+1
    
        if size >= cube_frame.shape[0] or size >= cube_frame.shape[1]:
            size = np.min([cube_frame.shape[0],cube_frame.shape[1]])-1        
            if size % 2 == 0:
                size -= 1
        size_all[j] = size
    
    
    # Crop it
    w = cube_crop_frames(cube,size_all.min(),crop_center[0],crop_center[1],verbose=verbose)
    
    if verbose:
        old = np.array([cube[k,:,:][crop_center[0],crop_center[1]] for k in range(n_frames)])
        new = np.array([w[k,(size_all.min()-1)//2,(size_all.min()-1)//2] for k in range(n_frames)])
        
        print ''
        print '########################################################'
        print 'For all frames, the target pixel values should be equal '
        print 'to the pixel values of the cropped frame centers        '
        print '########################################################'
        print 'Target position   |  cropped frame center  |  Difference'
        print '---------------      --------------------     ----------'
        for i in range(n_frames):
            space0 = ''.join([' ' for j in range(15-len(str(int(old[i]))))])
            space1 = ''.join([' ' for j in range(19-len(str(int(new[i]))))])
            print '{:.2f}{}|  {:.2f}{}|  {}'.format(old[i],space0,new[i],space1,old[i]-new[i])

    # save
    if save: 
        from vip.fits import write_fits
        #import os        
        ### Loop > save > Determine the last / in the filepath to deduce  
        ###               the file path and the last . to deduce the file 
        ###               name.
        #index_0 = [k for k,letter in enumerate(filepath) if letter == '/']
        #index_1 = [j for j,letter in enumerate(filepath) if letter =='.'] 
        
        #if len(index_0) == 0:
        #    path = ''                
        #else:
        #    path = fileList[0][:index_0[-1]+1]

        #filename = filepath[index_0[-1]+1:index_1[-1]]
        
        ### Loop > save > If doest not exist, create the path/reg/
        ###               repository to store the processed images.
        #if not os.path.exists(path+'reg/'):
        #    os.makedirs(path+'reg/')
        
        ### Loop > save > Create valid header
        #if header:
        #    header_valid = create_header(headers[i])
        #else:
        #    header_valid = None
        
        ### Loop > save > Write the fits
        output = kwargs.pop('filename','cube_crop.fits')#path+'reg/'+filename+'_reg.fits' 
        write_fits(output, w, header=None, verbose=False)
                   
        if verbose:
            print ''
            print '{} successfully saved'.format(output) 
            print ''
                
    # Display
    if display:
        display_array_ds9(w)    
    
    # Return
    return w
   


###############################################################################    