# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:57:40 2015

@author: Olivier Wertz, Carlos Gonzalez Gomez, Olivier Absil, see credits.
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from vip.fits import open_fits as open_fits_vip
from vip.fits import vipDS9, write_fits
from vip.conf import timeInit, timing
from vip.calib import frame_shift
from vip.calib import cube_crop_frames

from astropy.coordinates import FK5
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.time import Time
from astropy.units import hourangle, degree 

from scipy.optimize import minimize

from os import listdir
from os.path import isfile, join, exists, basename, dirname
from os import makedirs


import warnings
warnings.filterwarnings('ignore')


__all__ = ['open_fits',
           'listing',
           'create_header',
           'extract_headers',
           'find_header_card',
           #'longestSubstringFinder',
           'master',
           'masterFlat',
           'applyFlat',
           'create_cube_from_frames',
           'load_images',
           'plot_surface',
           'moffat',
           'cone',
           'gauss2d',
           'gauss2d_sym',
           'chisquare',
           'vortex_center',
           'vortex_center_routine',
           'vortex_center_from_dust_signature',
           'timeExtract',
           'optimized_frame_size',
           'cube_crop_frames_optimized',
           'registration',
           'cube_registration',
           'precess',
           'premat',
           'get_parang',
           'get_parallactic_angles',
           'get_parallactic_angles_old']


# To preserve backward compatibility with existing code
def display_array_ds9(*args):
    """ """
    ds9 = vipDS9()
    ds9.display(*args)
    
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
        
    verbose : boolean (optional)
        If True, additional informations are displayed in the shell.        
        
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
    except: # If a *missing END card* error is raised 
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

def extract_headers(file_list):
    """
    Extract all the common header items from a fits file list and put it in a 
    new dict-type object.
    
    Parameters
    ----------
    file_list : str
        A list of all image paths.
        
    Returns
    -------
    out : dict
        {'card0' : [val01, val02, ...],
        'card1' : [val11, val12, ...]}    
    
    """
    _ , header = open_fits(file_list[0], header=True, verbose=False)    
    headers = {key : [value] for key, value in header.items()}
    
    for f in file_list[1:]:
        _ , header = open_fits(f, header=True, verbose=False)
        for key,value in header.items():
            try:
                headers[key].append(value)
            except KeyError:
                pass
    
    for key in headers.keys():
        try:
            headers[key].append(header.cards[key][-1])
        except KeyError:
            pass
    
    return headers

# -----------------------------------------------------------------------------

def find_header_card(header, card, criterion='find', info=False):
    """
    Check and return (if exists) the header card value according to the given 
    criterion.
    
    Parameters
    ----------
    header : dict
        Valid header object.        
    card : str
        The header card or part of a header card we want to extract.
    criterion : str
        Type of search:
            find: try to find all header cards which contain /card
            start: try to find all header cards which start with /card
            end: try to find all header cards which end with /card
            
    Return
    ------
    out : dict or boolean
        If there are results to returned, dict-type object. Otherwise, False.
    
    """
    if criterion == 'xfind':
        try:
            res = {card : header[card]}
        except KeyError:
            res = {}
    elif criterion == 'find':
        res = {key : value for key, value in header.items() if key.find(card) > -1}
    elif criterion == 'start':
        res = {key : value for key, value in header.items() if key.startswith(card)}
    elif criterion == 'end':
        res = {key : value for key, value in header.items() if key.endswith(card)}

    if bool(res) is False:
        res = {card : 'not found'}
        infos = {card : 'nope'}
    elif info:
        try:
            infos = {key : header.cards[key][2] for key in res.keys()}
        except AttributeError:
            infos = {key : 'nope' for key in res.keys()}

    if info:
        return res, infos
    else:
        return res


# -----------------------------------------------------------------------------
#
#def longestSubstringFinder(string1, string2):
#    """
#    Return the longest common substring between two strings.
#    
#    Parameters
#    ----------
#    string1, 2: str
#        The two strings to compare.
#        
#    Return
#    ------
#    out : str
#        The longest common substring.
#        
#    """
#    answer = ""
#    len1, len2 = len(string1), len(string2)
#    for i in range(len1):
#        match = ""
#        for j in range(len2):
#            if (i + j < len1 and string1[i + j] == string2[j]):
#                match += string2[j]
#            else:
#                if (len(match) > len(answer)): answer = match
#                match = ""
#    return answer 
    
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

    if repository.endswith('.'+ext):
        if isfile(repository):
            return [repository]
        else:
            raise IOError('File does not exist: {}'.format(repository))
    elif not repository.endswith('/'):
        repository += '/'
            
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

def master(fileList, header=False, norm=True, display=False, save=False, 
           verbose=False, full_output=True, path_output=None, filename='master', 
           filtering=None, method='median'):
    """
    Create a master image (median) from a set of single images.
    
    Parameters
    ----------
    fileList : list
        A list of all single image paths.
        
    header : boolean (optional)
        If True, the headers of each single files are returned into a dict
        file.
        
    norm : boolean (optional)
        If True, the master image is normalized.
        
    display : boolean (optional)
        If True, the master image is opened with DS9.
        
    save : boolean (optional)
        If True, the master image is saved as a fits file in the same folder as 
        the single images.
        
    Returns
    -------
    out : numpy.array
        The master image as a numpy array with the same dimension as the single
        images.
        
        If header is True, a dict is also returned as a second output object. 
        
    """
    
    # TODO: si header est True, ecrire le header dans le master image.
    if verbose: 
        start_time = timeInit()
        print 'BUILDING THE MASTER IMAGE'
        print ''
        print 'Save = {}'.format(save)
     
    # Shape and number of files
    l, c = open_fits(fileList[0]).shape    
    n_image = len(fileList)
    
    # Initializate variables
    flats = np.zeros([l,c,n_image])
    #headers = []

    # Loop: open all images and concatenate them into a bigger array
    for j in range(n_image):
        if header:
            flats[:,:,j], h = open_fits(fileList[j], header=True)
            #headers.append(h)
        else:
            flats[:,:,j] = open_fits(fileList[j], header=False)
    
    # Create the master image -> median
    if method == 'median':
        mimage = np.median(flats, axis=2)
        norm_factor = np.median(mimage)
    else:
        mimage = np.median(flats, axis=2)
        norm_factor = np.median(mimage)

    # Normalization
    if norm:
        mimage = mimage/norm_factor
        
    # filtering
    if filtering is not None:    
        mimage[np.where(mimage < np.median(mimage)-filtering*np.std(mimage))] = 1
    if verbose:
        print 'filtering = {}'.format(filtering)

    # Display
    if display:
        display_array_ds9(mimage)

    # Save    
    if save:
        if path_output is None:
            ## Save > Determine the path in which the files will be stored
            index = [k for k,letter in enumerate(fileList[0]) if letter == '/'] 
            if len(index) == 0:
                path_output = ''
            else:
                path_output = fileList[0][:index[-2]+1]

        if not exists(path_output):
            makedirs(path_output)
            
        ## Save > Write the fits             
        write_fits(join(path_output,'{}.fits'.format(filename)), mimage, verbose=verbose)        
    
    # Headers
    if header:
        headers = extract_headers(fileList)
    
    if verbose:         
        print ''
        print '-------------------------------------------------------------------'
        print 'Master image successfully created'
        timing(start_time)
    
    # Return output(s)
    if full_output:
        if header:    
            return mimage, headers
        else:
            return mimage
    else:
        return None
    

# ----------------------------------------------------------------------------- 

def masterFlat(fileList, **kwargs):
    """
    Override the former version of masterFlat by calling the function master().

    Parameters  
    ----------
    fileList : list
        A list of all single image paths.
        
    kwargs : dict-type
        Additional parameters are passed to master().
        
    Return
    ------
    out : numpy.array
        The master flat as a numpy array with the same dimension as the single
        images.
        
        If header is True, a dict is also returned as a second output object.    
    
    """
    path_output = kwargs.pop('path_output', '')    
    if path_output is None: path_output = ''    
    filename = kwargs.pop('filename','mflat')
    
    return master(fileList, path_output=join(path_output,'calibration',''), 
                  filename=filename, **kwargs)


# ----------------------------------------------------------------------------- 

def applyFlat(fileList, path_mflat, header=False, display=False, save=False, 
              verbose=False, full_output=True, path_output=''):
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
    if verbose: 
        start_time = timeInit()
        print 'PREPROCESSING IMAGES'
        print ''
        print 'Save = {}'.format(save)
        
        
    # Open the master flat
    if isinstance(path_mflat, str):
        mflat = open_fits(path_mflat, header=False)
    else:
        mflat = path_mflat

    
    # Check if *fileList* is a list and raise an error if not
    if isinstance(fileList, str):
        if fileList.endswith('/'): # If True, fileList is a repository ... 
            fileList = listing(fileList)
        else: # ... otherwise, it's a file path.
            fileList = [fileList]
    elif not isinstance(fileList, list):
        raise TypeError('fileList must be a list or a str, {} given'.format(type(fileList)))
    
    # Initializate few parameters
    #processed_all = dict()
    l, c = open_fits(fileList[0]).shape
    processed_all_cube = np.zeros([len(fileList),l,c])
    headers = dict()
    
    if save:
        if path_output is None: path_output = ''
        subrep_in_path_output = join(path_output,basename(dirname(fileList[0])) + '_flatted','')            
                    
        if not exists(subrep_in_path_output):
            makedirs(subrep_in_path_output)   

        
    # Loop: process and handle each file 
    for i, filepath in enumerate(fileList):
        ## Loop > Open file and retrieve the header if needed
        #if header:
        raw, headers[fileList[i]] = open_fits(filepath, header=True)
        #else:
        #    raw = open_fits(filepath, header=False)
        
        ## Loop > Process the image and store it 
        #processed = raw/mflat
        #processed_all[filepath] = raw/mflat        
        processed_all_cube[i,:,:] = raw/mflat

        
        ## Loop > display
        if display:
            display_array_ds9(processed_all_cube[i,:,:])
        
        ## Loop > save
        if save:      
            #filename = filepath[index_0[-1]+1:index_1[-1]]
            filename = basename(filepath).split('.')[0]

            
            ### Loop > save > Create valid header
            #if header:
            header_valid = create_header(headers[fileList[i]])
            #else:
            #    header_valid = None
            
            ### Loop > save > Write the fits
            output = subrep_in_path_output+filename+'_flatted.fits'   
            write_fits(output, processed_all_cube[i,:,:], header=header_valid, verbose=False)
            
            #if verbose:
            #    print '/flatted/{} successfully saved'.format(output[-output[::-1].find('/'):])

    if verbose: 
        if save:
            print ''
            print 'Fits files successfully created'
        print ''
        print '-------------------------------------------------------------------'
        print 'Images succesfully preprocessed'        
        timing(start_time)

    # Return the output(s)
    if full_output:
        if header:
            return processed_all_cube, extract_headers(fileList)
        else:
            return processed_all_cube
    else:
        return None




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
   
    #if header:
    first, h = open_fits(file_list[0], header=True)
    headers = [create_header(h)]
    #else:
     #   first = open_fits(file_list[0])
        
    l, c = first.shape
    
    cube = np.zeros([len(file_list), l, c])    
    cube[0,:,:] = first    
    
    if verbose:
        print 'Frame {} is added to the cube'.format(file_list[0])
    
    for k,filename in enumerate(file_list[1:]):
        #if header:
        temp, h_temp = open_fits(filename, header=True)
        #else:
        #    temp = open_fits(filename, header=False)
            
        if temp.shape != (l,c):
            print 'Each frame must have the same dimension as the first one ({},{}), {} given for {}'.format(l,c,temp.shape,filename)
            continue
        else:
            cube[k+1,:,:] = temp
        
        headers.append(h_temp)
        if verbose:
            print 'Frame {} is added to the cube'.format(filename)
                
    if save:       
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
        return cube, extract_headers(file_list)
    else:
        return cube
    

# -----------------------------------------------------------------------------
def load_images(path, header=False, verbose=False):
    """
    Load a set of images and return a cube build from the images.
    
    Parameters
    ----------
    path: str
        Path to the images to load.
    
    header : boolean (optional)
        If True, the function returns a list of all fits image headers.
        
    verbose : boolean (optional)
        If True, additional informations are displayed in the shell.
        
    Return
    ------
    out : numpy.array
        The N x l x c cube where N is the total number of frames, l x c the 
        size of each frame in pixels. If header is True, is also returns a 
        list of all headers.       
    """
    if path.endswith('.fits'):
        image = open_fits(path, header=header, verbose=verbose)
        l,c = image.shape
        temp = np.zeros([1,l,c])
        temp[0,:,:] = image
        return temp
    else:
        file_list = listing(path)
        return create_cube_from_frames(file_list, header=header, verbose=verbose)

    
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

    ds9_indexing : boolean
        If True, match 1-indexing with Python 0-indexing. Furthermore, pixel 
        coordinates in DS9 is inverted in comparison with the corresponding 
        Python array entry.        

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
        #center = (image.shape[0]//2,image.shape[1]//2)
        size = image.shape[0]
        x = np.outer(np.arange(0,size,1), np.ones(size))
        y = x.copy().T 
        z = image
        
    elif ds9_indexing:
        center = (center[0]-1,center[1]-1) 
        cy,cx = center
        if size % 2:  # if size is odd             
            x = np.outer(np.arange(0,size,1), np.ones(size))
        else: # otherwise, size is even
            x = np.outer(np.arange(0,size+1,1), np.ones(size+1))
        y = x.copy().T            
        z = image[cx-size//2:cx+size//2+1,cy-size//2:cy+size//2+1]
    
    
    plt.figure(figsize=kwargs.pop('figsize',(5,5)))
    ax = plt.axes(projection='3d')
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
    #from scipy.optimize import minimize
    
    # Create the grid of pixels
    if size % 2: #odd
        x = np.outer(np.arange(0,size,1), np.ones(size))
    else:
        x = np.outer(np.arange(0,size+1,1), np.ones(size+1))
    y = x.copy().T

    # Initializate variables    
    cy, cx = center
    
    if ds9_indexing:
        cx = cx - 1 # To match DS9 (1 -> 1024) and python (0 -> 1023) pixel indexing
        cy = cy - 1 # To match DS9 (1 -> 1024) and python (0 -> 1023) pixel indexing
    
    data = image[cx-size//2:cx+size//2+1,cy-size//2:cy+size//2+1]
    n = data.shape[0]*data.shape[1]
    
    # Display
    if display:
        plot_surface(data,figsize=kwargs.pop('figsize',(8,5)), cmap=kwargs.pop('cmap','jet'))
    
    # Start the minimization
    solu = minimize(chisquare,
                    p_initial,
                    args=(x,y,data,fun,n),
                    method = kwargs.pop('method','Nelder-Mead'),
                    options=kwargs.pop('options',{'xtol':1e-04, 'maxiter':1e+05,' maxfev':1e+05}),
                    **kwargs)    
    #print solu                    
    # Determine the absolute position of the VORTEX center when the coordinate
    # of the center of the pixel image[0,0] is [1,1] (as it is for DS9)
    if ds9_indexing:                 
        center_vortex = [cy+1+solu.x[1]-size//2, cx+1+solu.x[0]-size//2] 
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
            ax.plot(solu.x[1],solu.x[0],'+g',markersize=10, markeredgewidth=1)
            im = ax.imshow(toplot[k], vmin=absolute_vmin, vmax=absolute_vmax, interpolation='nearest', origin='lower')            
            ax.set_title(labels[k])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85,0.36,0.035,0.31])
        fig.colorbar(im, cax=cbar_ax)
        if savefig:
            fig.savefig('data_model.png')
        
        fig = plt.figure(figsize=kwargs.get('figsize',(6,5)))
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
        print 'Position of the VORTEX center (in DS9): [{:.3f},{:.3f}]'.format(center_vortex[0],center_vortex[1]) 
        print 'Relative position of the VORTEX center in the box: [{:.3f},{:.3f}]'.format(solu.x[1],solu.x[0])
        print ''
        print 'Note: The center position is given with regard to the DS9 convention,'
        print 'i.e. [1,1] corresponds to the center of the first pixel.'        
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

def vortex_center_routine(path_files, center, size, fun=gauss2d, preprocess=False, 
                          path_mflat=None, additional_parameters=[5,5], cards=None, 
                          verbose=False, **kwargs):
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
    
    if verbose: 
        start_time = timeInit()    
        print 'FIND SIGNATURE CENTER'
    
    if preprocess and path_mflat is None:
        raise ValueError('If "preprocess" is True, the path to the master flat (path_mflat) must be passed to the function.')
                
    # If required, listing all files in the main repository
    open_file = True                
    if isinstance(path_files,str):
        if path_files.endswith('/'): # If True, fileList is a repository ... 
            file_list = listing(path_files)
        else:
            file_list = [path_files]
        n = len(file_list)
    elif isinstance(path_files, list):
        file_list = path_files        
        n = len(file_list)
    else:  
        open_file = False
        if len(path_files.shape) == 3:
            file_list = path_files
            n = file_list.shape[0]
        elif len(path_files.shape) == 2:
            n = 1
            file_list = np.zeros([1,path_files.shape[0],path_files.shape[1]])
            file_list[0,:,:] = path_files
        
    
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
    
    # Optimization Options
    options = kwargs.pop('options',{'xtol':1e-04, 'maxiter':1e+05,' maxfev':1e+05})
    # Let's go !
    #for k, filename in enumerate(file_list):
    for k in range(n):
        if verbose > 1:
            print '###################################'
            if k+1 < 10:
                print 'Step {}/{}                         #'.format(k+1,n)
            elif k+1 >= 10:
                print 'Step {}/{}                        #'.format(k+1,n)
            print '###################################'
        elif verbose:
            pass#print 'Step {}/{}'.format(k+1,n)
        
        # Load and preprocess images
        #if preprocess:
        #    preprocessed = applyFlat(file_list[k],path_mflat, header=True, display=False, save=False)
        #    image_flatted, header = preprocessed[0][filename],preprocessed[1][filename]
        #else:
        if open_file:
            image_flatted, header = open_fits(file_list[k], header=True)
        else:
            image_flatted = file_list[k,:,:]
        
        # Background and I0 rought estimation
        bkg_ini = np.median(image_flatted)
        i0_ini = np.max(image_flatted[center[k][0]-size//2:center[k][0]+size//2,center[k][1]-size//2:center[k][1]+size//2]) - bkg_ini
        
        # Initial parameters
        p_initial = np.array([x_ini,y_ini,i0_ini,bkg_ini]+additional_parameters)

        # Extract the user header card(s)
        if cards is not None and open_file:
            for card in cards:
                try:
                    cards_all[card].append(header[card])
                except:
                    if verbose > 1:
                        print '{}: invalid header card or not in the header.'.format(card)
                    cards_all[card].append(None)
        else:
            cards_all = None
        
        # Minimization 
        result = vortex_center(image_flatted, 
                                  center[k], 
                                  size, 
                                  p_initial, 
                                  fun,
                                  display= False, 
                                  verbose=vc_verbose,
                                  method = 'Nelder-Mead', 
                                  options = options,
                                  **kwargs)
        
        center_all[k,:] = result[0]
        success_all[k] = result[1].success

    if verbose:
        if all(success_all):
            print ''
            print 'All numerical minimizations have converged.'
        else:
            print ''
            print 'At least one numerical minimization has not converged.'
            
        print ''
        print '-------------------------------------------------------------------'
        print 'DONE'
        timing(start_time)

    if cards is None:    
        return center_all, success_all, file_list
    else:
        return center_all, success_all, file_list, cards_all        


# -----------------------------------------------------------------------------
def vortex_center_from_dust_signature(sci, sky, dust_options, vortex_options,
                                      verbose=True, full_output=False):
    """
    Determine the VORTEX center for a set of sci images from the relative position
    of a dust with respect to the vortex in a single or set of sky images.
    
    Parameters
    ----------
    sci : numpy.array
        The sci image(s) as an array.
        
    sky : numpy.array
        The sky image(s) as an array.        
        
    dust_options : dict
        Options related to the determination od the dust position using the
        function vortex_center_routine(). The dict should have the following 
        keys:
        - center: estimation for the center of the dust signature
        - size: the size of the box in which the center is determined
        - fun: see Note, here below
        - parameters: additional parameters for fun
        
    vortex_options : int
        Options related to the determination od the vortex position using the
        function vortex_center_routine(). The dict should have the following 
        keys:
        - center: estimation for the center of the vortex signature
        - size: the size of the box in which the center is determined
        - fun: see Note, here below
        - parameters: additional parameters for fun   

    verbose : boolean (optional)
        If True, additional informations are displayed in the shell.        
      
    full_output : boolean (optional)
        If True, the vortex and dust center positions in the sky images as
        well as the relative positions are also returned.
        
    Return
    ------
    out : numpy.array
        If full_output is False: 
            - position of the vortex in the sci images
        If full_output is True:
            - position of the vortex in the sci images
            - position of the dust in the sci images
            - position of the vortex in the sky images
            - position of the dust in the sky images
   
    Note
    -----
    fun : callable
        VORTEX signature model. The Vortex_Preprocessing module already include:
        + gauss2d
        + gauss2d_sym
        + moffat
        + cone        
               
    """
    if verbose: 
        start_time = timeInit()
        print 'VORTEX CENTER FROM DUST SIGNATURE'
        print ''
    # DUST centers in sci
    center_all_dust, _, _ = vortex_center_routine(sci, dust_options['center'], 
                                                  dust_options['size'], dust_options.get('fun',gauss2d_sym), 
                                                  additional_parameters=dust_options.get('parameters',[5]), 
                                                  verbose=False)
    if verbose:
        print 'Dust center positions in sci images: Done.'
    
    # DUST center in sky
    center_dust_in_sky, _, _ = vortex_center_routine(sky, dust_options['center'], 
                                                  dust_options['size'], dust_options.get('fun',gauss2d_sym), 
                                                  additional_parameters=dust_options.get('parameters',[5]), 
                                                  verbose=False)                                                 
    if verbose:
        print 'Dust center position in sky image: Done'
        
    # VORTEX center in sky
    center_vortex_in_sky, _, _ = vortex_center_routine(sky, vortex_options['center'], 
                                                  vortex_options['size'], vortex_options.get('fun',gauss2d_sym), 
                                                  additional_parameters=vortex_options.get('parameters',[5]), 
                                                  verbose=False)
                                                 
    if verbose:
        print 'Vortex center position in sky image: Done'    
    
    # VORTEX centers in sci
    relative_position = np.mean(center_vortex_in_sky - center_dust_in_sky, axis=0)
   
    center_all = center_all_dust + np.tile(relative_position,(center_all_dust.shape[0],1))    
    
    if verbose:
        print ''
        print '-------------------------------------------------------------------'
        print 'Vortex center in sci images successfully determined.'
        timing(start_time)
   
    if full_output:
        return center_all, center_all_dust, center_vortex_in_sky, center_dust_in_sky 
    else:
        return center_all


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
        pass
        
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
            print index_0, index_1
            filename = filepath[index_0[-1]+1:index_1[-1]]
            
            ### Loop > save > If doest not exist, create the path/reg/
            ###               repository to store the processed images.
            if not exists(path+'reg/'):
                makedirs(path+'reg/')
            
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

def cube_crop_frames_optimized(cube, ceny, cenx, ds9_indexing=True, verbose=True, 
                               display=False, save=False, **kwargs):
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
    w = cube_crop_frames(cube,size_all.min(),crop_center,verbose=verbose)
    
    if verbose:
        old = np.array([cube[k,:,:][crop_center[0],crop_center[1]] for k in range(n_frames)])
        new = np.array([w[k,(size_all.min()-1)//2,(size_all.min()-1)//2] for k in range(n_frames)])
        
        print ''
        print '########################################################'
        print 'For all frames, the target pixel values should be equal '
        print 'to the pixel values of the cropped frame centers.       '
        print 'In other words, if Difference = 0 than that is ok !     '
        print '########################################################'
        print 'Target position   |  cropped frame center  |  Difference'
        print '---------------      --------------------     ----------'
        for i in range(n_frames):
            space0 = ''.join([' ' for j in range(15-len(str(int(old[i]))))])
            space1 = ''.join([' ' for j in range(19-len(str(int(new[i]))))])
            print '{:.2f}{}|  {:.2f}{}|  {}'.format(old[i],space0,new[i],space1,old[i]-new[i])

    # save
    if save: 
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


# -----------------------------------------------------------------------------

def optimized_frame_size(cube):
    """
    """
    n_frames, l, c = cube.shape
    frame_shape = np.array([l,c])
    
    center_cube = np.floor(frame_shape/2)
    size_all = np.zeros(n_frames)
    
    for j in range(n_frames):
        cube_frame = cube[j,:,:]
    
        q = np.where(cube_frame==0)
        qr = np.sqrt((q[0]-center_cube[0])**2 + (q[1]-center_cube[1])**2)
        try:
            q_zero = qr.min()
        except ValueError:
            q_zero = np.inf    
            
        q_edge_x = np.min([np.abs(center_cube[0]-cube_frame.shape[0]),center_cube[0]])
        q_edge_y = np.min([np.abs(center_cube[1]-cube_frame.shape[1]),center_cube[1]])
        side = np.min([q_zero,q_edge_x,q_edge_y])
    
        size = 2*(side)+1

        if size >= cube_frame.shape[0] or size >= cube_frame.shape[1]:
            size = np.min([cube_frame.shape[0],cube_frame.shape[1]])-1        
            if size % 2 == 0:
                size -= 1
        size_all[j] = size

    size_min = np.array(size_all).min()-1
    if size_min%2==0: size_min -= 1
    
    return size_min


# -----------------------------------------------------------------------------

def cube_registration(cube, center_all, cube_output_size=None, ds9_indexing=True,
                      save=True, bp_removal=False, verbose=True, path_output='',
                      filename='cube'):
    """
    """
    start_time = timeInit(verbose=verbose)
    if verbose: 
        print 'REGISTRATION AND CROP'
        print ''
        print 'Save = {}'.format(save)    
    n, l, c = cube.shape
    frame_shape = np.array([l,c])
    reg = np.zeros_like(cube)
    
    center_cube = np.floor(frame_shape/2) 
    
    if ds9_indexing:
        based = 1
    else:
        based = 0
                
    for i,frame in enumerate(cube):
        shift =  center_cube - (center_all[i,:]-based) 
        reg[i,:,:] = frame_shift(cube[i,:,:],shift[1],shift[0])
    
    if cube_output_size is None:
        cube_output_size = optimized_frame_size(reg)

    if cube_output_size%2==0: cube_output_size -= 1
                             
    reg_crop = cube_crop_frames(reg, cube_output_size, center_cube, 
                                verbose=False)

    if save:
        path_for_cube = join(path_output,'cube')
        if not exists(path_for_cube):
            makedirs(path_for_cube)  
        output_filename = join(path_for_cube,'{}_{}{}{}.fits'.format(filename,start_time.year,start_time.month,start_time.day))
        write_fits(output_filename, reg_crop, header=None, verbose=False)
                                
    if verbose: 
        if save:
            print ''
            print 'The following fits file has been successfully created:'
            print '{}'.format(filename)
        print ''
        print '-------------------------------------------------------------------'
        print 'Registred and croped cube successfully created'        
        timing(start_time)                                

    return reg_crop


# -----------------------------------------------------------------------------

def get_parallactic_angles(file_list, save=False, path_output=''):
    """
    Determine the true image orientation (= true parallactic angle + instrumental contributions) in the NIRC2 frames.
    
    Parameters
    ----------
    fileList : list
        A list of all image paths.

    save : boolean
        If True, the parallactic angle list is saved at a fits file.

    path_output : string
        Path to which the file has to be saved.
    """
    parallactic_angles = np.zeros(len(file_list))
    
    for k, filename in enumerate(file_list):
        _, header = open_fits(filename, header=True)

        # compute the hour angle at the middle of the frame in a pythonic way
        expstart = header['EXPSTART']
        expstop = header['EXPSTOP']
        ftr = [3600,60,1]
        exptime = sum([a*b for a,b in zip(ftr, map(float,expstop.split(':')))]) - \
                  sum([a*b for a,b in zip(ftr, map(float,expstart.split(':')))])
        ha_mid = np.radians(header['HA'] + exptime/2.*360./24./3600.)  # convert exptime/2 from seconds of time into degrees

        # precess the star coordinates to the appropriate epoch
        ra = header['RA']
        dec = header['DEC']
        coor = SkyCoord(ra=ra, dec=dec, unit=(degree,degree), frame=FK5, equinox='J2000.0')
    	obs_epoch = Time(header['DATE-OBS'], format='iso', scale='utc')
    	coor_curr = coor.transform_to(FK5(equinox=obs_epoch))

        # derive the true parallactic angle of the object at the middle of the frame
        ra = np.radians(coor_curr.ra)
        dec = np.radians(coor_curr.dec)
        lat = np.radians(19. + 49.7/60.)
        parangle = -np.degrees(np.arctan2(-np.sin(ha_mid), np.cos(dec)*np.tan(lat)-np.sin(dec)*np.cos(ha_mid)))

        # add instrumental contribution to obtain the true angle of the nirc2 frames
        parallactic_angles[k] = parangle.value + header['ROTPOSN']-header['INSTANGL'] + (header['PARANG']-header['PARANTEL'])

        # for the record, this is the Crepp version, not correct close to zenith
        #parallactic_angles[k] = header['ROTPPOSN']+header['PARANTEL']-header['EL']-header['INSTANGL']  
    
    if save:
        pa_path = join(path_output,'PA')
        if not exists(pa_path):
            makedirs(pa_path)          
        write_fits(join(pa_path,'parallactic_angles.fits'), parallactic_angles, header=None, verbose=False)
    
    return parallactic_angles

# -----------------------------------------------------------------------------

def get_parallactic_angles_old(file_list, save=False, path_output=''):
    """
    For the record, this is the previous version of the function, based on the formula used by Justin Crepp.
    """
    parallactic_angles = np.zeros(len(file_list))
    
    for k, filename in enumerate(file_list):
        _, header = open_fits(filename, header=True)
        parallactic_angles[k] = header['ROTPPOSN']+header['PARANTEL']-header['EL']-header['INSTANGL']  
    
    if save:
        if not exists(path_output+'PA/'):
            makedirs(path_output+'PA/')          
        write_fits(path_output+'PA/parallactic_angles.fits', parallactic_angles, header=None, verbose=False)
    
    return parallactic_angles
    

###############################################################################  

def precess(ra0, dec0, equinox1, equinox2, doprint=False, fk4=False, 
            radian=False):
   """
    NAME:
         PRECESS
    PURPOSE:
         Precess coordinates from EQUINOX1 to EQUINOX2.
    EXPLANATION:
         For interactive display, one can use the procedure ASTRO which calls
         PRECESS or use the /PRINT keyword.   The default (RA,DEC) system is
         FK5 based on epoch J2000.0 but FK4 based on B1950.0 is available via
         the /FK4 keyword.
   
         Use BPRECESS and JPRECESS to convert between FK4 and FK5 systems
    CALLING SEQUENCE:
         PRECESS, ra, dec, [ equinox1, equinox2, /PRINT, /FK4, /RADIAN ]
   
    INPUT - OUTPUT:
         RA - Input right ascension (scalar or vector) in DEGREES, unless the
                 /RADIAN keyword is set
         DEC - Input declination in DEGREES (scalar or vector), unless the
                 /RADIAN keyword is set
   
         The input RA and DEC are modified by PRECESS to give the
         values after precession.
   
    OPTIONAL INPUTS:
         EQUINOX1 - Original equinox of coordinates, numeric scalar.  If
                  omitted, then PRECESS will query for EQUINOX1 and EQUINOX2.
         EQUINOX2 - Equinox of precessed coordinates.
   
    OPTIONAL INPUT KEYWORDS:
         /PRINT - If this keyword is set and non-zero, then the precessed
                  coordinates are displayed at the terminal.    Cannot be used
                  with the /RADIAN keyword
         /FK4   - If this keyword is set and non-zero, the FK4 (B1950.0) system
                  will be used otherwise FK5 (J2000.0) will be used instead.
         /RADIAN - If this keyword is set and non-zero, then the input and
                  output RA and DEC vectors are in radians rather than degrees
   
    RESTRICTIONS:
          Accuracy of precession decreases for declination values near 90
          degrees.  PRECESS should not be used more than 2.5 centuries from
          2000 on the FK5 system (1950.0 on the FK4 system).
   
    EXAMPLES:
          (1) The Pole Star has J2000.0 coordinates (2h, 31m, 46.3s,
                  89d 15' 50.6"); compute its coordinates at J1985.0
   
          IDL> precess, ten(2,31,46.3)*15, ten(89,15,50.6), 2000, 1985, /PRINT
   
                  ====> 2h 16m 22.73s, 89d 11' 47.3"
   
          (2) Precess the B1950 coordinates of Eps Ind (RA = 21h 59m,33.053s,
          DEC = (-56d, 59', 33.053") to equinox B1975.
   
          IDL> ra = ten(21, 59, 33.053)*15
          IDL> dec = ten(-56, 59, 33.053)
          IDL> precess, ra, dec ,1950, 1975, /fk4
   
    PROCEDURE:
          Algorithm from Computational Spherical Astronomy by Taff (1983),
          p. 24. (FK4). FK5 constants from "Astronomical Almanac Explanatory
          Supplement 1992, page 104 Table 3.211.1.
   
    PROCEDURE CALLED:
          Function PREMAT - computes precession matrix
   
    REVISION HISTORY
          Written, Wayne Landsman, STI Corporation  August 1986
          Correct negative output RA values   February 1989
          Added /PRINT keyword      W. Landsman   November, 1991
          Provided FK5 (J2000.0)  I. Freedman   January 1994
          Precession Matrix computation now in PREMAT   W. Landsman June 1994
          Added /RADIAN keyword                         W. Landsman June 1997
          Converted to IDL V5.0   W. Landsman   September 1997
          Correct negative output RA values when /RADIAN used    March 1999
          Work for arrays, not just vectors  W. Landsman    September 2003
          Convert to Python                     Sergey Koposov  July 2010
   """
   scal = True
   if isinstance(ra0, np.ndarray):
      ra = ra0.copy()  
      dec = dec0.copy()
      scal = False
   else:
      ra=np.array([ra0])
      dec=np.array([dec0])
   npts = ra.size 
   
   if not radian:   
      ra_rad = np.deg2rad(ra)     #Convert to double precision if not already
      dec_rad = np.deg2rad(dec)
   else:   
      ra_rad = ra
      dec_rad = dec
   
   a = np.cos(dec_rad)
   
   x = np.zeros((npts, 3))
   x[:,0] = a * np.cos(ra_rad)
   x[:,1] = a * np.sin(ra_rad)
   x[:,2] = np.sin(dec_rad)
   
   # Use PREMAT function to get precession matrix from Equinox1 to Equinox2
   
   r = premat(equinox1, equinox2, fk4=fk4)
   
   x2 = np.transpose(np.dot(np.transpose(r), np.transpose(x)))      #rotate to get output direction cosines
   
   ra_rad = np.zeros(npts) + np.arctan2(x2[:,1], x2[:,0])
   dec_rad = np.zeros(npts) + np.arcsin(x2[:,2])
   
   if not radian:   
      ra = np.rad2deg(ra_rad)
      ra = ra + (ra < 0.) * 360.e0            #RA between 0 and 360 degrees
      dec = np.rad2deg(dec_rad)
   else:   
      ra = ra_rad
      dec = dec_rad
      ra = ra + (ra < 0.) * 2.0e0 * np.pi
   
   if doprint:   
      print 'Equinox (%.2f): %f,%f' % (equinox2, ra, dec)
   if scal:
      ra, dec = ra[0], dec[0]
   return ra, dec    
   


def premat(equinox1, equinox2, fk4=False):
   """
    NAME:
          PREMAT
    PURPOSE:
          Return the precession matrix needed to go from EQUINOX1 to EQUINOX2.
    EXPLANTION:
          This matrix is used by the procedures PRECESS and BARYVEL to precess
          astronomical coordinates
   
    CALLING SEQUENCE:
          matrix = PREMAT( equinox1, equinox2, [ /FK4 ] )
   
    INPUTS:
          EQUINOX1 - Original equinox of coordinates, numeric scalar.
          EQUINOX2 - Equinox of precessed coordinates.
   
    OUTPUT:
         matrix - double precision 3 x 3 precession matrix, used to precess
                  equatorial rectangular coordinates
   
    OPTIONAL INPUT KEYWORDS:
          /FK4   - If this keyword is set, the FK4 (B1950.0) system precession
                  angles are used to compute the precession matrix.   The
                  default is to use FK5 (J2000.0) precession angles
   
    EXAMPLES:
          Return the precession matrix from 1950.0 to 1975.0 in the FK4 system
   
          IDL> matrix = PREMAT( 1950.0, 1975.0, /FK4)
   
    PROCEDURE:
          FK4 constants from "Computational Spherical Astronomy" by Taff (1983),
          p. 24. (FK4). FK5 constants from "Astronomical Almanac Explanatory
          Supplement 1992, page 104 Table 3.211.1.
   
    REVISION HISTORY
          Written, Wayne Landsman, HSTX Corporation, June 1994
          Converted to IDL V5.0   W. Landsman   September 1997
   """

   deg_to_rad = np.pi / 180.0e0
   sec_to_rad = deg_to_rad / 3600.e0
   
   t = 0.001e0 * (equinox2 - equinox1)
   
   if not fk4:   
      st = 0.001e0 * (equinox1 - 2000.e0)
      #  Compute 3 rotation angles
      a = sec_to_rad * t * (23062.181e0 + st * (139.656e0 + 0.0139e0 * st) + t * (30.188e0 - 0.344e0 * st + 17.998e0 * t))
      
      b = sec_to_rad * t * t * (79.280e0 + 0.410e0 * st + 0.205e0 * t) + a
      
      c = sec_to_rad * t * (20043.109e0 - st * (85.33e0 + 0.217e0 * st) + t * (-42.665e0 - 0.217e0 * st - 41.833e0 * t))
      
   else:   
      
      st = 0.001e0 * (equinox1 - 1900.e0)
      #  Compute 3 rotation angles
      
      a = sec_to_rad * t * (23042.53e0 + st * (139.75e0 + 0.06e0 * st) + t * (30.23e0 - 0.27e0 * st + 18.0e0 * t))
      
      b = sec_to_rad * t * t * (79.27e0 + 0.66e0 * st + 0.32e0 * t) + a
      
      c = sec_to_rad * t * (20046.85e0 - st * (85.33e0 + 0.37e0 * st) + t * (-42.67e0 - 0.37e0 * st - 41.8e0 * t))
      
   
   sina = np.sin(a)
   sinb = np.sin(b)
   sinc = np.sin(c)
   cosa = np.cos(a)
   cosb = np.cos(b)
   cosc = np.cos(c)
   
   r = np.zeros((3, 3))
   r[0,:] = np.array([cosa * cosb * cosc - sina * sinb, sina * cosb + cosa * sinb * cosc, cosa * sinc])
   r[1,:] = np.array([-cosa * sinb - sina * cosb * cosc, cosa * cosb - sina * sinb * cosc, -sina * sinc])
   r[2,:] = np.array([-cosb * sinc, -sinb * sinc, cosc])
   
   return r
   
   
def get_parang(header, latitude, epoch=None):
    """
    Calculates the parallactic angle for a frame, taking coordinates and
    local sidereal time from fits-headers (frames taken in an alt-az telescope 
    with the image rotator off).
    
    The coordinates in the header are assumed to be J2000 FK5 coordinates.
    The spherical trigonometry formula for calculating the parallactic angle
    is taken from Astronomical Algorithms, Eq. (13.1) (Meeus, 1998).
    
    Parameters
    ----------
    header : dictionary
        Header of current frame.
    latitude : float
        Latitude of the observatory in degrees.
        
    Returns
    -------
    out : float
        The parallactic angle
        
    """
    
    dec0 = header['DEC']
    
    if epoch is None:
        try:
            epoch = header['EQUINOX']
            print 'Header card EQUINOX: {}'.format(epoch)
        except KeyError:
            print 'Header card EQUINOX not found. FK5 epoch was selected.'
            epoch = 'FK5'
    
    obs_epoch = Time(header['DATE-OBS'], format='iso', scale='utc')
    if epoch == 'FK5':
        _, dec = precess(header['RA'],dec0,2000,obs_epoch.decimalyear)
    elif epoch == 'todate':
        dec = dec0
    else:
        _, dec = precess(header['RA'],dec0,epoch,obs_epoch.decimalyear)
    
    dec_curr = np.deg2rad(dec)
    hour_angle = np.deg2rad(header['HA'])
    latitude = np.deg2rad(latitude)
    
    pa = -np.rad2deg(np.arctan2(-np.sin(hour_angle), np.cos(dec_curr) * \
    np.tan(latitude) - np.sin(dec_curr) * np.cos(hour_angle))) 
    
    return pa  