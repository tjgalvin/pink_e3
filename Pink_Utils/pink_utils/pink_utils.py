"""
Basic functions to help interact with PINK binaries and related objects. 

To be treated as a private module for the moment until more mature. 

TODO: Add proper and consistent documentation to classes and docstrings
TODO: Consistency for positions on images. Consider (y, x) type standard
      with origin top left. Goes for the annotation_map and SOM plotting
      code
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import struct
import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame, ICRS
from scipy.ndimage import rotate as img_rotate
from collections import defaultdict

__author__ = 'Tim Galvin'

# ------------------------------------------------------------------------------
# Image manipulation
# ------------------------------------------------------------------------------

def zoom(img, in_y, in_x):
    diff_x = in_x // 2
    diff_y = in_y // 2
    
    cen_x = img.shape[1] // 2
    cen_y = img.shape[0] // 2
    
    return img[cen_y-diff_y:cen_y+diff_y,
               cen_x-diff_x:cen_x+diff_x]

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Position SkyCoord related tasks
# ------------------------------------------------------------------------------

def estimate_position(pos: SkyCoord, dx: int, dy: int, pix_scale: u= 1*u.arcsec):
    """Calculate a RA/Dec position given a central position and offsets. 
    
    Arguments:
        pos {SkyCoord} -- Position offsets are relative from
        dx {int} -- 'RA' pixel offsets
        dy {int} -- 'Dec' pixel offsets
        pix_scale {u} -- Pixel scale. Assume square pixels. 
    """
    # Turn pixel offsets to angular
    dx *= pix_scale
    dy *= pix_scale

    # RA increases right to left
    dx = -dx
    
    new_frame = SkyOffsetFrame(origin=pos)
    new_pos = SkyCoord(lon=dx, lat=dy, frame=new_frame)
    
    return new_pos.transform_to(ICRS)

    # # Turn pixel offsets to angular
    # dx = dx * pix_scale * np.cos(pos.dec.radian)
    # dy *= pix_scale

    # # Obtain relative position. RA on sky increases right to left. 
    # # Think there is a cos(dec) term in here that needs to be handled?
    # offset_pos = SkyCoord(pos.ra - dx, pos.dec + dy)

    # return offset_pos


def great_circle_offsets(pos1: SkyCoord, pos2: SkyCoord, pix_scale: u=None):
    """Compute the great circle offsets between two points. Transform to 
    pixel offsets if a pix_scale is given. 

    TODO: Add center keyword
    TODO: Allow rectangular pixels
    
    Arguments:
        pos1 {SkyCoord} -- First position
        pos2 {SkyCoord} -- Second position
    
    Keyword Arguments:
        pix_scale {u} -- Pixel scale used to transform to pixel dimensions (default: {None})
    """

    offsets = pos1.spherical_offsets_to(pos2)

    if pix_scale is None:
        return pix_scale
    
    # In pixel space, RA increases right to left
    offsets = (-(offsets[0].to(u.arcsecond)/pix_scale.to(u.arcsecond)), 
                (offsets[1].to(u.arcsecond)/pix_scale.to(u.arcsecond)))

    return offsets

# ------------------------------------------------------------------------------

def no_ticks(ax):
    '''Disable ticks
    '''
    try:
        for a in ax:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
    except TypeError:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def header_offset(path):
    """Determine the offset required to ignore the header information
    of a PINK binary
    
    Arguments:
        path {str} -- Path to the pink binary file
    """
    with open(path, 'rb') as fd:
        offset = 0
        for line in fd:
            if not line.startswith(b'#'):
                return offset
            else:
                offset = fd.tell()
            

class heatmap:
    '''Helper to interact with a heatmap output
    '''
    def __init__(self, path):
        '''Path to the heatmap to load
        '''
        self.path = path
        self.fd = None # File descriptor to access if needed
        self.header_info = None
        self.offset = header_offset(self.path)
        self.data = self._read_data()


    @property
    def f(self):
        """Helper to open a persistent filedescriptor. Will always start
        at beginning of file
        """
        if self.fd is None:
            self.fd = open(self.path, 'rb')
        
        # Skip PINK header if it exists
        self.fd.seek(self.offset)

        return self.fd

    @property
    def file_head(self):
        '''Return the file header information from PINK
        '''
        # Get file handler seeked to zero
        f = self.f
    
        no_images, som_width, som_height, som_depth = struct.unpack('i' * 4, f.read(4*4))
        
        return (no_images, som_width, som_height, som_depth)


    @property
    def details(self):
        if self.header_info is None:
            self.header_info = self.file_head
        
        return self.header_info


    def _read_data(self):
        """Function to read in the data upon creating the heatmap
        """
        # Get file handler seeked to after the header
        f = self.f

        # no_images, som_width, som_height, som_depth = struct.unpack('i' * 4, f.read(4*4))
        no_images, som_width, som_height, som_depth = self.details

        size = som_width * som_height * som_depth
        image_width = som_width
        image_height = som_depth * som_height

        # Seek the image number here
        f.seek(4*4, 0)
        array = np.array(struct.unpack('f' * size*no_images, f.read(no_images*size * 4)))
        data = np.ndarray([no_images, som_width, som_height, som_depth], 'float', array)
        data = np.swapaxes(data, 1, 3)
        data = np.reshape(data, (no_images, image_height, image_width))

        return data


    def _ed_to_prob(self, ed, stretch=10, *args, **kwargs):
        '''Function to conver the euclidean distance to a likelihood
        '''
        prob = 1. / ed**stretch
        prob = prob / prob.sum()
        
        return prob


    def _get_ed(self, index=0, *args, **kwargs):
        '''Get the Euclidean distance of the i't page
        '''

        data = self.data[index]

        return data


    def ed(self, index=0, prob=False, *args, **kwargs):
        '''Get the slice of a heatmap that has been mapped
        '''
        arr = self._get_ed(index=index)
        if prob:
            arr = self._ed_to_prob(arr, *args, **kwargs)
        
        return arr


    def get_bmu(self, index:int):
        """Return the position of the best matching neuron for a given
        object index
        
        TODO: Support other modes (e.g. worst neuron, n-th neuron)

        Arguments:
            index {int} -- Index of source to get BMU of
        """
        hmap = self.ed(index=index)
        pos_min = np.unravel_index(np.argmin(hmap), hmap.shape)

        return pos_min

    def get_bmu_ed(self, index: int):
        """Return the ED of the best matching neuron
        
        TODO: Support other modes (e.g. worst neuron, n-th neuron)

        Arguments:
            index {int} -- Index of image to get the ED of
        """

        return np.min(self.ed(index=index))



class transform:
    '''Helper to interact with the transform output from pink
    '''
    def __init__(self, path):
        '''Path to the heatmap to load
        '''
        self.path = path
        self.fd = None # File descriptor to access if needed
        self.header_info = None
        self.offset = 0
        self.data = None

        # Skip the PINK header information
        with open(self.path, 'rb') as fd:
            for line in fd:
                if not line.startswith(b'#'):
                    self.offset == fd.tell()
                    break
            
        
    @property
    def f(self):
        """Helper to open a persistent filedescriptor. Will always start
        at beginning of file
        """
        if self.fd is None:
            self.fd = open(self.path, 'rb')
        
        # Skip PINK header if it exists
        self.fd.seek(self.offset, 0)

        return self.fd

    @property
    def file_head(self):
        '''Return the file header information from PINK
        '''
        # Get file handler seeked to zero
        f = self.f
    
        no_images, som_width, som_height, som_depth = struct.unpack('i' * 4, f.read(4*4))
        
        return (no_images, som_width, som_height, som_depth)


    @property
    def details(self):
        if self.header_info is None:
            self.header_info = self.file_head
        
        return self.header_info


    def _get_transform(self, index=0, *args, **kwargs):
        '''Get the transform of the i't page
        '''
        if self.data is None:
            # Get file handler seeked to zero
            f = self.f

            no_images, som_width, som_height, som_depth = self.details

            if no_images < index:
                return None

            size = som_width * som_height * som_depth

            # Want to avoid magic numbers
            pixel_size = np.dtype(np.int8).itemsize + np.dtype(np.float32).itemsize  
            
            # Seek the image number here, 4*4 is for the four bytes for header info
            f.seek(4*4 + self.offset, 0)
            data = np.fromfile(f, dtype = np.dtype([('flip', np.int8), 
                                                    ('angle', np.float32)]), 
                                count = no_images*size)
            data = np.reshape(data, (no_images, som_width, som_height*som_depth))
            self.data = data
        else:
            data = self.data
            
        return data[index]


        # # Get file handler seeked to zero
        # f = self.f

        # no_images, som_width, som_height, som_depth = self.details

        # if no_images < index:
        #     return None

        # size = som_width * som_height * som_depth

        # # Want to avoid magic numbers
        # pixel_size = np.dtype(np.int8).itemsize + np.dtype(np.float32).itemsize  
        
        # # Seek the image number here, 4*4 is for the four bytes for header info
        # f.seek(index * size * pixel_size + 4*4 + self.offset, 0)
        # data = np.fromfile(f, dtype = np.dtype([('flip', np.int8), 
        #                                         ('angle', np.float32)]), 
        #                       count = size)

        # return data


    def transform(self, index=0, prob=False, *args, **kwargs):
        '''Get the slice of a heatmap that has been mapped
        '''
        arr = self._get_transform(index=index)
        
        return arr


    def get_neuron_transform(self, index: int, pos: tuple):
        """Return the transform matrix for a desired source at the corresponding neuron
        position
        
        Arguments:
            index {int} -- Desired source in collection to look at
            pos {tuple} -- Neuron position the transform should be retrieved for
        """
        rot = self.transform(index=index).reshape(self.header_info[1:])
        flip, ro = rot[pos[1], pos[0]][0]    

        return flip, ro

class image_binary:
    '''Helper to interact with a heatmap output
    '''
    def __init__(self, path):
        '''Path to the image binary to load
        
        TODO: Add the PINK header consumer to the file_head and opening methods/properties
        '''
        self.path = path
        self.offset = header_offset(self.path)

    def get_image(self, index=0, channel=0):
        '''Return the index-th image that was dumped to the binary image file that
        is managed by this instance of Binary
        
        index - int
            The source image to return
        channel - int
            The channel number of the image to return
        '''
        with open(self.path, 'rb') as f:
            no_images, no_channels, width, height = struct.unpack('i' * 4, f.read(4 * 4))
            if index > no_images:
                return None
            if channel > no_channels:
                return None

            size = width * height
            f.seek(4*4 + self.offset + (index*no_channels + channel) * size*4)
            array = np.array(struct.unpack('f' * size, f.read(size*4)))
            data = np.ndarray([width,height], 'float', array)

            return data

    @property
    def file_head(self):
        '''Return the file header information from PINK
        '''
        with open(self.path, 'rb') as f:
            no_images, no_channels, width, height = struct.unpack('i' * 4, f.read(4*4))
            
            return (no_images, no_channels, width, height)


class som:
    '''Class to interact with a SOM object
    '''
    def __init__(self, path):
        self.path = path
        self.fd = None # File descriptor to access if needed
        self.header_info = None
        self.offset = 0

        # Skip the PINK header information
        with open(self.path, 'rb') as fd:
            for line in fd:
                if not line.startswith(b'#'):
                    self.offset == fd.tell()
                    break

        self.data = None                

    @property
    def f(self):
        """Helper to open a persistent filedescriptor. Will always start
        at beginning of file
        """
        if self.fd is None:
            self.fd = open(self.path, 'rb')
        
        # Skip PINK header if it exists
        self.fd.seek(self.offset)

        return self.fd


    @property
    def file_head(self):
        # Get file handler seeked to zero
        f = self.f
    
        # Unpack the header information
        numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = struct.unpack('i' * 6, f.read(4*6))

        return (numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height)

    def get_som(self, channel=0):
        '''Get out a SOM given a channel
        channel - int
              The channel number to extract
        '''
                # Get file handler seeked to zero
        som = self.f

        # Unpack the header information
        numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = struct.unpack('i' * 6, som.read(4*6))
        SOM_size = np.prod([SOM_width, SOM_height, SOM_depth])

        # Check to ensure that the request channel exists. Remeber we are comparing
        # the index
        if channel > numberOfChannels - 1:
            # print(f'Channel {channel} larger than {numberOfChannels}... Returning...')
            return None

        # Load SOM into memory
        if self.data is None:

            dataSize = numberOfChannels * SOM_size * neuron_width * neuron_height
            array = np.array(struct.unpack('f' * dataSize, som.read(dataSize * 4)))

            image_width = SOM_width * neuron_width
            image_height = SOM_depth * SOM_height * neuron_height
            data = np.ndarray([SOM_width, SOM_height, SOM_depth, numberOfChannels, neuron_width, neuron_height], 'float', array)
            data = np.swapaxes(data, 0, 5) # neuron_height, SOM_height, SOM_depth, numberOfChannels, neuron_width, SOM_width
            data = np.swapaxes(data, 0, 2) # SOM_depth, SOM_height, neuron_height, numberOfChannels, neuron_width, SOM_width
            data = np.swapaxes(data, 4, 5) # SOM_depth, SOM_height, neuron_height, numberOfChannels, SOM_width, neuron_width
            data = np.reshape(data, (image_height, numberOfChannels, image_width))

            self.data = data
        
        else:
            data = self.data

        if channel < 0 or channel is None:
            # Leave data as is and return
            pass
        else:
            data = data[:,channel,:]

        return data

    def get_neuron(self, x, y, channel=0):
        '''Extract a neuron out of the SOM given the (x,y) and optional channel number
        
        x/y - int
             The integer positions of the neuron to slice out
        channel - int
             The channel of the SOM to return
        '''
        som_spec = self.file_head
        data = self.get_som(channel=channel)
    
        d = data[y*som_spec[5]:(y+1)*som_spec[5], 
                 x*som_spec[5]:(x+1)*som_spec[5]]

        return d.copy()


class Neuron:
    """A class to help manage a single neuron from a SOM
    
    TODO: Add proper returns to documentation
    """
    def __init__(self, img: np.ndarray, pixel_scale: u = 1*u.arcsec, mask:list = None):
        """Create a new instance of the class
        
        img {numpy.ndarray} : The image of the neuron extracted
        """
        
        self.img = img
        self.pixel_scale = pixel_scale
        self.mask = mask # Expecting to apply some type of masking operation at somepoint
                         # Presumably some type of list?
        
    def apply_mask(self, img:np.ndarray=None):
        """Stub function. If a masking operation has to be applied do it here
        """
        
        if img is None:
            img = self.img.copy()
        
        if self.mask is None:
            return img

        # Assume a list of (y, x) to (y, x) as a bounding box?
        # Relative to center position
        grid = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        for m in self.mask:
            v1, v2 = m
            y1, x1 = v1
            y2, x2 = v2

            mask = ((y1 < grid[0]) & (grid[0] < y2) & (x1 < grid[1]) & (grid[1] < x2))
            img[~mask] = img[mask].min()

        return img
    
    def crop(self, size: tuple, img: np.ndarray=None):
        """Crop neuron or transform to desired size
        
        size {tuple}: size of region (y, x)
        img {np.ndarray}: The image to crop. If none use the raw neuron
        """
        if img is None:
            img = self.img
            
        y, x = img.shape
        in_y, in_x = size

        start_x = x//2-(in_x//2)
        start_y = y//2-(in_y//2)

        return img[start_y:start_y+in_y,start_x:start_x+in_x]

    def transform(self, transform: tuple, size: tuple=None, mask: bool=False):
        """Rotate the image of neuron according to the transform
        
        transform {tuple}: tuple of the form (flip, angle) with angle in radians
        size {tuple}: size of the output image to crop
        mask {bool} : apply masking operation
        """
        
        # Apply mask first, so the mask does not have to be transformed as well
        if mask:
            img_trans = self.apply_mask()
        else:
            img_trans = self.img.copy()

        flip, angle = transform
        
        img_trans = sci_rotate(img_trans, -np.rad2deg(angle), reshape=False)
        if flip == 1:
            img_trans = img_trans[:,::-1]
        
        if size is not None:
            img_trans = self.crop(size, img=img_trans)
        
        return img_trans
    
    def argmax(self, transform: tuple=None, size: np.ndarray=None, mask: bool=False):
        """Return the position of the maximum pixel. Apply transform and cropping
        if specified
        
        transform {tuple}: tuple of the form (flip, angle) with angle in radians
        pixel_scale {astropy.units}: size of the pixels. Supports square pixels only
        mask {bool} : apply masking operation
        """
        
        if transform is not None:
            img = self.transform(transform, size=size, mask=mask)
        else:
            img = self.img
        
        pos = np.unravel_index(np.argmax(img), img.shape)
        
        return pos


class Annotation():
    """Structure to save annotated positions and other information
    """
    def __init__(self, key: tuple, neurons: list):
        """Create new Annotation instance
        
        Arguments:
            key {tuple} -- Location of neuron
            neuron {list} -- list of the PINK neurons
        """
        self.key = key
        self.neurons = neurons
        self.neuron_dims = [tuple(n.shape) for n in neurons]
        self.clicks = defaultdict(list)
        self.type = None
    
    def add_click(self, channel: int, position: tuple):
        """Add a new click position to the data structure
        
        Arguments:
            channel {int} -- The channel the click was made in
            position {tuple} -- Position of the click on the image (x, y)
        """
        self.clicks[channel].append(position)

    def table_data(self):
        """Helper function to create a dictionary with information to 
        save in table form. Will not return the neurons, just the type
        and click information. 
        """
        # Remove the default factory? Not sure if really needed
        data = {f'{k}_clicks': v for k,v in self.clicks.items()}
        data['key'] = self.key
        data['type'] = self.type
        data['dims'] = self.neuron_dims

        return data

    def transform_neuron(self, transform: tuple= None, channel: int=0):
        """Given a PINK transform, apply it to save images. 
        
        Arguments:
            transform {tuple} -- [PINK transform information
        
        Keyword Arguments:
            channel {int} -- Which channel to return. (default: {0})
        """
        if transform is None:
            transform = (0,0)
        flip, angle = transform

        # Pink version < 1 transposes neurons. Fortran/C mismatch.
        img = self.neurons[channel].T

        if flip == 1:
            img = img[::-1,:]

        # Scipy ndimage rotate goes anticlockwise. PINK goes
        # clockwise
        img = img_rotate(img, np.rad2deg(-angle), reshape=False)
        
        return img

    def transform_clicks(self, transform: tuple= None, channel: int=0, center: bool=True):
        """Transform saved click information following PINK produced
        transform matix
        
        TODO: When overlaying NONTRANSFORMED clicks onto neuron they
              the ax.plot() has to reverse the x/y, i.e. ax.plot(c[1],c[0])
              This does NOT need to be done for TRANSFORMED points. Not
              sure why.

        Keyword Arguments:
            transform {tuple} -- PINK produced transform (flip, angle). (default: {None})
            channel {int} -- Channel whose clicks to return (default: {0})
            center {bool} -- Do rotation around image center (default: {True})
        """
        if transform is None:
            transform = (0,0)

        clicks = self.clicks[channel]
        neuron_dim = self.neuron_dims[channel]
        flip, angle = transform

        # Clicks stored in (x, y) format. Need to switch to (y, x)
        # for consistency
        clicks = [(c[1], c[0]) for c in clicks]

        if center:
            clicks = [(c[0] - neuron_dim[0]/2, c[1] - neuron_dim[1]/2) for c in clicks]

        trans_clicks = []
        for c in clicks:
            off_y, off_x = c

            if flip == 1:
                off_x = -off_x
                # off_y = -off_y

            trans_y = off_y*np.cos(angle) - off_x*np.sin(angle)
            trans_x = off_y*np.sin(angle) + off_x*np.cos(angle)

            trans_clicks.append( (trans_y, trans_x) )
        
        return trans_clicks




