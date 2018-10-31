"""
Basic functions to help interact with PINK binaries and related objects. 

To be treated as a private module for the moment until more mature. 

TODO: Add proper and consistent documentation to classes and docstrings
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import struct
import astropy.units as u
from scipy.ndimage import rotate as sci_rotate

__author__ = 'Tim Galvin'

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

class heatmap:
    '''Helper to interact with a heatmap output
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
            
            self.offset == fd.tell()
            

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


    def _ed_to_prob(self, ed, stretch=10, *args, **kwargs):
        '''Function to conver the euclidean distance to a likelihood
        '''
        prob = 1. / ed**stretch
        prob = prob / prob.sum()
        
        return prob


    def _get_ed(self, index=0, *args, **kwargs):
        '''Get the Euclidean distance of the i't page
        '''

        if self.data is None:
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

            self.data = data
        else:
            data = self.data

        data = data[index]

        return data

        # ------
        # Original
        # # Seek the image number here
        # # f.seek(index * size * 4, 1)
        # f.seek(index * size * 4 + 4*4, 0)
        # array = np.array(struct.unpack('f' * size, f.read(size * 4)))
        # data = np.ndarray([som_width, som_height, som_depth], 'float', array)
        # data = np.swapaxes(data, 0, 2)
        # data = np.reshape(data, (image_height, image_width))

        # return data


    def ed(self, index=0, prob=False, *args, **kwargs):
        '''Get the slice of a heatmap that has been mapped
        '''
        arr = self._get_ed(index=index)
        if prob:
            arr = self._ed_to_prob(arr, *args, **kwargs)
        
        return arr


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


class image_binary:
    '''Helper to interact with a heatmap output
    '''
    def __init__(self, path):
        '''Path to the image binary to load
        
        TODO: Add the PINK header consumer to the file_head and opening methods/properties
        '''
        self.path = path

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
            f.seek((index*no_channels + channel) * size*4, 1)
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
    
        d = data[x*som_spec[5]:(x+1)*som_spec[5], 
                 y*som_spec[5]:(y+1)*som_spec[5]]

        return d.copy()


class neuron:
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