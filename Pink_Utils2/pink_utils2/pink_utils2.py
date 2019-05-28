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
import struct as st
import astropy.units as u
import networkx as nx
from astropy.coordinates import SkyCoord, SkyOffsetFrame, ICRS
from scipy.ndimage import rotate as img_rotate
from collections import defaultdict
__author__ = 'Tim Galvin'


def header_offset(path):
    """Determine the offset required to ignore the header information
    of a PINK binary. The header format spec: lines with a '#' start are
    ignored until a '# END OF HEADER' is found. Only valid at beginning of
    file and 
    
    Arguments:
        path {str} -- Path to the pink binary file
    """
    with open(path, 'rb') as fd:
        for line in fd:
            if not line.startswith(b'#'):
                return 0
            elif line == b'# END OF HEADER':
                return fd.tell()
            

class Heatmap():
    '''Helper to interact with a heatmap output
    '''
    def __init__(self, path):
        '''Path to the heatmap to load
        '''
        self.path = path
        self.offset = header_offset(self.path)

        with open(self.path, 'rb') as of:
            of.seek(self.offset)
            self.header_info = st.unpack('iiii', of.read(4*4))

        self.data = np.memmap(self.path, dtype=np.float32, offset=self.offset+4*4, 
                    order='C', shape=self.header_info)


    def __repr__(self):
        """Print the class
        """
        return f"Heatmap file of {self.header_info} shape with offset {self.offset}"


    @property
    def file_head(self):
        '''Return the file header information from PINK, here for consistency
        '''
        return self.header_info


    def ed(self, index=0, *args, squeeze=True, **kwargs):
        '''Get the slice of a heatmap that has been mapped
        '''
        arr = self.data[index]

        if squeeze:
            arr = np.squeeze(self.data[index])

        return arr


    def bmu_pos(self, index:int):
        """Return the position of the best matching neuron for a given
        object index

        Arguments:
            index {int} -- Index of source to get BMU of
        """
        hmap = self.ed(index=index, squeeze=False)
        pos_min = np.unravel_index(np.argmin(hmap.reshape(-1)), hmap.shape)

        return pos_min


class Transform():
    '''Helper to interact with the transform output from pink
    '''
    def __init__(self, path):
        '''Path to the Tranform to load
        '''
        self.path = path
        self.offset = header_offset(self.path)

        with open(self.path, 'rb') as of:
            of.seek(self.offset)
            self.header_info = st.unpack('iiii', of.read(4*4))

        self.data = np.memmap(self.path, dtype=np.dtype([('flip', np.int8),('angle', np.float32)]), 
                    offset=self.offset+4*4, order='C', shape=self.header_info)
            
        
    def __repr__(self):
        """Pretty print the class
        """
        return f"Transform file of {self.header_info} shape with offset {self.offset}"


    @property
    def file_head(self):
        '''Return the file header information from PINK
        '''
        return self.header_info


    def transform(self, index: int=0, *args, **kwargs):
        '''Get the slice of a transform that has been mapped
        '''
        arr = np.squeeze(self.data[index])
        
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


class Images():
    '''Helper to interact with a image binary
    '''
    def __init__(self, path):
        '''Path to the image binary to load
        '''
        self.path = path
        self.offset = header_offset(self.path)

        with open(self.path, 'rb') as of:
            of.seek(self.offset)
            self.header_info = st.unpack('i' * 4, of.read(4 * 4))

        self.data = np.memmap(self.path, dtype=np.float32, offset=self.offset+4*4, order='C',
                    shape=self.header_info)


    def __repr__(self):
        """Pretty print the class
        """
        return f"Image binary file of {self.header_info} shape with offset {self.offset}"


    def get_image(self, index: int=0, channel: int=0, transform: tuple=None, **kwargs):
        """Interface to get an image/channel and optionally apply transform
        
        Keyword Arguments:
            index {int} -- index of image (default: {0})
            channel {int} -- default channel of image (default: {0})
            transform {tuple} -- transform to apply (default: {None})
        
        Returns:
            np.ndarray -- Image with transform applied
        """
        img = self.data[index, channel]

        if transform is not None:
            img = rotate_src_image(img, transform, **kwargs)

        return img


    def get_index_dump(self, index: int):
        """Reterive the raw floats from an image binary for the given index. 
        
        Arguments:
            index {int} -- Index of image data to copy
        """
        return self.data[index,:,:,:].flatten()


    @property
    def file_head(self):
        '''Return the file header information from PINK
        '''
        return self.header_info


class SOM():
    '''Class to interact with a SOM object. 

    TODO: Move towards the memmap interface. Be care though as a number of reshapes/reorders
    have to be used to actually make it work, and these no longer map the file. Actually reads
    it in. 
    '''
    def __init__(self, path):
        self.path = path
        self.fd = None # File descriptor to access if needed
        self.header_info = None
        self.offset = header_offset(self.path)

        self.data = None                


    def __repr__(self):
        """Pretty print the class
        """
        return f"SOM file of {self.header_info} shape with offset {self.offset}"


    @property
    def f(self):
        """Helper to open a persistent filedescriptor. Will always start
        at beginning of file
        """
        # if self.fd is None:
        #     self.fd = open(self.path, 'rb')
        
        fd = open(self.path, 'rb')

        # Skip PINK header if it exists
        fd.seek(self.offset)

        return fd


    @property
    def file_head(self):
        # Get file handler seeked to zero
        f = self.f
    
        # Unpack the header information
        numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = st.unpack('i' * 6, f.read(4*6))

        return (numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height)

    def get_som(self, channel=0):
        '''Get out a SOM given a channel
        channel - int
              The channel number to extract
        '''
                # Get file handler seeked to zero
        som = self.f

        # Unpack the header information
        numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = st.unpack('i' * 6, som.read(4*6))
        SOM_size = np.prod([SOM_width, SOM_height, SOM_depth])

        # Check to ensure that the request channel exists. Remeber we are comparing
        # the index
        if channel > numberOfChannels - 1:
            # print(f'Channel {channel} larger than {numberOfChannels}... Returning...')
            return None

        # Load SOM into memory
        if self.data is None:

            dataSize = numberOfChannels * SOM_size * neuron_width * neuron_height
            array = np.array(st.unpack('f' * dataSize, som.read(dataSize * 4)))

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
        # Pink version < 1 transposes neurons. Fortran/C mismatch.
        img = self.neurons[channel].T

        if transform is None:
            return img

        flip, angle = transform

        if flip == 1:
            img = img[::-1,:]

        # Scipy ndimage rotate goes anticlockwise. PINK goes
        # clockwise
        img = img_rotate(img, np.rad2deg(-angle), reshape=False)
        
        return img

    def transform_clicks(self, transform: tuple= None, channel: int=0, center: bool=True, 
                        order: bool=False):
        """Transform saved click information following PINK produced
        transform matix
        
        TODO: When overlaying NONTRANSFORMED clicks onto neuron they
              the ax.plot() has to reverse the x/y, i.e. ax.plot(c[1],c[0])
              This does NOT need to be done for TRANSFORMED points. Not
              sure why.

        Keyword Arguments:
            transform {tuple} -- PINK produced transform (flip, angle). (default: {None})
            channel {int} -- Channel whose clicks to return (default: {0})
            center {bool} -- Do rotation around image center. Can also specify tuple of dimensions. (default: {True})
            order {bool} -- Order the clicks from closest to centre to furthest (default: {False})
        """

        clicks = self.clicks[channel].copy()
        neuron_dim = self.neuron_dims[channel]

        # Clicks stored in (x, y) format. Need to switch to (y, x)
        # for consistency
        clicks = [(c[1], c[0]) for c in clicks]

        if order:
            clicks = sorted(clicks, key=lambda c: np.sqrt((c[0] - neuron_dim[0]/2)**2. + (c[1] - neuron_dim[1]/2)**2. ))
            
        if center:
            clicks = [(c[0] - neuron_dim[0]/2, c[1] - neuron_dim[1]/2) for c in clicks]

        if transform is None:
            return clicks

        flip, angle = transform

        trans_clicks = []
        for c in clicks:
            off_y, off_x = c

            if flip == 1:
                off_x = -off_x
                # off_y = -off_y

            trans_y = off_y*np.cos(angle) - off_x*np.sin(angle)
            trans_x = off_y*np.sin(angle) + off_x*np.cos(angle)

            trans_clicks.append( (trans_y, trans_x) )
        
        if center:
            try:
                trans_clicks = [(c[0] + center[0]/2, c[1] + center[1]/2) for c in trans_clicks]
            except:
                pass

        return trans_clicks

# ------------------------------------------------------------------------------
# Transformation functions (for images)
# ------------------------------------------------------------------------------

def rotate_src_image(img: np.ndarray, transform: tuple, *args, reshape: bool=False, **kwargs):
    """Transform an input source image following some PINK transform. This aligns the 
    source image onto features within a neuron. 
    
    Arguments:
        img {np.ndarray} -- Image to transform
        transform {tuple} -- Transform option from PINK

    Keyword Arguments:
        reshape {bool} -- Reshape the image when rotating (Default: {False})
    
    Returns:
        np.ndarray -- The transformed image
    """
    flip, angle = transform

    img = img.T
    
    # Scipy ndimage rotate goes anticlockwise. PINK goes
    # clockwise
    img = img_rotate(img, np.rad2deg(-angle), reshape=reshape)

    if flip == 1:
        img = img[:,::-1]

    return img
    
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


def connvis_graph(ed: Heatmap, min_edge: float=0.0, full: bool=False, log: bool=False):
    """Produce a graph based on the CONNvis method
    
    Arguments:
        ed {pu.heatmap} -- Instance linking to a PINK mapping

    Keyword Arguments:
        min_edge {float} -- Minimum weighted value to retain an edge {Default: 0.0}
        full {bool} -- If True, use the entire set of ordered neurons when building links or just the
                       first and second (deafult: {False})
        log {bool} -- Log the link edges (default: {False})
    """
    data = ed.data
    data = data.reshape((data.shape[0], np.prod(data.shape[1:])))

    sort = np.argsort(data, axis=1)
    links = defaultdict(float)

    for i in sort[:]:
        if full:
            for x, y in zip(i[:-1], i[1:]):
            # Order of a link do not matter
                if y > x:
                    x, y = y, x
                links[(x,y)] += 1.
        else:
            x = i[0]
            y = i[1]
            # Order of a link do not matter
            if y > x:
                x, y = y, x
            links[(x,y)] += 1.

    if log:
        for k, v in links.items():
            links[k] = np.log10(v)

    max_val = max(links.values())

    G = nx.Graph()
    
    for k, v in links.items():
        val = v/max_val
        if val > min_edge:
            G.add_edge(k[0], k[1], weight=1.01 - val)#, dist=val)
        else:
            G.add_nodes_from(k)   
 
    return G