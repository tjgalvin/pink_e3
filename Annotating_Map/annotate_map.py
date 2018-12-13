"""Example code to Lasso select regions on a image and keyboard items to
perform actions

Only support two channel SOMs (for the moment)

If running on osx, run using `pythonw`. For whatever reason the terminal 
will not give up focus when using the deafult osx backend. This makes the
keyboard shortcuts useless. 
"""

import matplotlib.pyplot as plt
plt.rcParams['keymap.save'] = ''

import argparse

import numpy as np
import pink_utils as pu

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import EllipseSelector
from skimage.morphology import convex_hull_image

# https://matplotlib.org/gallery/widgets/textbox.html
# Add textbox for other information?

# http://scikit-image.org/docs/stable/api/skimage.measure.html#ellipsemodel
# Potentially `fit` a bounding ellipses to estimate centroid

msg = """Possible actions are:
'c' - Clear current mask selection
'm' - Derive a mask from a 3 sigma level clip
'h' - Perform 'm' and then apply a convex hull
's' - Save and move onto next item
'0'-'9' - Record a class label. Default is 0. 
"""

dd = {}
dl = {}

class SaveCallback():
    """Helper class to retain items from the matplotlib callbacks. Only a 
    weak reference is retained, so is modified to create a new copy (say an int)
    the reference is lost/garbage collected once matplotlib finishes
    """
    def __init__(self):
        self.SIGMA = 3


def annotate_neuron(r_n: np.ndarray, w_n: np.ndarray, key: tuple):
    """Interactive process for annotation
    
    Arguments:
        r_n {np.ndarray} -- PINK radio neuron
        w_n {np.ndarray} -- PINK IR neuron
        key {tuple} -- position of the neuron
    """
    
    print('\n')
    print(msg)

    # Structure to keep references to objects created in matplotlib callbacks
    a = SaveCallback()
    dl[key] = None

    # Items needed to be bound to
    x, y = np.meshgrid(np.arange(r_n.shape[1]), np.arange(r_n.shape[0]))
    pix = np.vstack((x.flatten(), y.flatten())).T

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))

    # Make mask
    mask = np.zeros_like(r_n)

    # Event handlers here
    def press(event):
        """Capture the keyboard pressing a button
        
        Arguments:
            event {matplotlib.backend_bases.KeyEvent} -- Keyboard item pressed
        """

        if event.key == 'c':
            mask[:,:] = 0 # Slice into, make sure bounded mask OK
            ax3.imshow(mask, cmap='bwr')
            fig.canvas.draw_idle()
        
        elif event.key == 'm':
            img_cp = np.ones_like(r_n)
            img_cp[r_n < a.SIGMA * r_n.std()] = 0
            mask[:,:] = img_cp # Slice into, make sure bounded mask OK

            ax3.imshow(mask, cmap='bwr')
            fig.canvas.draw_idle()

        elif event.key == 'h':
            rn_mask = r_n < a.SIGMA * r_n.std()
            ones = np.ones_like(r_n)
            ones[rn_mask] = 0

            ones = convex_hull_image(ones)
            mask[ones] = 1

            ax3.imshow(mask, cmap='bwr')
            fig.canvas.draw_idle()

        elif event.key in ['s', 'enter']:
            if dl[key] is not None:
                print(f"{key}: Moving on and saving.\n")
                dd[key] = mask
                plt.close(fig)
            else:
                print(f"{key}: Need to apply label before moving on")

        elif event.key == 'down':
            a.SIGMA = a.SIGMA - 1
            print(f"Sigma moved to {a.SIGMA}")

        elif event.key == 'up':
            a.SIGMA = 1 + a.SIGMA
            print(f"Sigma moved to {a.SIGMA}")

        elif event.key in ['0','1','2','3','4','5','6','7','8','9']:
            print(f"{key}: Label is set to: {event.key}")
            dl[key] = event.key

            fig.suptitle(f"{key} - Label: {event.key}")
            fig.canvas.draw_idle()

        else:
            print(f"Captured key is: {event.key}")
            print('\n',msg)


    # Callback for lasso selector
    def onselect(verts):

        # Select elements in original array bounded by selector path:
        p = Path(verts)
        ind = p.contains_points(pix, radius=1)
        
        mask.flat[ind] = r_n.flat[ind]
        ax3.imshow(mask, cmap='bwr')
        
        max_point = np.unravel_index(np.argmax(mask), mask.shape)
        # ax3.plot(max_point[1], max_point[0], 'ro', ms=10)

        fig.canvas.draw_idle()


    # Attach the widgets onto the figure
    fig.canvas.mpl_connect('key_press_event', press)
    lasso1 = LassoSelector(ax1, onselect)
    lasso2 = LassoSelector(ax2, onselect)

    ax1.imshow(r_n, cmap='bwr')
    ax2.imshow(w_n, cmap='bwr')
    ax3.imshow(np.zeros_like(r_n), cmap='bwr')

    fig.suptitle(f"{key} - Label: {dl[key]}")
    fig.tight_layout()

    plt.show()


def compact_dd(som: pu.som, y: int=None, x: int=None):
    """Flatten the masks to a structure similar to a PINK SOM. 

    Only support quad shapes and SOMs with no depth. 
    
    Arguments:
        som {pu.som} -- SOM which we have masked
        y {int} -- SOM height. None if taken from som
        x {int} -- SOM width. None if takenf from som
    """

    chan, height, width, depth, n_height, n_width = som.file_head

    if y is not None:
        height = y
    if x is not None:
        width = x
    
    mask = np.zeros((height*n_height, width*n_width))

    for y in range(height):
        for x in range(width):
            key = (y, x)

            if key in dd.keys():
                mask[y*n_height:y*n_height + n_height,
                     x*n_width :x*n_width +  n_width] = dd[key]
            else:
                print(f"{key} is missing from dd. Continuing.")


    fig, ax = plt.subplots(1,1)

    ax.imshow(mask)

    # plt.show()
    fig.savefig('example.png')

    return mask


def dump_mask(som: pu.som, mask: np.ndarray, path: str, y: int=None, x: int=None):
    """Create a SOM-like binart of the mask. Mask will always be a single channel
    
    Arguments:
        som {pu.som} -- The SOM whose header information will be used
        mask {np.ndarray} -- Flattened mask data to save
        path {str} -- Path to save the data to
        y {int} -- Height of the SOM. If None, get from som
        x {int} -- Width of the SOM. If None, get from som
    """
    import struct as st

    if y is None:
        y = som.file_head[1]
    if x is None:
        x = som.file_head[2]

    with open(path, 'wb') as of:
        of.write(st.pack('i', som.file_head[0]))
        of.write(st.pack('i', y))
        of.write(st.pack('i', x))
        of.write(st.pack('i', 1))
        of.write(st.pack('i', som.file_head[4]))
        of.write(st.pack('i', som.file_head[5]))

        np.asfortranarray(mask.astype('f')).tofile(of)


def dump_labels(labels: dict, path: str):
    """Save the label structure
    
    Arguments:
        labels {dict} -- Dict structure containing the annotated labels
        path {str} -- Path to save the dictionary to
    """
    import pickle

    pickle.dump(labels, open(path, 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Small utility to assist in the annotation of a PINK SOM. Can '\
                                                 'record user specified labels and facilitate creation of masking regions.')
    parser.add_argument('SOM', help='Path to a PINK SOM')
    parser.add_argument('output', help='Base path for labels and mask to be outputted as. File formats automatically appended. ')
    parser.add_argument('--save', '-s', action='store_false', default=True, help='Save the annotated labels and mask.')


    args = parser.parse_args()
    args = vars(args)

    som = pu.som(args['SOM'])

    chan, height, width, depth, n_height, n_width = som.file_head

    max_y = height
    max_x = width

    for y in range(max_y):
        for x in range(max_x):
            key = (y, x)

            r_n = som.get_neuron(y=y, x=x, channel=0)
            w_n = som.get_neuron(y=y, x=x, channel=1)

            annotate_neuron(r_n, w_n, key)

    mask = compact_dd(som, y=max_y, x=max_x)

    if args['save']:
        dump_mask(som, mask, f"{args['output']}_mask.bin", y=max_y, x=max_x)

        dump_labels(dl, f"{args['output']}_labels.pkl")
