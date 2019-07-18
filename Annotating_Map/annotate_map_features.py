"""Script to interactively annotate features on SOM neurons produced
by PINK. Should be agnostic to the file format of the PINK files if the
pink_utils is updated correctly. TYhis has not been done yet. 
"""

import matplotlib.pyplot as plt
plt.rcParams['keymap.save'] = ''

import pickle
import argparse
import pandas as pd
import numpy as np
import pink_utils2 as pu

from pink_utils import Annotation
from itertools import combinations
from collections import defaultdict
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import EllipseSelector
from skimage.morphology import convex_hull_image

msg = """Possible actions are:
'n' - Move to next neuron
'b' - Move back to previous neuron
'c' - Clear current state of clicks
'q' - Exit program
'0-9' - Assign a type to neuron
"""

marker_style = ['ro', 'g*', 'yv']

# ------------------------------------------------------------------------------------------------
# Interactive matplotlib functions and callables
# ------------------------------------------------------------------------------------------------

class Callback():
    """Helper class to retain items from the matplotlib callbacks. Only a 
    weak reference is retained, so is modified to create a new copy (say an int)
    the reference is lost/garbage collected once matplotlib finishes
    """
    def __init__(self):
        self.next_move = None


def overlay_clicks(results: Annotation, mask_ax: plt.Axes):
    """Plot the markers from an Annotation instance onto the mask axes
    
    Arguments:
        results {Annotation} -- Container of current click information
        mask_ax {matplotlib.Axes} -- Instance of the axes panel
    """
    for chan in results.clicks:
        marker = marker_style[chan]
        points = results.clicks[chan]

        for p in points:
            mask_ax.plot(p[0], p[1], marker, ms=12)


def make_fig1_callbacks(callback: Callback, results: Annotation, fig1: plt.Figure, axes: plt.Axes):
    """Create the function handlers to pass over to the plt backends
    
    Arguments:
        callback {Callback} -- Class to have hard references for communication between backend and code
        results {Annotation} -- Store annotationg information for each of the neurons
        fig1 {plt.Figure} -- Instance to plt figure to plot to
        axes {plt.Axes} -- List of active axes objectd on figure. The last is taken to be the mask
    
    Returns:
        callables -- Return the functions to handle figure key press and button press events
    """

    mask_ax = axes.flat[-1]
    
    def fig1_press(event):
        """Capture the keyboard pressing a button
        
        Arguments:
            event {matplotlib.backend_bases.KeyEvent} -- Keyboard item pressed
        """
        if event.key == 'n':
            if results.type is not None:
                print("Moving to next neuron")
                callback.next_move = 'next'
                plt.close(fig1)
            else:
                print('Ensure type is set')

        if event.key == 'b':
            if results.type is not None:
                print("Moving back to previous neuron")
                callback.next_move = 'back'
                plt.close(fig1)
            else:
                print('Ensure type is set')
        
        elif event.key == 'c':
            print('Clearing clicks')
            results.clicks = defaultdict(list)
    
            mask_im = np.zeros_like(results.neurons[0]) # Will always be at least 1 neuron

            mask_ax.clear() # Clears axes limits
            mask_ax.imshow(mask_im)

            overlay_clicks(results, mask_ax)

            fig1.canvas.draw_idle()
        
        elif event.key == 'q':
            print('Exiting...')
            callback.next_move = 'quit'
            plt.close(fig1)

        elif event.key in ['0','1','2','3','4','5','6','7','8','9']:
            results.type = event.key

            fig1.suptitle(f"{results.key} - Label: {results.type}")
            fig1.canvas.draw_idle()


    def fig1_button(event):
        """Capture the mouse button press
        
        Arguments:
            event {matplotlib.backend_bases.Evenet} -- Item for mouse button press
        """
        if fig1.canvas.manager.toolbar.mode != '':
            print(f'Toolbar mode is {fig1.canvas.manager.toolbar.mode}')
            return

        if event.xdata != None and event.ydata != None and \
           event.inaxes != mask_ax:
            
            index = np.argwhere(axes.flat == event.inaxes)[0,0]
            results.add_click(index, (event.xdata, event.ydata))
            overlay_clicks(results, mask_ax)

            for ax in axes.flat:
                if ax != mask_ax:
                    ax.plot(event.xdata, event.ydata, 'go', ms=12)

        fig1.canvas.draw_idle()


    return fig1_press, fig1_button


def create_mask(neurons: list):
    """Helper function to create an empty mask 
    
    Arguments:
        neurons {list} -- List of the PINK neurons
    """

    return np.zeros_like(neurons[0])


def annotate_neuron(neurons, key: tuple, cmap: str=None, results: Annotation=None,
                    figsize=(12,4)):
    """Annotate a set of neurons
    
    Arguments:
        neurons {list} -- List of numpy arrays of the PINK SOM neurons
        key {tuple} -- Location of the neuron
    
    Keyword Arguments:
        cmap {str} -- Colour mapping to use (default: {None})
        results {Annotation} -- Existing annotation overlays (default: {None})
        figsize {tuple} -- Figure size to plot with (default: {(12,4)})

    Returns:
        Callable, Annotation -- Any callback and marked annotation information
    """

    if cmap is None:
        cmap = 'bwr'

    mask    = create_mask(neurons)

    print('\n')
    print(msg)

    fig1, axes = plt.subplots(1, len(neurons)+1, figsize=figsize, 
                                sharex=True, sharey=True) # Display the neurons
    mask_ax = axes.flat[-1]

    fig1_callback = Callback()

    if results is None:
        results = Annotation(key, neurons)

    fig1_key, fig1_button = make_fig1_callbacks(fig1_callback, results, fig1, axes)
    fig1.canvas.mpl_connect('key_press_event', fig1_key)
    fig1.canvas.mpl_connect('button_press_event', fig1_button)

    for n, ax in zip(neurons, axes.flat):
        ax.imshow(np.sqrt(n), cmap=cmap)
        # ax.axvline(n.shape[1]/2 - 1, color='black')
        # ax.axhline(n.shape[0]/2 - 1, color='black')

    mask_ax.imshow(mask)
    mask_ax.set(title='Masking ')

    fig1.suptitle(f"{key}")

    fig1.tight_layout()

    plt.show()

    return fig1_callback, results


def perform_annotations(som: str, save: str=False):
    """Perform the annotation of the neurons contained in the provided SOM
    
    Arguments:
        som {str} -- Path to the SOM to annotate
        save {bool} -- Base path to save items to. If False, no saving of items. (default: {False})
    """
    som = pu.SOM(som)

    chan, height, width, depth, n_height, n_width = som.file_head

    max_z = depth
    max_y = height
    max_x = width

    combos = [(y,x,z) for x in range(max_x) for y in range(max_y) for z in range(max_z)]

    position = 0
    annotations = {}

    while position < len(combos):
        key = combos[position]
        x, y, z = key

        # Note that the depth is ignored here
        neurons = [som.get_neuron(y=y, x=x, channel=c) for c in range(chan)]

        callback, results = annotate_neuron(neurons, key)

        if callback.next_move == 'next':
            position += 1
            annotations[key] = results
        
        elif callback.next_move == 'back':
            annotations[key] = results
            if position >= 1:
                position -= 1
            else:
                print('Can\'t move back. Position zero.')
        
        elif callback.next_move == 'quit':
            break

        if save != False:
            print('Saving the annotations...')
            save_annotations_table(annotations,  f"{save}-table.csv")
            save_annotations_pickle(annotations, f"{save}-table.pkl")


    if save != False:
        save_annotations_table(annotations,  f"{save}-table.csv")
        save_annotations_pickle(annotations, f"{save}-table.pkl")

# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Saving utilities for the annotated features and their class instance
# ------------------------------------------------------------------------------------------------
def update_annotation(som: str, save: str, key: tuple):
    """Will allow for a single neuron to be updated given an existing
    pickled annotation set
    
    Arguments:
        som {str} -- Path to the SOM file with the neuron to annotate
        save {str} -- Path to existing annotated pickle set
        key {tuple} -- Key to load in and replace annotations
    """
    som = pu.SOM(som)

    chan, height, width, depth, n_height, n_width = som.file_head

    with open(f"{save}", "rb") as of:
        annotations = pickle.load(of)

    x, y, z = key

    # Note that the depth is ignored here
    neurons = [som.get_neuron(y=y, x=x, channel=c) for c in range(chan)]

    callback, results = annotate_neuron(neurons, key)

    if callback.next_move == 'next':
        annotations[key] = results
    
    save_annotations_table(annotations,  f"{save}".replace('pkl','csv'))
    save_annotations_pickle(annotations, f"{save}")

# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Saving utilities for the annotated features and their class instance
# ------------------------------------------------------------------------------------------------

def save_annotations_table(annotations: dict, outfile: str):
    """Save a table of the annotated features to write to disk
    
    Arguments:
        annotations {dict} -- Dictionary whose values are instances of Annotation
        outfile {str} -- Location to write the pandas dataframe to
    """

    vals = [a.table_data() for a in annotations.values()]

    df = pd.DataFrame(vals)

    df.to_csv(outfile)


def save_annotations_pickle(annotations: dict, outfile: str):
    """Save the annotations dictionary as a pickle object
    
    Arguments:
        annotations {dict} -- Collections of annotated neurons
        outfile {str} -- Pickle file to create
    """
    with open(outfile, 'wb') as pkl:
        pickle.dump(annotations, pkl)

# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# View existing features
# ------------------------------------------------------------------------------------------------
def place_clicks(ax: plt.Axes, clicks: list):
    """Place existing clicks onto axes
    
    Arguments:
        ax {plt.Axes} -- Plot with image on it
        clicks {list} -- List of clicks to place
    """
    ax.plot([c[0] for c in clicks],[c[1] for c in clicks], 'go', ms=12)


def plot_neuron_features(neuron: Annotation):
    """Plot a single neuron and its overlaid components
    
    Arguments:
        neuron {Annotation} -- Annotated features of a neuron
    """
    fig, axes = plt.subplots(1, len(neuron.neurons))

    for chan, (n, ax) in enumerate(zip(neuron.neurons, axes.flat)):
        ax.imshow(n, cmap='bwr')
        place_clicks(ax, neuron.clicks[chan])

    fig.suptitle(neuron.key)
    fig.tight_layout()
    plt.show()


def visualise_annotations(annotations: str):
    """Visualise neurons which have already been annotated. The path will load in
    the previously pickled object
    
    Arguments:
        annotations {str} -- Path to pickled Annotations set
    """
    with open(annotations, 'rb') as load:
        annotations = pickle.load(load)

    for k, v in annotations.items():
        plot_neuron_features(annotations[k])

# ------------------------------------------------------------------------------------------------



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Small utility to assist in the annotation of a PINK SOM. Can '\
                                                 'record user specified labels and facilitate creation of masking regions.')
    
    group1 = parser.add_argument_group(title='Performing annotation')
    # group1.add_argument('SOM', help='Path to a PINK SOM')
    group1.add_argument('--save', '-s', nargs=1, default=False, help='Save the annotated labels and mask using this as the base path.')
    group1.add_argument('--annotate', '-a', default=False, nargs=1, help='Annotate neurons from SOM provided')
    group1.add_argument('--key','-k', default=None, nargs=1, help='If a single neuron needs to be updated. Requires the save option to point to a pickled instance of a previous annotation set. ')

    group2 = parser.add_argument_group(title='Visualise annotated neurons')
    group2.add_argument('--visualise', '-v', nargs=1, default=False,help='Visualise existing annotated neurons')

    args = parser.parse_args()
    args = vars(args)

    print(args)

    if args['annotate'] != False:
        if args['key'][0] is not None and args['save'][0] is not False:
            from ast import literal_eval

            key = literal_eval(args['key'][0]) 
            update_annotation(args['annotate'][0], args['save'][0], key)
        else:
            perform_annotations(args['annotate'][0], save=args['save'][0])

    if args['visualise'] != False:
        visualise_annotations(args['visualise'][0])