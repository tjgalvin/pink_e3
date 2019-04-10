"""Produce a set of source at various similarity measures for a 
neuron
"""
import argparse
import numpy as np
import pink_utils as pu
import matplotlib.pyplot as plt
from itertools import product


def plot_image(imgs: pu.image_binary, idx:int, neurons: list, out: str, title: str=None):
    """Plot a specific image to a desired out path
    
    Arguments:
        imgs {pu.image_binary} -- reference to pink image binary
        idx {int} -- index of the item to plot
        neurons {list} -- iterable that contains images of the neurons
        out {str} -- file path of the save file to make
    
    Keyword Arguments:
        title {str} -- Figure title to add (default: {None})
    """
    no_chan = imgs.file_head[1]

    fig, (axes, axes2) = plt.subplots(2,no_chan, figsize=(10,10))

    for c, ax in zip(range(no_chan), axes.flat):
        img = imgs.get_image(index=idx, channel=c)

        ax.imshow(img, cmap='bwr')
        pu.no_ticks(ax)

    for c, ax in zip(range(no_chan), axes2.flat):
        img = neurons[c]

        ax.imshow(img, cmap='bwr')
        pu.no_ticks(ax)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    fig.savefig(out)
    plt.close(fig)


def perc(som: pu.som, ed: pu.heatmap, imgs:pu.image_binary, base_out:str, levels: int=10):
    """Iterate over a SOM and return a set of images at different
    percentile level of similarity
    
    Arguments:
        som {pu.som} -- instance of a loaded SOM
        ed {pu.heatmap} -- instance to the heatmap/similarity matrix
        imgs {pu.image_binary} -- reference to the image binary
        base_out {str} -- output folder to save items to 

    Keyword Arguments:
        levels {int} -- number of images at percentile levels to plot (default: {10})
    """
    som_dim = ed.file_head[1:]
    som_chan = som.file_head[0]

    x = np.arange(som_dim[0])
    y = np.arange(som_dim[1])

    data = ed.data.reshape((ed.file_head[0], np.prod(som_dim)))
    argmin = np.unravel_index(np.argmin(data, axis=1), som_dim)
    perc = np.arange(levels)

    for coord in product(x,y):
        
        # argmask = (argmin[0] == coord[0]) & (argmin[1] == coord[1])
        argmask = np.argwhere((argmin[0] == coord[0]) & (argmin[1] == coord[1]))[:, 0]

        # No matching BMU sources
        if len(argmask) == 0:
            continue

        data = ed.data[argmask, coord[0], coord[1]]

        argsort = np.argsort(data)
        perc_idx = argsort[ ((len(argsort) - 1) * (perc / len(perc))).astype(np.int) ]

        print(coord)
        print(som.file_head)
        neurons = [som.get_neuron(y=coord[0], x=coord[1], channel=i) for i in range(som_chan)]
        print(len(perc), len(perc_idx))
        
        for p, idx in zip(perc, perc_idx):
            midx = argmask[idx]
            p_level = (p / len(perc)) * 100.
            title = f"Neuron ({coord[0]}, {coord[1]}) - Top {p_level:.2f}% of distribution"

            print(f"\t {p_level:.1f}")
            plot_image(imgs, midx, neurons, f"{base_out}/neuron_{coord[0]}_{coord[1]}_{p}_{midx}.png", title=title)

    
if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Given a image binary and simiarlirt matrix, produce a '\
                                    'set of images across a range of similarities for each neuron')
    parse.add_argument('som', help='Path to a PINK SOM file', nargs=1)
    parse.add_argument('similarity', help='Path to similarity matrix/mapping binary', nargs=1)
    parse.add_argument('images', help='Path to PINK image file that matches with SIMILARITY mapping', nargs=1)
    parse.add_argument('levels', help='The number of percentile levels to plot', nargs=1, type=int)
    parse.add_argument('base_out', help='Base name for output', nargs=1)

    args = parse.parse_args()

    som = pu.som(args.som[0])
    ed = pu.heatmap(args.similarity[0])
    imgs = pu.image_binary(args.images[0])
    levels = args.levels[0]
    base_out = args.base_out[0]


    perc(som, ed, imgs, base_out, levels=levels)
