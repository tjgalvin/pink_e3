"""Produce a set of source at various similarity measures for a 
neuron
"""
import argparse
import numpy as np
import pink_utils as pu
import matplotlib.pyplot as plt
from itertools import product


def plot_image(imgs: pu.image_binary, idx:int, out: str):
    """Plot a specific image to a desired out path
    
    Arguments:
        imgs {pu.image_binary} -- reference to pink image binary
        idx {int} -- index of the item to plot
        out {str} -- file path of the save file to make
    """
    no_chan = imgs.file_head[1]

    fig, axes = plt.subplots(1,no_chan)

    for c, ax in zip(range(no_chan), axes.flat):
        img = imgs.get_image(index=idx, channel=c)

        ax.imshow(img, cmap='bwr')
        pu.no_ticks(ax)

    fig.tight_layout()
    fig.savefig(out)


def perc(som: pu.som, ed: pu.heatmap, imgs:pu.image_binary, levels: int=10):
    """Iterate over a SOM and return a set of images at different
    percentile level of similarity
    
    Arguments:
        som {pu.som} -- instance of a loaded SOM
        ed {pu.heatmap} -- instance to the heatmap/similarity matrix
        imgs {pu.image_binary} -- reference to the image binary
    
    Keyword Arguments:
        levels {int} -- number of images at percentile levels to plot (default: {10})
    """
    som_dim = ed.file_head[1:]

    x = np.arange(som_dim[0])
    y = np.arange(som_dim[1])

    data = ed.data.reshape((ed.file_head[0], np.prod(som_dim)))
    argmin = np.unravel_index(np.argmin(data, axis=1), som_dim)
    perc = np.arange(levels)

    for coord in product(x,y):
        
        argmask = (argmin[0] == coord[0]) & (argmin[1] == coord[1])
        argmask = np.argwhere((argmin[0] == coord[0]) & (argmin[1] == coord[1]))[:, 0]

        data = ed.data[argmask, coord[0], coord[1]]

        argsort = np.argsort(data)
        perc_idx = argsort[ ((len(argsort) - 1) * (perc / len(perc))).astype(np.int) ]

        for idx in perc_idx:
            plot_image(imgs, idx, f"test{idx}.pdf")

    


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Given a image binary and simiarlirt matrix, produce a '\
                                    'set of images across a range of similarities for each neuron')
    parse.add_argument('som', help='Path to a PINK SOM file', nargs=1)
    parse.add_argument('similarity', help='Path to similarity matrix/mapping binary', nargs=1)
    parse.add_argument('images', help='Path to PINK image file that matches with SIMILARITY mapping', nargs=1)
    parse.add_argument('levels', help='The number of percentile levels to plot', nargs=1)
    parse.add_argument('base_out', help='Base name for output', nargs=1)

    args = parse.parse_args()

    som = pu.som(args.som[0])
    ed = pu.heatmap(args.similarity[0])
    imgs = pu.image_binary(args.images[0])
    levels = float(args.levels[0])
    base = args.base_out[0]


    perc(som, ed, images, levels=levels)
