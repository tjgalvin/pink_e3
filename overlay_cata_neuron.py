"""Accept a trained SOM, a similarity binary, transform binary and catalogue
to overlay sources from the catalog onto each neuron
"""
import pink_utils as pu
import numpy as np 
import matplotlib.pyplot as plt
import astropy.units as u
from argparse import ArgumentParser
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

def plot_overlay(img: np.ndarray, pos_pix: np.ndarray, neuron: tuple, 
                out: str='./', mask: np.ndarray=None):
    """Create a figure to overlay onto
    
    Arguments:
        img {np.ndarray} -- neuron image
        pos_pix {np.ndarray} -- offset positions of sources
        neuron {tuple} -- key to index of neuron on the SOM
    
    Keyword Arguments:
        out {str} -- Output base name to save figures to 
        mask {np.ndarray} -- Which objects were matched to this neuron (Default: {None})
    """
    fig, (ax, ax2) = plt.subplots(1,2, figsize=(7.5,5))

    kde_factor = ((3.5*u.arcmin.to(u.arcsecond))/(1.5*u.arcsecond)/30).value


    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.annotate('(a)', xy=(0.05,0.9), xycoords='axes fraction', bbox=bbox_props)
    
    ax.imshow(np.sqrt(som.get_neuron(channel=0, y=neuron[0], x=neuron[1])), cmap='bwr',origin='lower left')
    ax.set(title=f"N = {mask.shape}, Matches = {pos_pix.shape}")

    divider = make_axes_locatable(ax2)

    axHistx = divider.append_axes('top', size='23%', pad='7%', sharex=ax2)
    axHisty = divider.append_axes('right', size='23%', pad='7%', sharey=ax2)
    cax = divider.append_axes('bottom', size='5%', pad='0%')
    ax2.annotate('(b)', xy=(0.05,0.9), xycoords='axes fraction', bbox=bbox_props)

    # no labels
    plt.setp(axHistx.get_xticklabels(), visible=False)
    plt.setp(axHisty.get_yticklabels(), visible=False)

    im = ax2.hexbin(pos_pix[:,1], pos_pix[:,0], gridsize=(50,50), bins='log', mincnt=1)
    # ax2.scatter(pix_pos[:,1], pix_pos[:,0], marker='o')
    axHistx.hist(pos_pix[:,1], bins=50, density=True)
    axHisty.hist(pos_pix[:,0], bins=50, density=True, orientation='horizontal')
    axHistx.set(ylabel='Density')
    axHisty.set(xlabel='Density')

    # KDE 1, needs density in the hist call
    xx = np.linspace(pos_pix[:,1].min(), pos_pix[:,1].max(),1000)
    kdex = gaussian_kde(pos_pix[:,1], bw_method=kde_factor / np.std(pos_pix[:,1], ddof=1))
    axHistx.plot(xx, kdex(xx), color='black')

    # KDE 2, needs density in the hist call
    yy = np.linspace(pos_pix[:,0].min(), pos_pix[:,0].max(),1000)
    kdey = gaussian_kde(pos_pix[:,0], bw_method=kde_factor / np.std(pos_pix[:,1], ddof=1))
    axHisty.plot(kdey(yy), yy, color='black')
    
    plt_scale = ((3.5*u.arcmin.to(u.arcsecond))/(1.5*u.arcsecond)/2).value

    ax2.set_xlim([-plt_scale, plt_scale])
    ax2.set_ylim([-plt_scale, plt_scale])

    pu.no_ticks([ax, ax2])

    fig.colorbar(im, cax=cax, orientation='horizontal', label='Counts')
    fig.tight_layout()

    f = f"{out}/prob_neuron_{'_'.join([str(i) for i in neuron])}.png"
    print(f'\tSaving {f}')
    fig.savefig(f"{f}")
    plt.close(fig)


def apply_transform(trans_info: tuple, offsets: np.ndarray):
    """Transform a set of pixel coordinates following a transform matrix
    
    NOTE: Not entirely convinced this is correct. I think I am messing up the
    proper transform by either (1) splitting/concating the PINK binary files
    to speed up processing [there is essentially an extra transpose from F<->C
    style notation], and/or (2) setting origin to lower left in the imshow().
    Be aware that this could be wrong.  

    Arguments:
        trans_info {tuple} -- tuple from pink (FLIP, RADIANS)
        offsets {np.ndarray} -- Pixel offsets from the center of image
    """
    flip, angle = trans_info

    trans_clicks = []
    for c in offsets:
        off_x, off_y = c

        if flip == 1:
            off_x, off_y = off_y, off_x
            off_x = -off_x
            # off_y = -off_y

        trans_y = off_y*np.cos(angle) - off_x*np.sin(angle)
        trans_x = off_y*np.sin(angle) + off_x*np.cos(angle)

        if flip == 0:
            trans_y, trans_x = trans_x, trans_y

        trans_clicks.append( (trans_y, trans_x) )

    return trans_clicks


def apply_transform_original(trans_info: tuple, offsets: np.ndarray):
    """Transform a set of pixel coordinates following a transform matrix
    
    Arguments:
        trans_info {tuple} -- tuple from pink (FLIP, RADIANS)
        offsets {np.ndarray} -- Pixel offsets from the center of image
    """
    flip, angle = trans_info

    trans_clicks = []
    for c in offsets:
        off_y, off_x = c

        if flip == 1:
            off_x = -off_x
            # off_y = -off_y

        trans_y = off_y*np.cos(angle) - off_x*np.sin(angle)
        trans_x = off_y*np.sin(angle) + off_x*np.cos(angle)

        trans_clicks.append( (trans_y, trans_x) )

    return trans_clicks


def overlay_points(neuron: tuple, ed: pu.heatmap, trans: pu.transform, srcs: Table, cata: Table, *args, **kwargs):
    """Produce a list of (x,y) of offsets sources that are near 
    objects for each source matching up to a neuron. 
    
    Arguments:
        neuron {tuple} -- BMU position matching to
        ed {pu.heatmap} -- Mapped similarity
        trans {pu.transform} -- Transform matrix for the similarity
        srcs {Table} -- Catalogue of sources matching those in ed/trans
        cata {Table} -- Catalogue of sources to overlay
    """
    head = som.file_head

    target_idx = neuron[1]*head[1] + neuron[0]

    print("\tSearching for matches")
    pos_min = np.argmin(ed.data.reshape(ed.data.shape[0],-1), axis=1)
    mask = np.argwhere(pos_min == target_idx)
    
    pix_pos = []
    print(f"\tFound {mask.shape}")
    print("\tIterating and looking for object")

    for i in mask:
        row = srcs[i]
        trans_info = trans.get_neuron_transform(index=i, pos=neuron)
        
        # Added to test the individual flipping cases to see what a 
        # problem was. Still not covinced that it is completely correct. 
        # Got to do with a transpose from splitting/concating PINK binaries. 
        # if trans_info[0] == 0:

        spos = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)
        
        res = search_around_sky(spos, cata, seplimit=5*u.arcmin)

        for idx in res[1]:
            offsets = pu.great_circle_offsets(spos, cata[idx], pix_scale=1.5*u.arcsec)
            offsets = apply_transform(trans_info, (offsets,))
            pix_pos.append(offsets)
                
    pix_pos = np.array(pix_pos).reshape(-1,2) # No idea why the reshape is needed. 

    return pix_pos, mask


def overlay_som(som: pu.som, ed: pu.heatmap, trans: pu.transform, 
                srcs: Table, cata: Table, *args, plot: bool=False,
                out_path: str='./', **kwargs):
    """Iterate over the neurons in a SOM, find the sources 
    with it as its BMU, and search for nearby objects in 
    a provide catalogue. Overlay these nearby sources onto
    the neuron
    
    Arguments:
        som {pu.som} -- PINK som file
        ed {pu.heatmap} -- PINK mapping file
        trans {pu.transform} -- Transform matrix for matching sources
        srcs {Table} -- Table of input sources matching the mapping outputs (ed/trans)
        cata {Table} -- Catalogue of objects to overlay
    
    Keyword Arguments:
        plot {bool} -- Save plotting (default: {False})
        out_path {str} -- Output directory for plotting (default: {'./'})
    """
    head = som.file_head

    for y in range(head[1]):
        for x in range(head[2]):
            neuron = (y, x)
            print(f"\nNeuron is {neuron}")

            pos_pix, mask = overlay_points(neuron, ed, trans, srcs, cata)

            print(f"\tMatches found {pos_pix.shape}")

            if plot:
                neuron_img = som.get_neuron(y=neuron[0], x=neuron[1])
                plot_overlay(neuron_img, pos_pix, neuron, out=out_path, mask=mask)



if __name__ == '__main__':
    parser = ArgumentParser(description='Overlay sources onto a neuron')
    parser.add_argument('som', help='PINK produced binary', type=str)
    parser.add_argument('similarity', help='PINK produced mapping file', type=str)
    parser.add_argument('transform', help='PINK produced transform file', type=str)
    parser.add_argument('sources', help='Catalogue of sources matching the object in the similarity/transform', type=str)
    parser.add_argument('catalogue', help='Path to a file with RA/Dec columns', type=str)
    parser.add_argument('--plot','-p',help='Create the plotting figures', action='store_true', default=False)
    parser.add_argument('--out-path','-o', help='Base path to write output to', default='./', type=str)
    args = parser.parse_args()

    print(args)

    som = pu.som(args.som)
    ed = pu.heatmap(args.similarity)
    trans = pu.transform(args.transform)
    srcs = Table.read(args.sources)
    cata = Table.read(args.catalogue)
    cata = SkyCoord(ra=cata['RA'], dec=cata['DEC'])

    overlay_som(som, ed, trans, srcs, cata, plot=args.plot, out_path=args.out_path)


