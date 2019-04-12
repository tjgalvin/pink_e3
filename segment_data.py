import pink_utils as pu
import numpy as np
import pandas as pd
import argparse
import sys
import networkx as nx
from tqdm import tqdm
from scipy.stats import percentileofscore
from collections import defaultdict
from itertools import product

def bmu_score(ed: pu.heatmap):
    """Produce the BMU scores for all sources in the heatmap instance
    
    Arguments:
        ed {pu.heatmap} -- Heatmaps for a set of mapped images
    """
    bmu_scores = []
    print('Generating the BMU scores...')
    for i in tqdm(range(ed.file_head[0])):
        hm = ed.get_bmu(index=i)
        bmu_scores.append( percentileofscore(ed.data[:,hm[0], hm[1]], ed.data[i, hm[0], hm[1]]) )

    return np.array(bmu_scores)


def connvis_clusters(ed: pu.heatmap, min_edge: float=0.1, log: bool=False):
    """Produce clusters based on the CONNvis method
    
    Arguments:
        ed {pu.heatmap} -- Instance linking to a PINK mapping

    Keyword Arguments:
        min_edge {float} -- Minimum weighted value to retain an edge {Default: 0.1}
        log {bool} -- Log the edges (default: False)
    """
    data = ed.data
    data = data.reshape((data.shape[0], np.prod(data.shape[1:])))

    sort = np.argsort(data, axis=1)
    links = defaultdict(float)

    for i in sort[:]:
        for x, y in zip(i[:-1], i[1:]):
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
            G.add_edge(k[0], k[1], weight=val)
        else:
            G.add_nodes_from(k)   
 
    subs = [G.subgraph(c) for c in nx.connected_components(G)]

    split_idx = []
    small_clusters = []
    nodes = []

    for s in subs:
        nodes += [k for k in s.nodes.keys()]
        keys = np.array([k for k in s.nodes.keys()])
        idxs = np.argwhere(np.in1d(sort[:,0], keys ) ).flatten()
        if len(keys) < 5:
            small_clusters.extend(idxs.tolist())
        else:
            split_idx.append(idxs)

    split_idx.append(np.array(small_clusters))

    print(sum([len(k) for k in split_idx]))
    print(len(nodes))
    return split_idx


def bmu_segment(ed: pu.heatmap):
    """Produce the split indexs for splitting based on the best matching neuron. 
    
    Arguments:
        ed {pu.heatmap} -- Reference to the SOM similarity matrix for each source
    """
    head = ed.file_head
    som_dim = head[1:]

    x = np.arange(som_dim[0])
    y = np.arange(som_dim[1])

    data = ed.data.reshape((head[0], np.prod(som_dim)))
    argmin = np.unravel_index(np.argmin(data, axis=1), som_dim)

    split_idxs = []
    isolated_idxs = []

    for coord in product(x,y):
        argmask = np.argwhere((argmin[0] == coord[0]) & (argmin[1] == coord[1]))[:, 0]
        if len(argmask) < 20:
            isolated_idxs.extend([i for i in argmask])
        else:
            split_idxs.append(argmask)

    if len(isolated_idxs) > 0:
        split_idxs.append(isolated_idxs)

    print(f"\nTotal of {sum(len(i) for i in split_idxs)} items matched to BMU")

    return split_idxs


def segment_data(data: np.ndarray, no_segs: int):
    """The data that will be split. It is assumed to represent some
    metric, like a similarity measure or BMU score. It will be sorted
    and divided into `no_segs` chunks. Returns a array of N length with
    indicies to split
    
    Arguments:
        data {np.ndarray} -- Data to be split
        no_segs {int} -- [description]
    """
    sort_idx = np.argsort(data)
    split_idx = np.array_split(sort_idx, no_segs)


    return split_idx


def max_size_split(imgs: pu.image_binary, no_segs: int):
    """Produce a collection of segments no larger than a specified number
    
    Arguments:
        ed {pu.heatmap} -- PINK produced similarity matrix
        no_segs {int} -- Number of segments to split the image binary to
    """
    no_srcs = imgs.file_head[0]
    print('Number of sources: ', no_srcs)
    ones = np.ones(no_srcs)
    print(ones.shape)

    idx = np.argwhere(ones)[:,0]
    split_idx = np.array_split(idx, no_segs)

    print(f"Total split: {sum(len(i) for i in split_idx)}")

    return split_idx


def write_segments(split_idx: np.ndarray, base_out: str, imgs: pu.image_binary,
                    cata: pd.DataFrame):
    """Create new binary files based on the split indicies
    
    Arguments:
        split_idx {np.ndarray} -- The collection of indicies to split data on
        base_out {str} -- Base name to create new files with
        imgs {pu.image_binary} -- Image binary to copy images from
        cata {pd.DataFrame} -- Catalogue of source information that accompanies imgs
    """
    for count, idxs in enumerate(split_idx):
        print(idxs.shape)
        pu.segment_image_bin(imgs, idxs, f"{base_out}_Seg{count}.bin")
        sub_df = cata.loc[idxs]

        sub_df.to_csv(f"{base_out}_Seg{count}.csv")


def save_segment_idxs(split_idx: np.ndarray, base_out: str):
    """Save the indicies created to split the data on
    
    Arguments:
        split_idx {np.ndarray} -- Array indices of segments
        base_out {str} -- Base path to write to
    """
    np.save(f"{base_out}_Seg_IDX.npy", split_idx)


def create_splits(ed: pu.heatmap, imgs: pu.image_binary, bmu_segs: int=None, min_edge: float=None, bmu: bool=False,
                 no_segs: int=None):
    """Create the arg idxs to split the image samples against
    
    Arguments:
        ed {pu.heatmap} -- PINK produced similarity matrix
        imgs {pu.image_binary} -- PINK image binary file

    Keywprd Arguments
        bmu_segs {int} -- Number of desired segments. Implicitly sets mode to `bmuscore` (Default: {None})
        min_edge {float} -- Minimum edge for CONNvis. Implicitly sets mode to `connvis` (Default: {None})
        bmu {bool} -- Segment objects based on their BMU. (Default: {False})
        no_segs {bool} -- Number of segments to split the image binary into (Default: {None})
    """

    if bmu_segs is not None:
        scores = bmu_score(ed)
        split_idx = segment_data(scores, no_segs)
    elif min_edge is not None:
        split_idx = connvis_clusters(ed, min_edge=min_edge)
    elif bmu:
        split_idx = bmu_segment(ed)
    elif no_segs is not None:
        split_idx = max_size_split(imgs, no_segs)
    else:
        raise ValueError("No `segmentation mode` set. ")
    

    return split_idx


def segment(ed: pu.heatmap, imgs: pu.image_binary, cata: pd.DataFrame,
            base_out: str, bmu_segs: int=None, min_edge: float=None, bmu: bool=False, 
            no_segs: int=None, write: bool=True):
    """Perform segmenting of items
    
    Arguments:
        ed {pu.heatmap} -- Heatmap of mapping to SOM
        imgs {pu.image_binary} -- Images that accompany the ed
        cata {pd.DataFrame} -- Catalogue of sources for imgs/ed
        base_out {str} -- Base output name
    
    Keyword Arguments
        bmu_segs {int} -- Number of segments (Default: {None})
        min_edge {int} -- Minimum edge required for CONNvis clustering (Default: {None})
        bmu {bool} -- Split sources based on their BMU (Default: {False})
        no_segs {bool} -- Maximum number of objects for each segment (Default: {False})
        write {bool} -- Whether to write out the segments (Default: {True})
    """
    split_idx = create_splits(ed, imgs, bmu_segs=bmu_segs, min_edge=min_edge, bmu=bmu, no_segs=no_segs)

    if split_idx is None:
        return

    if write:
        write_segments(split_idx, base_out, imgs, df)
        save_segment_idxs(split_idx, base_out)
    else:
        print("\nSize of splits")
        for c, idxs in enumerate(split_idx):
            print(f"{c}: {len(idxs)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Divide an image binary into segments based on "\
                                     "how well they score to their BMU")
    parser.add_argument('image_binary', help="Path to the Pink image binary file")
    parser.add_argument('similarity', help="Similarity matrix produced from mapping image_binary to SOM")
    parser.add_argument('catalog', help='Catalogue that accompanies the image_binary')
    parser.add_argument('base_out', help="Base path for output files (segment and file type appended)")
    parser.add_argument('--bmu_segs', help="Number of segments to split based on the BMU scores", default=None, type=int)
    parser.add_argument('--min_edge', help="Minimum edge to define a link using CONNvis", default=None, type=float)
    parser.add_argument('--bmu', help="Split sources based on which BMU. Will produce many, many segments.", default=False, action='store_true')
    parser.add_argument('--no-segs', help='Number of segments to split the image binary into', default=None, type=int)
    parser.add_argument('--write', help='Write out the segment files', default=False, action='store_true')
    
    args = parser.parse_args()

    if args.bmu_segs is None and args.min_edge is None and args.bmu is False and args.no_segs is None:
        print('A segmentation method has to be set. ')
        sys.exit(1)
    elif sum([args.bmu_segs is not None, args.min_edge is not None, args.bmu, args.no_segs is not None]) > 1:
        print('Only a single segmentation mathod can be set. ')
        sys.exit(1)

    print('Loading Heatmap...')
    try:
        ed = pu.heatmap(args.similarity)
    except:
        print('\nWarning: Could not load heatmap. Continuing')
        ed = 'None'

    print('Loading images...')
    imgs = pu.image_binary(args.image_binary)
    print('Loading catalogue')
    df = pd.read_csv(args.catalog)
    base_out = args.base_out
    bmu_segs = args.bmu_segs
    min_edge = args.min_edge
    bmu = args.bmu
    no_segs = args.no_segs

    write = args.write

    segment(ed, imgs, df, base_out, bmu_segs=bmu_segs, min_edge=min_edge, bmu=bmu, no_segs=no_segs, write=write)




    
