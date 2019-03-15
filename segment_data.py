import pink_utils as pu
import numpy as np
import pandas as pd
import argparse
import networkx as nx
from tqdm import tqdm
from scipy.stats import percentileofscore
from collections import defaultdict

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


def connvis_clusters(ed: pu.heatmap, min_edge: float=0.1):
    """Produce clusters based on the CONNvis method
    
    Arguments:
        ed {pu.heatmap} -- Instance linking to a PINK mapping

    Keyword Arguments:
        min_edge {float} -- Minimum weighted value to retain an edge {Default: 0.1}
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
    
    max_val = max(links.values())

    G = nx.Graph()
    
    for k, v in links.items():
        val = v/max_val
        if val > 0.15:
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


def create_splits(ed: pu.heatmap, no_segs: int=None, min_edge: float=None):
    """Create the arg idxs to split the image samples against
    
    Arguments:
        ed {pu.heatmap} -- PINK produced similarity matrix
        
    Keywprd Arguments
        no_segs {int} -- Number of desired segments. Implicitly sets mode to `bmuscore` (Default: {None})
        min_edge {float} -- Minimum edge for CONNvis. Implicitly sets mode to `connvis` (Default: {None})
    """

    if no_segs is not None:
        scores = bmu_score(ed)
        split_idx = segment_data(scores, no_segs)
    elif min_edge is not None:
        split_idx = connvis_clusters(ed, min_edge=min_edge)

    return split_idx


def segment(ed: pu.heatmap, imgs: pu.image_binary, cata: pd.DataFrame,
            base_out: str, no_segs: int=None, min_edge: float=None, write: bool=True):
    """Perform segmenting of items
    
    Arguments:
        ed {pu.heatmap} -- Heatmap of mapping to SOM
        imgs {pu.image_binary} -- Images that accompany the ed
        cata {pd.DataFrame} -- Catalogue of sources for imgs/ed
        base_out {str} -- Base output name
    
    Keyword Arguments
        no_segs {int} -- Number of segments (Default: {None})
        min_edge {int} -- Minimum edge required for CONNvis clustering (Default: {None})
        write {bool} -- Whether to write out the segments (Default: {True})
    """
    split_idx = create_splits(ed, no_segs=no_segs, min_edge=min_edge)

    if split_idx is None:
        return

    if write:
        write_segments(split_idx, base_out, imgs, df)
        save_segment_idxs(split_idx, base_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Divide an image binary into segments based on "\
                                     "how well they score to their BMU")
    parser.add_argument('image_binary', help="Path to the Pink image binary file",
                        nargs=1)
    parser.add_argument('similarity', help="Similarity matrix produced from mapping image_binary to SOM",
                        nargs=1)
    parser.add_argument('catalog', help='Catalogue that accompanies the image_binary', nargs=1)
    parser.add_argument('base_out', help="Base path for output files (segment and file type appended)", nargs=1)
    parser.add_argument('--no_segs', help="Number of segments to split against based on the BMU scores", nargs=1, default=None)
    parser.add_argument('--min_edge', help="Minimum edge to define a link using CONNvis", nargs=1, default=None)
    parser.add_argument('--write', help='Write out the segment files', default=False, action='store_true')
    
    args = parser.parse_args()

    if args.no_segs is None and args.min_edge is None:
        print('A segmentation method has to be set. ')
        import sys
        sys.exit(1)

    print('Loading Heatmap...')
    ed = pu.heatmap(args.similarity[0])
    print('Loading images...')
    imgs = pu.image_binary(args.image_binary[0])
    print('Loading catalogue')
    df = pd.read_csv(args.catalog[0])
    base_out = args.base_out[0]
    no_segs = int(args.no_segs[0]) if args.no_segs is not None else None
    min_edge = float(args.min_edge[0]) if args.min_edge is not None else None

    write = args.write

    segment(ed, imgs, df, base_out, no_segs=no_segs, min_edge=min_edge, write=write)




    
