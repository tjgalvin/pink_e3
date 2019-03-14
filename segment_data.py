import pink_utils as pu
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.stats import percentileofscore

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
        pu.segment_image_bin(imgs, idxs, f"{base_out}_Seg{count}.bin")
        sub_df = cata[idxs]

        sub_df.to_csv(f"{base_out}_Seg{count}.csv")


def save_segment_idxs(split_idx: np.ndarray, base_out: str):
    """Save the indicies created to split the data on
    
    Arguments:
        split_idx {np.ndarray} -- Array indices of segments
        base_out {str} -- Base path to write to
    """
    np.save(f"{base_out}_Seg_IDX.npy", split_idx)


def segment(ed: pu.heatmap, imgs: pu.image_binary, cata: pd.DataFrame,
            base_out: str, no_segs: int):
    """Perform segmenting of items
    
    Arguments:
        ed {pu.heatmap} -- Heatmap of mapping to SOM
        imgs {pu.image_binary} -- Images that accompany the ed
        cata {pd.DataFrame} -- Catalogue of sources for imgs/ed
        base_out {str} -- Base output name
        no_segs {int} -- Number of segments
    """

    scores = bmu_score(ed)
    split_idx = segment_data(scores, no_segs)

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
    parser.add_argument('no_segs', help="Number of segments to split against", nargs=1)
    
    args = parser.parse_args()

    ed = pu.heatmap(args.similarity)
    imgs = pu.image_binary(args.image_binary)
    df = pd.read_csv(args.catalog)
    base_out = args.base_out
    no_segs = args.no_segs

    segment(ed, imgs, df, base_out, no_segs)




    