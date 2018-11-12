#!/usr/bin/env python

"""This script is used in conjuction with the label dictionary to
segment a PINK mapped dataset

Important to remember images are (y,x) when in numpy arrays. Also, critical
to properly write out images to binary file in correct row/column order expected
by PINK. 

TODO: Add proper kwargs to certain functions, or rethink passing args to 
      segment_main
"""
import os
import struct as st
import pickle
import argparse
import pandas as pd
import numpy as np
import pink_utils as pu
from tqdm import tqdm
import matplotlib.pyplot as plt

def som_size(labels: dict):
    """Find the size of the SOM. Assume a quad shape
    
    Arguments:
        labels {dict} -- The annotated labels of the SOM map. Keys are positions. 
    """
    pos = 0
    key = (0,0)
    for p in labels.keys():
        if np.prod(p) > pos:
            key = p
    
    return [k+1 for k in key]


def make_mask(label: dict, target: str, shape: tuple, fill_val: int=1):
    """Produce a numpy mask array based on the location of labels matching
    the target in the `label` dictionary.  
    
    Arguments:
        label {dict} -- Dictionary whose keys are the location of the corresponding label valu
        target {str} -- Desired target value to make mask for
        shape {tuple} -- Shape of SOM map. TODO: Include this in label structure
    
    Keyword Arguments:
        fill_val {int} -- Value to place in array when match found (default: {1})
    
    Returns:
        [np.ndarray] -- Masking array
    """

    mask = np.zeros(shape, dtype=type(fill_val))
    for pos in label:
        if label[pos] == target:
            mask[pos] = fill_val

    return mask


def ed_to_prob(data: np.ndarray, stretch: float=1.):
    """Convert the Euclidean distance matrix from PINK to a proability like matrix
    
    Arguments:
        data {np.ndarray} -- Three dimension all of Euclidean distances [NImages, height, width]
    
    Keyword Arguments:
        thresh {float} -- Stretching parameter to introduce non-linearity (default: {1.})
    
    Returns:
        np.ndarray -- Probability matrix of shape [NImages, height, width]
    """

    assert len(data.shape) == 3, 'Length not correct'
    
    prob = data - data.min(axis=(1,2))[:, None, None]
    prob = 1. / (1. + prob)**stretch
    prob = prob / prob.sum(axis=(1,2))[:, None, None]
    
    return prob


def make_pink_binary(match_idx: np.ndarray, imgs: pu.image_binary, out: str):
    """Produce a new PINK image binary file of the matched objectd
    
    Arguments:
        match_idx {np.ndarray} -- indicies of the objects to include in new binary
        imgs {pu.image_binary} -- original PINK image binary
        out {str} -- Base name of the output file
    """
    out_file = f"{out}.bin"
    if os.path.exists(out_file):
        raise FileExistsError

    no_chans = imgs.file_head[1]
    with open(out_file, 'wb') as of:
        of.write(st.pack('iiii', *(match_idx.shape + imgs.file_head[1:])))
        for j in tqdm(match_idx):
            for c in range(imgs.file_head[1]):
                np.asfortranarray(imgs.get_image(index=j, channel=c)).astype('f').tofile(of)
            

def target_prob(prob: np.ndarray, mask: np.ndarray, match: float, negate: bool=False):
    """Derive and apply the probability of an object matching a label region
    
    Arguments:
        prob {np.ndarray} -- Probability matrix of shape [NImage, height, width]
        mask {np.ndarray} -- Mask of label region
        match {float} -- Matching level a object needs to be included
        
    Keyword Arguments:
        negate {bool} -- Negate the matching criteria (default: {1})
    
    Returns:
        {np.ndarray} -- Matching probabilities for all objects
        {np.ndarray} -- Indicies matching the criteria
    """
    match_prob = prob[:,mask].sum(axis=1)
    match_pos  = match_prob > match

    if negate:
        match_pos = ~match_pos

    return match_prob, np.argwhere(match_pos).T[0]


def make_plots(match_prob: np.ndarray, match: float, out_base):
    """Create defined diagnostic plots
    
    Arguments:
        match_prob {np.ndarray} -- Matching probability of all sources
        match {float} -- Matching threshold
    """

    fig, ax = plt.subplots(1,1)

    ax.hist(match_prob, bins=100)
    ax.axvline(match, color='black')

    fig.savefig(f"{out_base}_distribution.pdf")


def segment_catalog(csv_in: str, match_idx: np.ndarray, out_base: str):
    """Segment out the matched sources from a given catalog
    
    Arguments:
        csv_in {str} -- Input csv file with the original catalog
        match_idx {np.ndarray} -- Integer indicies of matched objectd to segment
        out_base {str} -- Base output path of new catalog
    """
    df = pd.read_csv(csv_in)

    sub_df = df.iloc[match_idx]

    out_file = f"{out_base}.csv"
    sub_df.to_csv(out_file)

    return out_file


def segment_main(args: dict):
    """Driver function used to segment data out from a PINK
    image file based on an annotated map   
    
    Arguments:
        args {dict} -- Options containing working parameters
    """
    labels = pickle.load(open(args['label_dict'], 'rb'))
    imgs = pu.image_binary(args['image_binary'])
    ed = pu.heatmap(args['mapping'])

    som_shape = som_size(labels)
    mask = make_mask(labels, args['target'], som_shape, fill_val=True)

    prob = ed_to_prob(ed.data, stretch=args['stretch'])

    match_prob, match_idx = target_prob(prob, mask, args['match'], negate=args['negate'])

    if args['plot']:
        make_plots(match_prob, args['match'], args['out_base'])

    make_pink_binary(match_idx, imgs, args['out_base'])

    if args['csv'] is not None:
        segment_catalog(args['csv'], match_idx, args['out_base'])

    np.save(f"{args['out_base']}_match_prob", match_prob)
    np.save(f"{args['out_base']}_match_idx", match_idx)


if __name__ == '__main__':
    dmesg = "Tool to split a PINK image binary into segments based on "\
            "a desired user label and a annotated PINK map. "
    parser = argparse.ArgumentParser(description=dmesg)

    parser.add_argument('label_dict', help='Pickle label dictionary of annotated map')
    parser.add_argument('image_binary', help='PINK image binary file')
    parser.add_argument('mapping', help='Mapped PINK Euclidean distance file')
    parser.add_argument('out_base', help='Base output path of the segmented PINK image binary and other items')
    parser.add_argument('-s', '--stretch', default=1, type=float, help='Stretching parameter for convert euclidean distance to likelihood')
    parser.add_argument('-m', '--match', default=0.7, help='Threshold object has to match for it to be copied')
    parser.add_argument('-n', '--negate', default=False, action='store_true', help='Negate the matching criteria')
    parser.add_argument('-t', '--target', default='0', help='Label target on annotated map to segment out')
    parser.add_argument('-c', '--csv', help='Path to Pandas dataframe with catalogue to segment out')
    parser.add_argument('-p', '--plot', default=False, action='store_true', help='Create diagnostic plots of intermidate stages')

    args = parser.parse_args()

    segment_main(vars(args))