#!/usr/bin/env python
"""Script to apply preprocessing steps to images and create a PINK binary.

Important to remember images are (y,x) when in numpy arrays. Also, critical
to properly write out images to binary file in correct row/column order expected
by PINK. 
"""
import struct as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits

np.random.seed(42) # Order is needed in the Universe


def background(img: np.ndarray, region_size: int=10):
    """Create slices to segment out the inner region of an image
    
    Arguments:
        img {np.ndarray} -- Image to work out background statistics
        region_size {int} -- Size of the inner most region. Equal size dimensions only. 
    """
    img_size = np.array(img.shape)
    center   = img_size // 2
    region   = region_size // 2
    return (slice(center[0]-region, center[0]+region),
            slice(center[1]-region, center[1]+region))


def background_stats(img: np.ndarray, slices: tuple):
    """Calculate and return background statistics. Procedure is following
    from Pink_Experiments repo written originally by EH/KP. 
    
    Arguments:
        img {np.ndarray} -- Image to derive background statistics for
        slices {tuple} -- Tuple of slices returned from background
    """
    empty = ~np.isfinite(img)
    mask  = np.full(img.shape, True) 
    
    mask[empty] = False 
    mask[slices] = False

    return {'mean': np.mean(img[mask]),
            'std': np.std(img[mask]),
            'min': np.min(img[mask]),
            'empty': empty}


def first_process(img: np.ndarray, *args, inner_frac: int=5, clip_level: int=1,
                 weight: float=1., **kwargs):
    """Procedure to preprocess FIRST data, following the original set of steps from 
    EH/KP
    
    Arguments:
        img {nd.ndarray} -- Image to preprocess
        inner_fact {int} -- Fraction of the inner region to extract
        clip_level {int} -- Clip pixels below this background threshold
        weight {float} -- Weight to apply to the channel
    """
    size = img.shape[0] # Lets assume equal pixel sizes
    slices = background(img, region_size=int(size/2))
    bstats = background_stats(img, slices)

    # Replace empty pixels
    img[bstats['empty']] = np.random.normal(loc=bstats['mean'], scale=bstats['std'],
                                            size=np.sum(bstats['empty'].flatten()))
    # Clip out noise pixels and scale image
    noise_level = clip_level*bstats['std']
    img = img - bstats['mean']
    img = np.clip(img, noise_level, 1e10)
    img = img - noise_level

    # Scale is of the maximum pixel in the inner region
    # Attempts to reduce the effect of nearby, unrelated sources
    # This is different to the Galvin+18 preprocessing
    scale = np.max(img[slices])
    if scale == 0.:
        raise ValueError(F"FIRST Scale is {scale}.")
    
    img = img / scale

    # Appply channel weight
    img = img * weight

    return img


def wise_process(img: np.ndarray, *args, inner_frac: int=5, 
                weight: float=1., log10: bool=True):
    """Procedure to preprocess WISE data
    
    Arguments:
        img {np.ndarray} -- [description]
    
    Keyword Arguments:
        weight {float} -- [description] (default: {1.})
        inner_fact {int} -- Fraction of the inner region to extract
        log10 {bool} -- [description] (default: {True})
    """
    size = img.shape[0] # Lets assume equal pixel sizes
    slices = background(img, region_size=int(size/5))
    bstats = background_stats(img, slices)

    # Replace empty pixels
    img[bstats['empty']] = np.random.normal(loc=bstats['mean'], scale=bstats['std'],
                                          size=np.sum(bstats['empty'].flatten()))
    # Clip out nasty values to allow the log10 to work
    # PINK does not like having nans or infs
    img = np.clip(img, 0.0001, 1e10)

    if log10:
        img = np.log10(img)

    # MinMax scale the image
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # apply channel weight
    img = img * weight

    return img


def get_fits(f: str):
    """Read in the FITS data for FIRST

    f {str} -- The filename to read in
    """
    with fits.open(f) as hdu1:
        img = hdu1[0].data.copy()

    return img


def main(files: list, out_path: str, *args, 
        first_path: str='Images/first',
        wise_path: str='Images/wise_reprojected', 
        first_weight: float=0.95,
        wise_weight: float=0.05,
        **kwargs):
    """Run the preprocessing on the set of files
    
    Arguments:
        files {list} -- Iterable with list of filenames to process
        out_path {str} -- Name of the file to create
        first_path {str} -- Relative path containing the first fits images
        wise_path {str} -- Relative path containing the wise reprojected fits images
        first_weight (float) -- Weighting applied to first channel
        wise_weight {float} -- Weighting applied to wise channel

    Raises:
        Exception -- Catch all used for removing files that fail preprocessing
    """

    shape = get_fits(f"{first_path}/{files[0]}").shape
    height, width = shape

    print(f'Derived height, width: {height}, {width}')

    # File handler
    with open(out_path, 'wb') as of:

        of.write(st.pack('i', len(files)))
        of.write(st.pack('i', 2))
        of.write(st.pack('i', width))
        of.write(st.pack('i', height))

        # Will be used to update PINK header at the end
        success = []
        failed  = [] 

        for f in tqdm(files):
            try:
                img_first = get_fits(f"{first_path}/{f}")
                img_first = first_process(img_first, *args, weight=first_weight, **kwargs) 
                
                img_wise = get_fits(f"{wise_path}/{f}")
                img_wise = wise_process(img_wise, *args, weight=wise_weight, **kwargs)

                np.asfortranarray(img_first.astype('f')).tofile(of)
                np.asfortranarray(img_wise.astype('f')).tofile(of)

                # Need to keep track of successfully written file names
                success.append(f)
            
            except ValueError as ve:
                print(f"{f}: {ve}")
                failed.append(f)
            except FileNotFoundError as fe:
                print(f"{f}: {fe}")
                failed.append(f)

            except Exception as e:
                raise e

        # Update header here
        of.seek(0)
        of.write(st.pack('i', len(success)))

        print(f"Have written out {len(success)}/{len(files)} files")

    return success, failed


if __name__ == '__main__':

    df = pd.read_csv('FIRST_Cata_Images.csv')
    files = df['filename'].values

    imgs = main(files, 'F1W1_95_5_imgs.bin')
    success_imgs, failed_imgs = imgs

    # If images were not successfully dumped, then they should be excluded
    # from the catalogue. Until a newer format is supported by PINK, we 
    # have to handle this in this manner. 
    sub_df = df[df['filename'].isin(success_imgs)]

    sub_df.to_csv('F1W1_95_5_Sources.csv')
    sub_df.to_pickle('F1W1_95_5_Sources.pkl')
    sub_df.to_json('F1W1_95_5_Sources.json')

    sub_df = df[df['filename'].isin(failed_imgs)]

    sub_df.to_csv('F1W1_95_5_Sources_Failed.csv')
    sub_df.to_pickle('F1W1_95_5_Sources_Failed.pkl')
    sub_df.to_json('F1W1_95_5_Sources_Failed.json')

