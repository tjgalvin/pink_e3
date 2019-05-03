"""Concatentate a series of similarity PINK mapping binaries into one

TODO: I think that I have to transpose the similarity array if I read in
as a numpy array. A fortran/c mismatch. Consider reading in directly
from the file handler to avoid issue
"""
import re
import pink_utils as pu
import numpy as np 
import argparse
import struct as st
from glob import glob
from tqdm import tqdm

def natural_sort(l: list):
    """Sort a set of strings with natural human sorting. 1,2,3,11,12,13 -
    not 1,11,12,13,2,3
    
    Arguments:
        l {list} -- List of items to sort
    
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    
    return
    
    
def concat_simarilty(files: list, out: str):
    """Concat a set of PINK similarity binaries into a single one
    
    Arguments:
        files {list} -- Collection of binaries to concatenate
        out {str} -- Output name of new file
    """
    # files = sorted(files, key=natural_sort)
    natural_sort(files)

    print('\nLoading initial file to capture example header...')
    ex_head = pu.transform(files[0]).file_head

    total = 0

    with open(out, 'wb')  as of:
        for i in ex_head:
            of.write(st.pack('i', i))

        for f in files:
            print(f'\nAdding {f}')
            trans = pu.transform(f)
            total += trans.file_head[0]
            f = trans.f
            f.seek(len(trans.file_head)*4)
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                of.write(chunk)

        print(f"\nUpdating the header to {total} items")
        of.seek(0)
        of.write(st.pack('i', total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Concatenate a series of transforms to one another")
    parser.add_argument('transform', nargs='+', help='Set of PINK mapping files to concatentat', type=str)
    parser.add_argument('output', help='Output of new concatenated file', type=str)

    args = parser.parse_args()

    concat_simarilty(args.transform, args.output)
