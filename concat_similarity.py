"""Concatentate a series of similarity PINK mapping binaries into one
"""

import pink_utils as pu
import numpy as np 
import argparse
import struct as st
from glob import glob
from tqdm import tqdm

def concat_simarilty(files: list, out: str):
    """Concat a set of PINK similarity binaries into a single one
    
    Arguments:
        files {list} -- Collection of binaries to concatenate
        out {str} -- Output name of new file
    """
    print('\nLoading initial file to capture example header...')
    ex_head = pu.heatmap(files[0]).file_head

    files.sort()
    total = 0

    with open(out, 'wb')  as of:
        for i in ex_head:
            of.write(st.pack('i', i))

        for f in files:
            print(f'\nAdding {f}')
            ed = pu.heatmap(f)
            total += ed.file_head[0]
            for i in tqdm(range(ed.file_head[0])):
                d = ed.get_bmu_ed(index=i)
                d.astype('f').tofile(of)

        print(f"\nUpdating the header to {total} items")
        of.seek(0)
        of.write(st.pack('i', total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Concatenate a series of similarity measures to one another")
    parser.add_argument('similarity', nargs='+', help='Set of PINK mapping files to concatentat', type=str)
    parser.add_argument('output', help='Output of new concatenated file', type=str)

    args = parser.parse_args()

    concat_simarilty(args.similarity, args.output)