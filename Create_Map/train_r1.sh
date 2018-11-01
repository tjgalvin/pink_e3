#!/bin/bash

# Script to produce the first SOM with the updated preprocessing.py
# code. Uses the same receipe from Galvin+18 developed by EH/KP. 
# Dimensions of SOM are hard coded to preserve a record. Names and
# path of binary files potentially may change depending on where 
# PINK is executed

if [ $# -ne 2 ]; then
    echo "USAGE: $0 IMGBIN BASE HEIGHT WIDTH"
    echo "      IMGBIN - PINK binary file with input images"
    echo "      BASE   - Base file name for the input and output files from PINK"
    
    exit 1
fi

imgbin=$1       # Name of the image binary file
base=$2         # Base name of the output SOM/transform files
height=10       # Height of SOM
width=10        # Width of SOM

# Step one
Pink -n 48 --inter-store keep --init random_with_preferred_direction --layout quadratic --num-iter 2 --progress 0.05 --som-width $width --som-height $height --dist-func gaussian 1.5 0.1 --train "$imgbin" "$base"_1.bin

# Step two
Pink -n 92 --init "$base"_1.bin --inter-store keep --layout quadratic --num-iter 2 --progress 0.05 --som-width $width --som-height $height --dist-func gaussian 1.0 0.05 --train "$imgbin" "$base"_2.bin

# Step three
Pink -n 92 --init "$base"_2.bin --inter-store keep --layout quadratic --num-iter 3 --progress 0.05 --som-width $width --som-height $height --dist-func gaussian 0.7 0.05 --train "$imgbin" "$base"_3.bin

# Step four
Pink --init "$base"_3.bin --inter-store keep --layout quadratic --num-iter 3 --progress 0.05 --som-width $width --som-height $height --dist-func gaussian 0.7 0.05 --train "$imgbin" "$base"_4.bin

# Step five
Pink --init "$base"_4.bin --inter-store keep --layout quadratic --num-iter 3 --progress 0.05 --som-width $width --som-height $height --dist-func gaussian 0.3 0.01 --train "$imgbin" "$base"_5.bin


