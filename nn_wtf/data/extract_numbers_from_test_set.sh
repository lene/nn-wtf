#!/usr/bin/env bash

data_dir=../../.nn_wtf-data
training_data=${data_dir}/t10k-images-idx3-ubyte

image_size=28
image_bytes=$[$image_size*$image_size]
header_size=16

if [ ! ${training_data} ]; then
    gunzip ${training_data}.gz || exit 1
fi

function n_images_filename() {
    echo train_images_${1}.raw
}

function nth_image_filename() {
    echo train_image_${1}.raw
}

# extract first n images
function extract_n_images {
    n=$1
    # extract first n images + header
    head -c$[$header_size+$n*$image_bytes] ${training_data} > tmp.raw
    # drop header data
    tail -c+$[$header_size+1] tmp.raw > $(n_images_filename $n)
    rm -f tmp.raw
    # check the image with ImageMagick display
    display -size ${image_size}x$[${n}*${image_size}] -depth 8 GRAY:$(n_images_filename $n)
}

function extract_nth_image() {
    n=$1
    if [ ! -f $(n_images_filename $n) ]; then
        extract_n_images $n
    fi
    tail -c${image_bytes} $(n_images_filename $n) > $(nth_image_filename $n)
    display -size ${image_size}x${image_size} -depth 8 GRAY:$(nth_image_filename $n)
}

# example usage
extract_n_images 1
extract_n_images 3
extract_nth_image 3
extract_nth_image 5