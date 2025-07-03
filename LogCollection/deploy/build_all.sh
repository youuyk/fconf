#!/bin/bash
app=$1
home_dir=$2

image_list=("${app}_master" "${app}_slave")

for image in ${image_list[@]}; do
    echo "Building container image" $image
    docker build -t $image:latest $home_dir/apps/$image
done
