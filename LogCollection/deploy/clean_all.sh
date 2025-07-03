#!/bin/bash

app=$1
image_list=("${app}_master" "${app}_slave")

for image in ${image_list[@]}; do
    echo "Delete container image for" $image
    docker rmi ${image}
done

