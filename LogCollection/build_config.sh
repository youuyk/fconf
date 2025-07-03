#!/bin/bash

yq -o shell 'del(.config.hosts) | .config' config.yaml > config_data.sh
echo host_len=$(yq '.config.hosts | length' config.yaml) >> config_data.sh

readarray host_data < <(yq e -o=j -I=0 '.config.hosts[]' config.yaml )
for (( i=0; i<${#host_data[*]}; i++ )); do
    echo host_ip[$i]=$(echo ${host_data[i]} | yq e '.host.ip' -) >> config_data.sh
    echo h_username[$i]=$(echo ${host_data[i]} | yq e '.host.username' -) >> config_data.sh
    echo h_password[$i]=$(echo ${host_data[i]} | yq e '.host.password' -) >> config_data.sh
done