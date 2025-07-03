#!/bin/bash

source config_data.sh

export SPARKLORD_HOME=$home
export app=$app
export slave_count=$slave_count
export SPL_JOB=$job
export SPARKLORD_MODE=$mode    #config, resource
export SPARKLORD_NETWORK=$container_network
export SPARKLORD_RUNMODE=$runmode
export OVERLAP_IP=$overlap_ip
export master_mode=$master_mode
export scenario_type=$scenario_type

# sparklord slave ip
hosts=( )
host_username=( )
host_password=( )

for (( i=0 ; i<$host_len; i++ ));
do
    hosts+=( ${host_ip[i]} )
    host_username+=( ${h_username[i]} )
    host_password+=( ${h_password[i]} )
done

export hosts
export host_username
export host_password

export BUILD_IMAGE=$build_image
export INPUT_DATA=$put_input_data

export log_version=$log_version
