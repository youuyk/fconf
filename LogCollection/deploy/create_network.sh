#!/bin/bash

if [ $# -eq 0 ]; then
    SPARKLORD_NETWORK="bridge"
else
    SPARKLORD_NETWORK=$1
fi

if [ $SPARKLORD_NETWORK == "ipvlan" ]; then
    ip_name=$(ip route | awk '/default/ {print $5; exit}')
    gateway=$(ip route | awk '/default/ {print $3; exit}')
    subnet=$(ip route | grep $ip_name | grep src | awk '{print $1; exit}')
    myip=$(ip route | grep $ip_name | grep src | awk '{print $9; exit}')

    docker network rm spark_network > /dev/null 2>&1
    docker network create -d ipvlan --subnet=$subnet --gateway=$gateway -o parent=$ip_name spark_network
elif [ $SPARKLORD_NETWORK == "bridge" ]; then
    docker network rm spark_network > /dev/null 2>&1
    docker network create -d bridge --subnet=172.20.1.0/24 --ip-range=172.20.1.0/24 --gateway=172.20.1.1 spark_network
fi
