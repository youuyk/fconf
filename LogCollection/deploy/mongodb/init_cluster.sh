#!/bin/bash
source ./config.sh

docker exec -it spl_master mongosh --eval "rs.initiate()"

for((i=0; i<${#slave[*]}; i++))
do
    slave_ip=${slave[i]}[*]
    slave_ip=(${!slave_ip})

    for((ip=0; ip<${#slave_ip[*]}; ip++))
    do
        docker exec -i spl_master mongosh --eval "rs.add({host:\"${slave_ip[ip]}:27017\", priority:1})"
    done
done
