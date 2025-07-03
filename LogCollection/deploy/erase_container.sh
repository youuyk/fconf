#!/bin/bash

if [ $# -eq 0 ]; then
    spl_id="spl"
else
    spl_id=$1
fi

container_list=$(docker ps -a --filter "name=${spl_id}" -q)

echo $container_list

if [ -z "$container_list" ]
then
    exit 1
fi
#echo "return_ip, cpu_list"
return_ip=$(docker inspect $container_list -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')

cpu_list=$(docker inspect $container_list -f '{{.HostConfig.CpusetCpus}}')

echo $return_ip
#echo $cpu_list 

for cpus in $cpu_list; do
    cpus=$(echo $cpus | tr "," "\n")
    for cpu in $cpus; do
	    rm /home/$(whoami)/sparklord2/resource/cpu/`printf "%02d" $cpu` 2> /dev/null
    done
done

docker stop $container_list >> /dev/null
docker rm $container_list >> /dev/null

echo $return_ip
