#!/bin/bash

spl_id=$1
cp_dir=$2
container_list=($(docker ps -a --filter "name=$spl_id" --format "{{.Names}}"))

echo "Removing old data"
for (( i=0; i<${#container_list[*]}; i++ )); do
    container_id=`docker ps -aq --no-trunc --filter name=${container_list[i]}`

    docker exec ${container_list[i]} bash -c "rm -rf /hadoop/logs/*.log"
    docker exec ${container_list[i]} bash -c "rm /spark/logs/*" 2> /dev/null
    docker exec ${container_list[i]} bash -c "rm -rf /data"
    docker exec ${container_list[i]} bash -c "rm -rf /tmp"

    rm -rf /var/lib/docker/containers/${container_id}/checkpoints/* > /dev/null &
    echo "Removed ${container_list[i]}"
done

echo "Stop docker"
docker stop `docker ps -a -q --filter "name=$spl_id"`

for (( i=0; i<${#container_list[*]}; i++ )); do
    echo "load data to ${container_list[i]}"
    container=`echo ${container_list[i]} | sed "s/${spl_id}_//"`
    container_id=`docker ps -aq --no-trunc --filter name=${container_list[i]}`

    mkdir -p /tmp/${spl_id}/${container}
    tar -Ipigz -xpf ${cp_dir}/${container}/data.tar.gz -C /tmp/${spl_id}
    tar -Ipigz -xpf ${cp_dir}/${container}/tmp.tar.gz -C /tmp/${spl_id}
    tar -Ipigz -xpf ${cp_dir}/${container}/log.tar.gz -C /tmp/${spl_id}/${container}
    docker cp /tmp/${spl_id}/data ${container_list[i]}:/
    docker cp /tmp/${spl_id}/tmp ${container_list[i]}:/

  
    logs=(`ls /tmp/${spl_id}/${container}/hadoop/logs`)
    for (( j=0; j<${#logs[*]}; j++ )); do
        docker cp /tmp/${spl_id}/${container}/hadoop/logs/${logs[j]} ${container_list[i]}:/hadoop/logs
    done

    mkdir /var/lib/docker/containers/${container_id}/checkpoints/${container_list[i]}
    tar -Ipigz -xpf ${cp_dir}/cp_${container}.tar.gz -C /var/lib/docker/containers/${container_id}/checkpoints
done

for (( i=0; i<${#container_list[*]}; i++ )); do
    echo "Restore Checkpoint ${container_list[i]}"
    container=`echo ${container_list[i]} | sed "s/${spl_id}_//"`
    docker start --checkpoint ${container}_cp ${container_list[i]} &
done

cp_line=`ps aux | grep "docker start" | wc -l`
while [ $cp_line -ne 1 ]; do
    sleep 1s
    cp_line=`ps aux | grep "docker start" | wc -l`
done

for (( i=0; i<${#container_list[*]}; i++ )); do
    container=`echo ${container_list[i]} | sed "s/${spl_id}_//"`
    logs=(`ls /tmp/${spl_id}/${container}/hadoop/logs`)
    for (( j=0; j<${#logs[*]}; j++ )); do
        cat /tmp/${spl_id}/${container}/hadoop/logs/${logs[j]} | docker exec -i ${container_list[i]} bash -c "cat > /hadoop/logs/${logs[j]}"
    done
done

rm -rf /tmp/${spl_id}
