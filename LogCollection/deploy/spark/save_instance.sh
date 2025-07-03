#!/bin/bash

spl_id=$1
container_list=($(docker ps -a --filter "name=$spl_id" --format "{{.Names}}"))
for (( i=0; i<${#container_list[*]}; i++ )); do
   
    echo "save data from ${container_list[i]}"
    docker exec ${container_list[i]} bash -c "tar zcpf /tmp.tar.gz /tmp && chmod 777 /tmp.tar.gz && mv /tmp.tar.gz /opt/${container_list[i]}/" > /dev/null 2>&1
    docker exec ${container_list[i]} bash -c "tar zcpf /data.tar.gz /data && chmod 777 /data.tar.gz && mv /data.tar.gz /opt/${container_list[i]}/" > /dev/null 2>&1
    docker exec ${container_list[i]} bash -c "rm -rf /hadoop/logs/userlogs && tar zcpf /log.tar.gz /hadoop/logs && chmod 777 /log.tar.gz && mv /log.tar.gz /opt/${container_list[i]}/" > /dev/null 2>&1
done

echo "saving checkpoint"
mkdir -p /tmp/${spl_id}
for (( i=0; i<${#container_list[*]}; i++ )); do
    docker checkpoint create --leave-running --checkpoint-dir /tmp/${spl_id} ${container_list[i]} ${container_list[i]}_cp &
done

cp_line=`ps aux | grep "docker checkpoint" | wc -l`
while [ $cp_line -ne 1 ]; do
    sleep 1s
    cp_line=`ps aux | grep "docker checkpoint" | wc -l`
done

mkdir /tmp/${spl_id}_tar
for (( i=0; i<${#container_list[*]}; i++ )); do
    container=`echo ${container_list[i]} | sed "s/${spl_id}_//"`
    mv /tmp/${spl_id}/${container_list[i]}_cp /tmp/${spl_id}/${container}_cp
    tar -Ipigz -cpf /tmp/${spl_id}/cp_${container}.tar.gz -C /tmp/${spl_id} .
    mv /tmp/${spl_id}/cp_${container}.tar.gz /tmp/${spl_id}_tar/cp_${container}.tar.gz
    rm -rf /tmp/${spl_id}/${container}_cp
done
