#!/bin/bash

spl_id=$1
container_list=$(docker ps -a --filter "name=$spl_id" --format "{{.Names}}")
for container in ${container_list[@]}; do
    docker exec -i ${container} bash -c "chmod 777 /hadoop/logs/*.log && mv /hadoop/logs/*.log /opt/${container}"
    docker exec -i ${container} bash -c "chmod 777 /spark/logs/spark.log && mv /spark/logs/spark.log /opt/${container}/spark.log" > /dev/null 2>&1
done
