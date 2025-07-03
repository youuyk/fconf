#!/bin/bash
source ./config.sh

spl_id=$1

for((i=0; i<${#hosts[*]}; i++))
do
    if [ "${hosts[i]}" == "127.0.0.1" ] || [ "${hosts[i]}" == "localhost" ]; then
        container_list=$(docker ps -a --filter "name=${spl_id}" -q)
        for c_list in $container_list; do
            echo "send configuration file" 
            # send spark configuration (fault injected)
            docker cp ${SPARKLORD_HOME}/config_files/tmp/${spl_id}/spark-defaults.conf ${c_list}:/spark/conf/spark-defaults.conf
            
            # send hadoop configuration (default)
            docker cp ${SPARKLORD_HOME}/config_files/tmp/${spl_id}/core-site.xml ${c_list}:/hadoop/etc/hadoop/core-site.xml
            docker cp ${SPARKLORD_HOME}/config_files/tmp/${spl_id}/hdfs-site.xml ${c_list}:/hadoop/etc/hadoop/hdfs-site.xml
            docker cp ${SPARKLORD_HOME}/config_files/tmp/${spl_id}/yarn-site.xml ${c_list}:/hadoop/etc/hadoop/yarn-site.xml
            docker cp ${SPARKLORD_HOME}/config_files/tmp/${spl_id}/mapred-site.xml ${c_list}:/hadoop/etc/hadoop/mapred-site.xml
        
        done
        continue
    fi

:<<'END' 
    #ssh ${host_username[i]}@${hosts[i]} rm -rf ~/spl_tmp
    ssh ${host_username[i]}@${hosts[i]} mkdir -p ~/${spl_id}_spl_docker/apps
    rsync -avzhP $SPARKLORD_HOME/config_files/tmp/${spl_id}/ ${host_username[i]}@${hosts[i]}:~/${spl_id}_spl_docker/apps
    rsync -avzhP $SPARKLORD_HOME/config_files/tmp/${spl_id}/ ${host_username[i]}@${hosts[i]}:~/${spl_id}_spl_docker/apps

    container_list=$(ssh ${host_username[i]}@${hosts[i]} 'docker ps -a --filter "name=${spl_id}" -q')
    for c_list in $container_list; do
        ssh ${host_username[i]}@${hosts[i]} "docker cp ~/${spl_id}_spl_docker/apps/core-site.xml ${c_list}:/hadoop/etc/hadoop/core-site.xml"
        ssh ${host_username[i]}@${hosts[i]} "docker cp ~/${spl_id}_spl_docker/apps/hdfs-site.xml ${c_list}:/hadoop/etc/hadoop/hdfs-site.xml"
        ssh ${host_username[i]}@${hosts[i]} "docker cp ~/${spl_id}_spl_docker/apps/yarn-site.xml ${c_list}:/hadoop/etc/hadoop/yarn-site.xml"
        ssh ${host_username[i]}@${hosts[i]} "docker cp ~/${spl_id}_spl_docker/apps/mapred-site.xml ${c_list}:/hadoop/etc/hadoop/mapred-site.xml"
    done
END

done
