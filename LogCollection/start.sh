#!/bin/bash

# $application $job $worker_count $iter_count [log_dir] 

if [ -z "`ls | grep config_data.sh`" ] || [ "`stat -c '%y' config.yaml`" \> "`stat -c '%y' config_data.sh`" ]; then
    ./build_config.sh
fi

source ./config.sh


ls ~/.ssh/id_rsa > /dev/null 2>&1
if [ $? -ne 0 ]; then
    ssh-keygen -f ~/.ssh/id_rsa -N ""
fi

for ((i = 0; i < ${#hosts[*]}; i++))
do
ssh-keygen -H -F ${hosts[$i]} > /dev/null 2>&1
if [ $? -ne 0 ]; then
echo ${host_username[$i]}@${hosts[$i]}
/usr/bin/expect <<EOE
set prompt "#"
spawn bash -c "ssh-copy-id ${host_username[$i]}@${hosts[$i]}"
expect {
"yes/no" { send "yes\r"; exp_continue}
-nocase "password" {send "${host_password[$i]}\r"; exp_continue }
$prompt
}
EOE
/usr/bin/expect <<EOE
set prompt "#"
spawn bash -c "ssh-copy-id root@${hosts[$i]}"
expect {
"yes/no" { send "yes\r"; exp_continue}
-nocase "password" {send "${host_password[$i]}\r"; exp_continue }
$prompt
}
EOE
fi
done

# deploy docker container
if [ ${BUILD_IMAGE} == 1 ]; then
    # docker image build 
    $SPARKLORD_HOME/deploy/build_all.sh $app $SPARKLORD_HOME
fi
#send docker files and build
for((i=0; i<${#hosts[*]}; i++))
do
    ssh ${host_username[i]}@${hosts[i]} mkdir -p ~/spl_docker/apps
    #echo "remove old containers in ${hosts[i]}"
    #ssh ${host_username[i]}@${hosts[i]} sh < $SPARKLORD_HOME/deploy/erase_container.sh

    echo "setting docker network in ${hosts[i]}"
    ssh ${host_username[i]}@${hosts[i]} "bash -s $SPARKLORD_NETWORK" < $SPARKLORD_HOME/deploy/create_network.sh
    
    if [ "${hosts[i]}" == "127.0.0.1" ] || [ "${hosts[i]}" == "localhost" ]; then
        continue
    fi
    
    rsync -avzhP $SPARKLORD_HOME/apps/${app}_master ${host_username[i]}@${hosts[i]}:~/spl_docker/apps
    rsync -avzhP $SPARKLORD_HOME/apps/${app}_slave ${host_username[i]}@${hosts[i]}:~/spl_docker/apps
    if [ ${BUILD_IMAGE} == 1 ]; then
        ssh ${host_username[i]}@${hosts[i]} 'bash -s' < $SPARKLORD_HOME/deploy/build_all.sh $app ~/spl_docker
    fi
done

mkdir -p $SPARKLORD_HOME/workers
mkdir -p $SPARKLORD_HOME/resource/cpu

# copy config files to spark master
cp $SPARKLORD_HOME/config_files/$app/* $SPARKLORD_HOME/apps/${app}_master

if [ $SPARKLORD_RUNMODE == "loadnrun" ]; then
    python3 $SPARKLORD_HOME/modules/deployer/loaded.py
else
    python3 $SPARKLORD_HOME/modules/deployer/${SPARKLORD_MODE}_fi.py
fi
