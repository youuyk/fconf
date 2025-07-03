#!/bin/bash

. ./config.sh

set +H
# if SPARKLORD_MODE == config
# usage: ./run_fi.sh job_name file_name conf_key conf_value
# example ./run_fi.sh sparkals.sh $SPARKLORD_HOME/worker/config_files/core-site.xml hadoop.security.authorization true

# if SPARKLORD_MODE == resource
# usage: ./run_fi.sh job_name cpu_count cpu_util memory
# example ./run_fi.sh sparkals.sh 1 60 2g

byte_to_number(){
    num=$1
    unit=${num: (-2):1}
    result=${num: :-2}
    if [ $unit == "G" ]; then
        result=$(echo "$result*1000000000" | bc | awk -F "." '{print $1}')
    elif [ $unit == "M" ]; then
        result=$(echo "$result*1000000" | bc | awk -F "." '{print $1}')
    elif [ $unit == "k" ]; then
        result=$(echo "$result*1000" | bc | awk -F "." '{print $1}')
    else
        result=${num: :-1}
    fi
    echo $result
}

ip_mutex_lock()
{
    if [ $OVERLAP_IP -eq 1 ]; then
        return 0
    fi
    IP_LOCK_FILE=$SPARKLORD_HOME/logs/ip.lock
    exec 100>$IP_LOCK_FILE
    flock -x 100 || exec 1
    echo "lock start - lock"
}

ip_mutex_release()
{
    if [ $OVERLAP_IP -eq 1 ]; then
        return 0
    fi
    flock -u 100
    exec 100>&-
    echo "lock end"
}

run_spark_and_save_logs()
{
    start_cpu=0
    start_netI=0
    start_netO=0
    start_blockI=0
    start_blockO=0
    for((i=0; i<${#hosts[*]}; i++))
    do
      
        container_list=$(ssh ${host_username[i]}@${hosts[i]} docker ps -a --filter "name=${spl_id}" -q)
        for c_list in $container_list; do
            tmp_cpu=$(ssh ${host_username[i]}@${hosts[i]} docker exec -t $c_list cat /proc/stat | head -1 | awk '{print $2 + $3 + $4}')
            tmp_block=($(ssh ${host_username[i]}@${hosts[i]} docker stats --no-stream --format "{{.BlockIO}}" $c_list | tr "/" " "))
            tmp_net=($(ssh ${host_username[i]}@${hosts[i]} docker stats --no-stream --format "{{.NetIO}}" $c_list | tr "/" " "))
            tmp_blockI=$(byte_to_number ${tmp_block[0]})
            tmp_blockO=$(byte_to_number ${tmp_block[1]})
            tmp_netI=$(byte_to_number ${tmp_net[0]})
            tmp_netO=$(byte_to_number ${tmp_net[1]})
            start_cpu=$(($start_cpu + $tmp_cpu))
            start_blockI=$(($start_blockI + $tmp_blockI))
            start_blockO=$(($start_blockO + $tmp_blockO))
            start_netI=$(($start_netI + $tmp_netI))
            start_netO=$(($start_netO + $tmp_netI))
        done
    done

    ./config.sh
    app_type=$app 

    # run spark application
    $SPARKLORD_HOME/modules/jobs/${app}/run_job.sh $LOG_DIR ${job} $spl_id



    echo Saving logs, stats
    end_cpu=0
    end_netI=0
    end_netO=0
    end_blockI=0
    end_blockO=0
    for((i=0; i<${#hosts[*]}; i++))
    do
        container_list=$(ssh ${host_username[i]}@${hosts[i]} docker ps -a --filter "name=${spl_id}" -q)
        for c_list in $container_list; do
            tmp_cpu=$(ssh ${host_username[i]}@${hosts[i]} docker exec -t $c_list cat /proc/stat | head -1 | awk '{print $2 + $3 + $4}')
            tmp_block=($(ssh ${host_username[i]}@${hosts[i]} docker stats --no-stream --format "{{.BlockIO}}" $c_list | tr "/" " "))
            tmp_net=($(ssh ${host_username[i]}@${hosts[i]} docker stats --no-stream --format "{{.NetIO}}" $c_list | tr "/" " "))
            tmp_blockI=$(byte_to_number ${tmp_block[0]})
            tmp_blockO=$(byte_to_number ${tmp_block[1]})
            tmp_netI=$(byte_to_number ${tmp_net[0]})
            tmp_netO=$(byte_to_number ${tmp_net[1]})
            end_cpu=$(($end_cpu + $tmp_cpu))
            end_blockI=$(($end_blockI + $tmp_blockI))
            end_blockO=$(($end_blockO + $tmp_blockO))
            end_netI=$(($end_netI + $tmp_netI))
            end_netO=$(($end_netO + $tmp_netI))
        done
    done

  
    echo "cpu: $(($end_cpu - $start_cpu))" >> ${LOG_DIR}/sparkresult.txt
    echo "netIO: $(($end_netI - $start_netI)) $(($end_netO - $start_netO))" >> ${LOG_DIR}/sparkresult.txt
    echo "blockIO: $(($end_blockI - $start_blockI)) $(($end_blockO - $start_blockO))" >> ${LOG_DIR}/sparkresult.txt


    for((i=0; i<${#hosts[*]}; i++))
    do
        ssh ${host_username[i]}@${hosts[i]} "bash -s $spl_id" < $SPARKLORD_HOME/deploy/${app}/save_logs.sh
    done

   
    $SPARKLORD_HOME/modules/jobs/${app}/get_app_status.sh ${LOG_DIR}/sparkresult.txt ${job} $spl_id

  
    mv $HOME/spl_tmp/${spl_id}_spl_master $LOG_DIR
    mkdir -p $HOME/spl_tmp/${spl_id}_spl_master $LOG_DIR

  
    for((i=0; i<${#hosts[*]}; i++))
    do
        rsync -avzhP ${host_username[i]}@${hosts[i]}:~/spl_tmp/${spl_id}* $LOG_DIR
    done

 
    STAT_INFO=$(sed 's/$/\\n/g' $LOG_DIR/LogInformation.txt)
    STAT_INFO=$(echo $STAT_INFO | sed 's/ //g')
    sed -i "1 i$STAT_INFO" $LOG_DIR/*/*

}

save_run_value(){

    SAVED_FILE=$SPARKLORD_HOME/logs/saved_list 
    LOCK_FILE=$SPARKLORD_HOME/logs/saved.lock
    exec 200>$LOCK_FILE 
    flock -x 200 || exit 1 

    echo "saving ${conf_key} ${conf_value} in ${SAVED_FILE}" 
    echo ${conf_key} ${conf_value} >> $SAVED_FILE 
    echo "finish ${conf_key} ${conf_key} in ${SAVED_FILE}" 

    flock -u 200 
    exec 200>&-
}

remove_running_value(){

    SAVED_FILE=$SPARKLORD_HOME/logs/running_list
    LOCK_FILE=$SPARKLORD_HOME/logs/running.lock
    exec 201>$LOCK_FILE 
    flock -x 201 || exit 1 

    sed -i "${conf_key} ${conf_value}/d" $SAVED_FILE  
    
    flock -u 201 
    exec 201>&- 

}
spl_id=$1

ip_name=$(ip route | awk '/default/ {print $5; exit}')
myip=$(ip route | grep $ip_name | grep src | awk '{print $9; exit}')

TIMESTAMP=$(date +%Y%m%d%H%M)
TIME_NANO=$(date +%s%N)
VM_NAME=$spl_id
master="sparkmaster"

cpu_count=2
cpu_util=`expr 100 \* $cpu_count`
memory=4g

# file to write information about experiment
INFO_FILE=LogInformation.txt

job=$2

mutation_idx=$6

echo "Make folder"
mkdir -p $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}

mkdir -p $SPARKLORD_HOME/config_files/tmp/${spl_id}/
 
if [ $app = "spark" ]; 
then
    cp $SPARKLORD_HOME/config_files/spark/* $SPARKLORD_HOME/config_files/tmp/${spl_id}/
    cp $SPARKLORD_HOME/config_files/hadoop/* $SPARKLORD_HOME/config_files/tmp/${spl_id}/
fi

echo program=${job} >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE

case $SPARKLORD_MODE in
    config)
  
    file_name=$3
   
    conf_key=$4
  
    conf_value=$5
    
   
    echo SPARKLORD_MODE=CONFIG_INJECTION >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo cpu_cores=$cpu_count >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo cpu_util=$cpu_util >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo memory=$memory >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo config_file_name=$(echo ${file_name} | awk -F '/' '{ print $NF }') >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo config_key=${conf_key} >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo config_value=${conf_value} >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE

    # modify config to specific value
    python3 $SPARKLORD_HOME/modules/config_modifier/conf_modifier.py $file_name $conf_key "$conf_value"
    ;;
    
    resource)
   
    cpu_count=$3
    cpu_util=$4
    memory=$5

   
    echo SPARKLORD_MODE=RESOURCE_INJECTION >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo cpu_cores=$cpu_count >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo cpu_util=$cpu_util >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo memory=$memory >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE

    ;;

    none|normal_run)
    
    echo cpu_cores=$cpu_count >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo cpu_util=$cpu_util >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
    echo memory=$memory >> $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO}/$INFO_FILE
esac

cpu_cores=$(($cpu_count * $slave_count + $cpu_count))
my_cores=$(grep -c processor /proc/cpuinfo)

conf_value=$(echo "$conf_value" | sed 's/\//%/g' | sed 's/\s*//g')


LOG_DIR=$SPARKLORD_HOME/logs/${TIMESTAMP}_ver${log_version}_c${cpu_count}u${cpu_util}m${memory}_$(echo ${file_name} | awk -F '/' '{ print $NF }')_${conf_key}_${conf_value:0:50}_${job}_${spl_id}
mv $SPARKLORD_HOME/logs/${VM_NAME}_${TIME_NANO} $LOG_DIR


if [ $(docker ps -a --filter "name=${spl_id}" -q | wc -l) -le 1 ]; then
    echo "run container" 
    $SPARKLORD_HOME/deploy/run_container.sh $cpu_count $cpu_util $memory $spl_id

   
    if [ $? -eq 30 ]; then
        exit 1
    fi
fi


if [ $SPARKLORD_RUNMODE == "run" ]; then
    run_spark_and_save_logs
    #SAVED_FILE=$SPARKLORD_HOME/logs/saved_list 
    #LOCK_PATH=$SPARKLORD_HOME/logs/saved.lock
    #python3 $SPARKLORD_HOME/modules/deployer/running_check.py "$conf_key" "$conf_value" "$mutation_idx" "$SAVED_FILE" "$LOCK_PATH"
    #save_run_value
    #remove_running_value
   
elif [ $SPARKLORD_RUNMODE == "prepare" ]; then
    for((i=0; i<${#hosts[*]}; i++))
    do
        ssh root@${hosts[i]} "bash -s $spl_id" < $SPARKLORD_HOME/deploy/${app}/save_instance.sh
    done
    mv $HOME/spl_tmp/${spl_id}_spl_master $LOG_DIR

    
    for((i=0; i<${#hosts[*]}; i++))
    do
        if [ ${hosts[i]} == "127.0.0.1" ]; then
            cp -rf ~/spl_tmp/${spl_id}* $LOG_DIR
            cp -rf /tmp/${spl_id}_tar/cp_* $LOG_DIR
        else
            rsync -avzhP ${host_username[i]}@${hosts[i]}:~/spl_tmp/${spl_id}* $LOG_DIR
            rsync -avzhP ${host_username[i]}@${hosts[i]}:/tmp/${spl_id}_tar/cp_* $LOG_DIR
        fi
        ssh root@${hosts[i]} rm -rf /tmp/${spl_id}
        ssh root@${hosts[i]} rm -rf /tmp/${spl_id}_tar
    done
elif [ $SPARKLORD_RUNMODE == "deploy" ]; then
    exit 1

elif [ $SPARKLORD_RUNMODE == "loadnrun" ]; then
    $SPARKLORD_HOME/deploy/${app}/config_injection.sh $spl_id

    cp_dir=`ls ${SPARKLORD_HOME}/checkpoints | grep -Fe "${conf_key}_${conf_value:0:50}" | sed 's/\\\$/\\\\\$/'`

    for((i=0; i<${#hosts[*]}; i++))
    do
        ssh root@${hosts[i]} "bash -s ${spl_id} ${SPARKLORD_HOME}/checkpoints/${cp_dir}" < ${SPARKLORD_HOME}/deploy/spark/restore_checkpoint.sh
        ssh root@${hosts[i]} rm -rf /tmp/ctrd*
    done
    run_spark_and_save_logs

fi

if [ $SPARKLORD_RUNMODE != "loadnrun" ]; then
    ip_mutex_lock
    for((i=0; i<${#hosts[*]}; i++))
    do
        old_ip=($(ssh ${host_username[i]}@${hosts[i]} "bash -s $spl_id" < $SPARKLORD_HOME/deploy/erase_container.sh))
        echo $old_ip

        if [ $OVERLAP_IP -eq 1 ]; then
            ip_file=$SPARKLORD_HOME/resource/ip_list
        else
            ip_file=$SPARKLORD_HOME/logs/ip_list
        fi

        for((ip_idx=0; ip_idx<${#old_ip[*]}; ip_idx++))
        do
            echo remove ${old_ip[ip_idx]} from ip list
            sed -i "/${old_ip[ip_idx]}/d" $ip_file
        done
    done
    ip_mutex_release
fi

rm $SPARKLORD_HOME/config_files/tmp/${spl_id}/*.xml

cont_list=($(ls $LOG_DIR | grep $spl_id))
for((i=0; i<${#cont_list[*]}; i++))
do
    newname=`echo ${cont_list[i]} | sed "s/${spl_id}_//"`
    mv ${LOG_DIR}/${cont_list[i]} "${LOG_DIR}/${newname}"
done
