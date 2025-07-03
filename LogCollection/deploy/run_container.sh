#!/bin/bash
source ./config.sh

get_newip()
{
    if [ $SPARKLORD_NETWORK == "ipvlan" ]; then 
        tmp=($(echo $1 | tr "." "\n"))
    elif [ $SPARKLORD_NETWORK == "bridge" ]; then
        tmp=($(echo "172.20.1.0" | tr "." "\n"))
    fi

    if [ $OVERLAP_IP -eq 1 ]; then
        ip_file=$SPARKLORD_HOME/resource/ip_list
    else
        ip_file=$SPARKLORD_HOME/logs/ip_list
    fi

    ip_list=$(cat $ip_file 2> /dev/null)
    for i in {10..254}
    do
        cur_ip=${tmp[0]}.${tmp[1]}.${tmp[2]}.$i
        echo $ip_list | grep $cur_ip > /dev/null
        if [ $? -ne 0 ]; then
            echo $cur_ip >> $ip_file
            echo $cur_ip
            break
        fi
    done
}

ip_mutex_lock()
{
    if [ $OVERLAP_IP -eq 1 ]; then
        return 0
    fi
    IP_LOCK_FILE=$SPARKLORD_HOME/logs/ip.lock
    exec 100>$IP_LOCK_FILE
    flock -x 100 || exec 1
    echo "lock start"
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

allocate_cpu()
{
    count=$1
    core_size=$(grep -c processor /proc/cpuinfo)
    cpu_allocated=$(ls $SPARKLORD_HOME/resource/cpu)
    result=( )
    for(( i=0; i<$core_size; i++ )); do
        if [ "$(echo $cpu_allocated | grep `printf "%02d" $i`)" == "" ]; then
            result+=( $i )
            touch $SPARKLORD_HOME/resource/cpu/`printf "%02d" $i`
            if [ ${#result[*]} -eq $count ]; then
                break
            fi
        fi
    done
    if [ ${#result[*]} -eq $count ]; then
        cpu_list=$(printf ",%s" "${result[@]}")
        echo ${cpu_list:1}
    else
        echo "-1"
    fi
}   

ip_name=$(ip route | awk '/default/ {print $5; exit}')
gateway=$(ip route | awk '/default/ {print $3; exit}')
subnet=$(ip route | grep $ip_name | grep src | awk '{print $1; exit}')
myip=$(ip route | grep $ip_name | grep src | awk '{print $9; exit}')

# docker network rm spark_network > /dev/null 2>&1
# docker network create -d ipvlan --subnet=$subnet --gateway=$gateway -o parent=$ip_name spark_network

cpu_count=$1
cpu_util=$2
memory=$3
spl_id=$4

rm $SPARKLORD_HOME/workers/${spl_id} > /dev/null 2>&1

ip_mutex_lock
master_ip=$(get_newip $myip)
rm -rf $HOME/spl_tmp/${spl_id}*
mkdir -p $HOME/spl_tmp/${spl_id}_spl_master
cpu_list=$(allocate_cpu $cpu_count)
echo $cpu_list
if [ $cpu_list == "-1" ]; then
    exit 30
fi
 
docker run --cap-add=NET_ADMIN --privileged=true --security-opt=seccomp:unconfined -t -d -h master --restart=no -v $HOME/spl_tmp:/opt --ip=$master_ip --cpuset-cpus="$cpu_list" --cpu-period=100000 --cpu-quota=$((1000 * $cpu_util)) --memory=$memory --name ${spl_id}_spl_master --network spark_network ${app}_master

ssh_key=$(docker exec -it ${spl_id}_spl_master cat /root/.ssh/id_rsa.pub)
s_id=0

all_ip=( $master_ip )
all_hosts=()

s_id=0
for((i=0; i<${#hosts[*]}; i++))
do
    ssh ${host_username[i]}@${hosts[i]} "rm -rf \$HOME/spl_tmp/${spl_id}_spl_slave*"
    for((ip=0; ip<$slave_count; ip++))
    do
        echo slave$s_id
        s_ip=$(get_newip $myip)
        echo $s_ip

        is_local=0
        if [ "${hosts[i]}" == "127.0.0.1" ] || [ "${hosts[i]}" == "localhost" ]; then
            is_local=1
        fi

        echo $s_ip >> $SPARKLORD_HOME/workers/${spl_id}
        
        ssh ${host_username[i]}@${hosts[i]} "mkdir -p \$HOME/spl_tmp/${spl_id}_spl_slave${s_id}"

     
        cpu_list=$(allocate_cpu $cpu_count)
        if [ $cpu_list == "-1" ]; then
            echo "no cpu" 
            exit 30
        fi

       
        ssh ${host_username[i]}@${hosts[i]} "docker run --privileged=true --cap-add=NET_ADMIN --security-opt=seccomp:unconfined -t -d -h slave$s_id --restart=no --ip $s_ip --cpuset-cpus=\"$cpu_list\" --cpu-period=100000 --cpu-quota=$((1000 * $cpu_util)) --memory=$memory -v $HOME/spl_tmp:/opt --name ${spl_id}_spl_slave${s_id} --network spark_network ${app}_slave"
        sleep 0.5s

   
        ssh ${host_username[i]}@${hosts[i]} "docker exec -t ${spl_id}_spl_slave${s_id} sh -c \"echo $ssh_key > /root/.ssh/authorized_keys\""
        ssh ${host_username[i]}@${hosts[i]} "docker exec -t ${spl_id}_spl_slave${s_id} sh -c \"echo $master_ip master >> /etc/hosts\""
        all_hosts+=("$s_ip slave$s_id")

        s_id=$(($s_id + 1))
    done
done
ip_mutex_release

for((i=0; i<${#hosts[*]}; i++))
do
    spl_list=($(ssh ${host_username[i]}@${hosts[i]} "docker ps -aq -f name=${spl_id}"))
    for((id=0; id<${#spl_list[*]}; id++))
    do
        for((iter=0; iter<${#all_hosts[*]}; iter++))
        do
            ssh ${host_username[i]}@${hosts[i]} "docker exec -t ${spl_list[id]} sh -c \"echo ${all_hosts[iter]} >> /etc/hosts\""
        done
    done
done

docker exec -t ${spl_id}_spl_master bash -c "rm /hadoop/etc/hadoop/workers"

docker cp ${SPARKLORD_HOME}/workers/${spl_id} ${spl_id}_spl_master:/hadoop/etc/hadoop/workers
docker cp ${SPARKLORD_HOME}/config_files/hadoop/ex_workers ${spl_id}_spl_master:/hadoop/etc/hadoop/ex_workers 

if [ $SPARKLORD_RUNMODE != "loadnrun" ]; then  
    echo "init cluster" 
    $SPARKLORD_HOME/deploy/${app}/init_cluster.sh $spl_id
fi
