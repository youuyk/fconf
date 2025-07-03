#!/bin/bash

. ./config.sh

worker_count=$((${#hosts[*]} * $slave_count))
LOG_DIR=$1
APPLICATION=$2
spl_id=$3
# whether use yarn or not 
master_mode=${master_mode}

CLASS_NAME=$(yq -e ".spark.\"${APPLICATION}\".class" $SPARKLORD_HOME/modules/jobs/args.yaml 2>&1)
if [ $? -ne 0 ] ; then
    CLASS_NAME="org.apache.spark.examples.$APPLICATION"
fi

APP_ARGS=$(yq -e ".spark.\"${APPLICATION}\".args" $SPARKLORD_HOME/modules/jobs/args.yaml 2>&1)
if [ $? -ne 0 ] ; then
    APP_ARGS=""
fi

JAR_PATH=$(yq -e ".spark.\"${APPLICATION}\".jar_path" $SPARKLORD_HOME/modules/jobs/args.yaml 2>&1)
if [ $? -ne 0 ] ; then
    JAR_PATH="/spark/examples/jars/spark-examples*.jar /spark/examples/jars/scopt_*"
fi

TIMEOUT=$(yq -e ".spark.\"${APPLICATION}\".timeout" $SPARKLORD_HOME/modules/jobs/args.yaml 2>&1)
if [ $? -ne 0 ] ; then
    TIMEOUT="90s"
fi

DEPLOY_MODE=$(yq -e ".spark.\"${APPLICATION}\".deploy_mode" $SPARKLORD_HOME/modules/jobs/args.yaml 2>&1)
if [ $? -ne 0 ] ; then
    DEPLOY_MODE="cluster"
fi

# if deploy mode is yarn (1)  -> run with yarn, 
# if not, run without yarn (0)

if [ $master_mode -eq 1 ] ; then
    echo "running $APPLICATION"
    start=`date +%s.%N`
    
    timeout $TIMEOUT docker exec -t ${spl_id}_spl_master bash -c "/spark/bin/spark-submit --class $CLASS_NAME --master yarn --deploy-mode $DEPLOY_MODE --driver-memory 4g --executor-memory 2g --executor-cores 2 --num-executors $worker_count --jars $JAR_PATH $APP_ARGS"
    IS_TIMEOUT=$?
    finish=`date +%s.%N`
    echo "Done!!"
fi 

diff=$( echo "$finish - $start" | bc -l )
echo 'time:' $diff
echo 'time:' $diff >> ${LOG_DIR}/sparkresult.txt

if [ $IS_TIMEOUT -eq 124 ] ; then
    IS_TIMEOUT=1
else
    IS_TIMEOUT=0
fi
echo 'timeout:' $IS_TIMEOUT >> ${LOG_DIR}/sparkresult.txt
