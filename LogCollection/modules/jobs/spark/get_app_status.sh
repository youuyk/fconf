#!/bin/bash

. ./config.sh

STAT_FILE=$1
APPLICATION=$2
spl_id=$3

CLASS_NAME=$(yq -e ".spark.$APPLICATION.class" $SPARKLORD_HOME/modules/jobs/args.yaml 2>&1)
if [ $? -ne 0 ] ; then
    CLASS_NAME="org.apache.spark.examples.$APPLICATION"
fi

#APP_RESULT=$(timeout 10s docker exec -t ${spl_id}_spl_master bash -c "/hadoop/bin/yarn application -list -appStates ALL" | grep $CLASS_NAME)
APP_RESULT=$(timeout 10s docker exec -t ${spl_id}_spl_master bash -c "/hadoop/bin/yarn application -list -appStates ALL")

echo 'result:' $APP_RESULT >> ${STAT_FILE}
