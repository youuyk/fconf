#!/bin/bash
source ./config.sh

spl_id=$1

$SPARKLORD_HOME/deploy/${app}/config_injection.sh $spl_id

timeout 5m docker exec -t ${spl_id}_spl_master bash -c "/hadoop/bin/hadoop namenode -format"
timeout 10m docker exec ${spl_id}_spl_master bash -c "/hadoop/sbin/start-all.sh"

if [ $INPUT_DATA -eq 1 ]; then
    timeout 1m docker exec -t ${spl_id}_spl_master sh -c "/hadoop/bin/hdfs dfs -put /spark/file.txt /file.txt"
    timeout 1m docker exec -t ${spl_id}_spl_master sh -c "/hadoop/bin/hdfs dfs -put /spark/cluster.txt /cluster.txt"
    timeout 1m docker exec -t ${spl_id}_spl_master sh -c "/hadoop/bin/hdfs dfs -put /spark/page.txt /page.txt"
    timeout 1m docker exec -t ${spl_id}_spl_master sh -c "/hadoop/bin/hdfs dfs -put /spark/lr_test.txt /lr_test.txt"

    timeout 1m docker exec -t ${spl_id}_spl_master sh -c "/hadoop/bin/hdfs dfs -mkdir -p /user/root/"
    timeout 1m docker exec -t ${spl_id}_spl_master sh -c "/hadoop/bin/hdfs dfs -put /spark/data /user/root"
    timeout 1m docker exec -t ${spl_id}_spl_master sh -c "/hadoop/bin/hdfs dfs -put /spark/examples /user/root/"
fi
