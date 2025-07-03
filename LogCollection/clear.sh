#!/bin/bash

. ./config.sh

echo "starting erasing container"
$SPARKLORD_HOME/deploy/erase_container.sh
echo "erase container"
rm -rf $SPARKLORD_HOME/workers
rm -rf $SPARKLORD_HOME/resource
rm -rf ~/spl_tmp
rm -rf ~/spl_docker
rm -rf $SPARKLORD_HOME/config_files/tmp
rm $SPARKLORD_HOME/logs/ip_list
rm $SPARKLORD_HOME/logs/running_list  
rm $SPARKLORD_HOME/logs/completed_list  
rm $SPARKLORD_HOME/logs/saved_list 
