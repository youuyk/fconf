import xml.etree.ElementTree as elemTree
import fcntl
import subprocess
import pickle
import random
import string
import time
import yaml
import os

def to_run():
    with open(os.environ['SPARKLORD_HOME']+'/run', 'r') as f:
        running = f.read()

    if running.strip() != "1":
        return None
                
    lock = open(os.environ['SPARKLORD_HOME']+'/logs/conf.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)

    if not os.path.exists(os.environ['SPARKLORD_HOME']+'/logs/completed_list'):
        open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'w').close()
        
    with open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'r') as f:
        cv = f.readlines()
    cv = [i.strip() for i in cv]

    cp_list = os.listdir(os.environ['SPARKLORD_HOME']+'/checkpoints')
    cp_list.sort()
    cp_run = None
    for cp in cp_list:
        if not cp.startswith('20'): continue
        #cp = os.environ['SPARKLORD_HOME']+'/checkpoints/' + cp 

        if cp in cv:
            continue
        
        print(f'Load data from {cp}')
        cp_run = cp
        break

    with open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'a') as f:
        f.write(f'{cp_run}\n')
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return cp_run

def get_ex_data(cp):
    with open(os.environ['SPARKLORD_HOME']+'/checkpoints/' + cp + '/LogInformation.txt', 'r') as f:
        data = f.readlines()
    result = {}
    for d in data:
        d = d.split('=')
        result[d[0]] = d[1].strip()
    
    return result
    
        
def read_all_checkpoints():
    spl_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    next_cp = to_run()
    
    while next_cp != None:
        ex_data = get_ex_data(next_cp)
        
        if ex_data['config_key'] == 'hadoop.tmp.dir':
            next_cp = to_run()
            continue

        docker_conf_path = os.environ['SPARKLORD_HOME'] + f'/config_files/tmp/{spl_id}/' + ex_data['config_file_name']

        with open(os.environ['SPARKLORD_HOME'] + '/config_files/config_type.yaml') as f:
            conf_class = yaml.load(f, Loader=yaml.FullLoader)[f'{os.environ["app"]}-config']

        out = subprocess.run([os.environ['SPARKLORD_HOME']+'/modules/deployer/run_fi.sh', spl_id, os.environ['SPL_JOB'], docker_conf_path, ex_data['config_key'], ex_data['config_value']])
        if out.returncode != 0: exit(1)
        
        next_cp = to_run()
        
if __name__ == '__main__':
    
    if 'SPARKLORD_HOME' not in os.environ.keys():
        print('SPARKLORD_HOME is not in env')
        print(f'run export SPARKLORD_HOME=[sparklord dir] first')
        exit(1)
    
    with open(os.environ['SPARKLORD_HOME'] + '/scenario.yaml') as f:
        scenario = yaml.load(f, Loader=yaml.FullLoader)['config']

    read_all_checkpoints()
