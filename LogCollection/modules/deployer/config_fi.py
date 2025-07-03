import xml.etree.ElementTree as elemTree
import fcntl
import subprocess
import pickle
import random
import string
import time
import yaml
import os
import shutil
    
def remove_folder(path, conf_key, conf_value):
    directory = os.listdir(path)
    for dirc in directory:
        if f"{conf_key}_{conf_value}_" in dirc:
            shutil.rmtree(f"{path}{dirc}")
            print(f"remove {path}{dirc}")

def get_value_candidate(conf_type, default_value):
 
    can_val = scenario[conf_type]
    can_val = [(-1, x) for x in can_val]
    mutation_idx = 1
    if default_value != None:
     
        default_value = ','.join(default_value.split('\n'))
    if default_value != None and default_value != '':
        mutation = list(default_value)
        for _ in range(2):
            idx = int(random.random()*100) % len(default_value)
            mutation[idx] = chr(ord(mutation[idx])+1)
        can_val.append((mutation_idx, ''.join(mutation)))

        if conf_type == 'ip' or conf_type == 'ips':
            ip_port = default_value.split(':')
            if ip_port[-1].isdigit():
                ip_port[-1] = '9999'
                can_val.append((-1, ':'.join(ip_port)))
                ip_port[-1] = '70000'
                can_val.append((-1, ':'.join(ip_port)))
    return can_val

def get_value_candidate_spark(conf_name):

    can_val = scenario[conf_name]
    can_val = [(-1, x) for x in can_val]
    return can_val

def can_run(conf, val, mutation_idx):
    ret = True

    #print(f'Checking {conf} {val} in completed_list...')
    lock = open(os.environ['SPARKLORD_HOME']+'/logs/conf.lock', 'w')
    #print('opened lock file')
    fcntl.flock(lock, fcntl.LOCK_EX)
    #print(f'start config lock')

    if mutation_idx != -1:
        val = f'mutation_{mutation_idx}'
    
    if not os.path.exists(os.environ['SPARKLORD_HOME']+'/logs/completed_list'):
        open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'w').close()
        
    with open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'r') as f:
        cv = f.readlines()
    cv = [i.strip() for i in cv]
    
    if f'{conf} {val}' not in cv:
        with open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'a') as f:
            f.write(f'{conf} {val}\n')
        #print(f'{conf} {val} is not in completed list\nrun {conf} {val}')
    else:
    
        ret = False

    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    #print(f'end config lock')
    return ret

def saved_run(conf, val, mutation_idx):
    
    ret = True     
    saved_list = os.environ['SPARKLORD_HOME'] + '/logs/saved_list'
    lock_path = os.environ['SPARKLORD_HOME'] + '/logs/saved.lock'
    
    lock = open(lock_path, "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
     
    if not os.path.exists(saved_list):
        open(saved_list, 'w').close()

    with open(saved_list, 'r') as fr:
        saved = fr.readlines()
    saved = [i.strip() for i in saved]


    if mutation_idx != -1:
        val = f"mutation_{mutation_idx}"
    if f"{conf} {val}" in saved:
        ret = False
    
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close() 

    return ret

def check_running_conf(conf, val, mutation_idx):
    ret = True 
    running_list = os.environ['SPARKLORD_HOME'] + '/logs/running_list'
    lock_path = os.environ['SPARKLORD_HOME'] + '/logs/running.lock'
    
    lock = open(lock_path, 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)

    if not os.path.exists(running_list):
        open(running_list, 'w').close() 
            
    with open(running_list, 'r') as fr:
        running_conf = fr.readlines()
    
    running_conf = [i.strip() for i in running_conf]

    if mutation_idx != -1:
        val = f"mutation_{mutation_idx}"
        
    if f"{conf} {val}" not in running_conf:
        with open(running_list, 'a') as fw:
            fw.write(f"{conf} {val}\n")    
    else:
         
        ret = False 
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return ret 

def save_run_value(conf_name, conf_val, mutation_idx, file_path, lock_path):
    
    lock = open(lock_path, "w")
    fcntl.flock(lock, fcntl.LOCK_EX)

    if mutation_idx != "-1":
        conf_val = f"mutation_{mutation_idx}"
    with open(file_path, "a") as fw:
        fw.write(f"{conf_name} {conf_val}\n")
    
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()

def getConfType(confName, configTypePath):
    
    with open(configTypePath) as f:
        configType = yaml.load(f, Loader = yaml.FullLoader)
    
    configType = configType['spark-config']
    return configType[confName]

def strip(x):
    return x.strip()


def read_prop_and_run(file, spl_id):
    path = os.environ['SPARKLORD_HOME'] + '/config_files/' + os.environ['app'] + '/' + file
    file_type = file.split(".")[-1]
    if file_type == "xml":
        tree = elemTree.parse(path)
        tree = list(tree.getroot())
    elif file_type == "conf":
        with open(path, "r") as fr:
            tree = fr.readlines()
            tree = list(map(strip, tree))
    docker_conf_path = os.environ['SPARKLORD_HOME'] + f'/config_files/tmp/{spl_id}/' + file
     
    with open(os.environ['SPARKLORD_HOME'] + f'/config_files/config_type_{os.environ["app"]}.yaml') as f:
        conf_class = yaml.load(f, Loader=yaml.FullLoader)[f'{os.environ["app"]}-config']
    
    # load configuration of running application 
    with open(os.environ['SPARKLORD_HOME'] + f'/app_config/{os.environ["SPL_JOB"]}_default.conf', "r") as fr:
        app_config = fr.readlines()
    app_config = list(map(strip, app_config)) 

    # read all property and modify each prop value
    for prop in tree:
        #info = {'value':None}
        conf = prop.split(" ")
        conf_name = conf[0]
        if len(conf) == 2:
            conf_val = conf[1]
        else:
            conf_val = None 
        
        # if conf_name not in application_conf, then skip 
        if conf_name not in app_config:
            #print(f"This configuration is not used in this application: {conf_name}") 
            continue 

        if conf_name not in conf_class:
            continue  
        if conf_name not in scenario:
            continue  
        if os.environ['scenario_type'] == "spark": 
            candidate = get_value_candidate_spark(conf_name) 
        else: 
            candidate = get_value_candidate(conf_class[conf_name], conf_val)
        
        #print(conf_name)
        #print(candidate) 
        cnt = 0 
        for mutation_idx, i in candidate:
            
            with open(os.environ['SPARKLORD_HOME']+'/run', 'r') as f: 
                running = f.read()  
            if running.strip() != "1": 
                break
        
            if can_run(conf_name, str(i), mutation_idx): 
                confType = conf_class[conf_name] 
                #confType = getConfType(conf_name, f'./config_files/config_type_.yaml')
            
                check_running_conf(conf_name, str(i), mutation_idx)
                mutation_idx = str(mutation_idx) 
                print(f"configuration_name:{conf_name}, fault_value: {i}") 
                out = subprocess.run([os.environ['SPARKLORD_HOME']+'/modules/deployer/run_fi.sh',  spl_id, os.environ['SPL_JOB'], docker_conf_path, conf_name,str(i), mutation_idx]) 
                if out.returncode != 0: 
                    exit(1)
                saved_file_path = os.environ['SPARKLORD_HOME'] +'/logs/saved_list' 
                saved_lock_path = os.environ['SPARKLORD_HOME'] + '/logs/saved.lock'
                save_run_value(conf_name, str(i), mutation_idx, saved_file_path, saved_lock_path)
            
                       
                    
def read_all_conf_files(): 
    spl_id =''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    
    # run for all .xml files [core-site.xml, hdfs-site.xml, mapred-site.xml,
    # yarn-site.xml]
    dir = os.environ['SPARKLORD_HOME'] + '/config_files/' + os.environ['app']
    # get the list of configuration files 
    file_list = os.listdir(dir) 
    file_list.sort() 
    for f in file_list:
        read_prop_and_run(f, spl_id)

if __name__ == '__main__':
    
    if 'SPARKLORD_HOME' not in os.environ.keys(): 
        print('SPARKLORD_HOME is not in env') 
        print(f'run export SPARKLORD_HOME=[sparklord dir] first') 
        exit(1)
    
    print(os.environ['scenario_type'])
    if os.environ['scenario_type'] == "spark": 
        with open(os.environ['SPARKLORD_HOME'] + '/scenario_spark.yaml') as f: 
            scenario = yaml.load(f, Loader=yaml.FullLoader)['config'] 

    read_all_conf_files()
