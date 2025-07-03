import subprocess
import fcntl
import random
import string
import yaml
import os

def can_run(app_name, idx, application):
    ret = True
    lock = open(os.environ['SPARKLORD_HOME']+'/logs/conf.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not os.path.exists(os.environ['SPARKLORD_HOME']+'/logs/completed_list'):
        open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'w').close()

    with open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'r') as f:
        completed = f.readlines()
    completed = [i.strip() for i in completed]
    

    if f'{app_name} {idx}' not in completed:
        with open(os.environ['SPARKLORD_HOME']+'/logs/completed_list', 'a') as f:
            f.write(f'{app_name} {idx}\n')
    else:
        ret = False

    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return ret

def run_normal_run():
    # load scenario yaml file with yaml module
    spl_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    for iter in range(1):
        if can_run(os.environ['SPL_JOB'], iter, os.environ['app']):
            out = subprocess.run([os.environ['SPARKLORD_HOME']+'/modules/deployer/run_fi.sh', spl_id, os.environ['SPL_JOB'], '2', '100', '4g'])
            if out.returncode != 0: exit(1)

if __name__ == '__main__':
    # exit if SPARKLORD_HOME is not set in the environment variable
    if 'SPARKLORD_HOME' not in os.environ.keys():
        print('SPARKLORD_HOME is not in env')
        print(f'run export SPARKLORD_HOME=[sparklord dir] first')
        exit(1)

    run_normal_run()