import subprocess
import random
import string
import yaml
import os

def run_resource_fi():
    # load scenario yaml file with yaml module
    spl_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    with open(os.environ['SPARKLORD_HOME'] + '/scenario.yaml') as f:
        scenario = yaml.load(f, Loader=yaml.FullLoader)['resource']

    # run deployer with resource settings
    for i in range(scenario['cpu']['from'], scenario['cpu']['to'] + (1 if scenario['cpu']['delta'] > 0 else -1), scenario['cpu']['delta']):
        for j in range(scenario['cpu_util']['from'], scenario['cpu_util']['to'] + (1 if scenario['cpu_util']['delta'] > 0 else -1), scenario['cpu_util']['delta']):
            for k in range(scenario['memory']['from'], scenario['memory']['to'] + (1 if scenario['memory']['delta'] > 0 else -1), scenario['memory']['delta']):
                out = subprocess.run([os.environ['SPARKLORD_HOME']+'/modules/deployer/run_fi.sh', spl_id, os.environ['SPL_JOB'], str(i), str(j), str(k)+scenario['memory']['unit']])
                if out.returncode != 0: exit(1)
        


if __name__ == '__main__':
    # exit if SPARKLORD_HOME is not set in the environment variable
    if 'SPARKLORD_HOME' not in os.environ.keys():
        print('SPARKLORD_HOME is not in env')
        print(f'run export SPARKLORD_HOME=[sparklord dir] first')
        exit(1)

    run_resource_fi()