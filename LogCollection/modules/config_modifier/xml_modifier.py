import sys
import os
import xml.etree.ElementTree as elemTree

def main(filepath, conf_key, conf_value):
    tree = elemTree.parse(filepath)

    prop = tree.findall(f".//name[.='{conf_key}']..")[0]
    info = {}
    for elem in prop:
        info[elem.tag] = [elem, elem.text]

    if 'value' not in info.keys():
        empty_val = elemTree.fromstring('<value></value>')
        prop.append(empty_val)
        info['value'] = [empty_val, '']

    info['value'][0].text = conf_value

    tree.write(filepath)

if __name__ == '__main__':
    # exit if argument is not available
    if len(sys.argv) != 4:
        print('usage: python3 config_modifier.py config_file_path_from_SPARKLORD_HOME config_key config_value')
        print('example: python3 config_modifier.py $SPARKLORD_HOME/worker/config_files/core-site.xml hadoop.security.authorization true')
        exit(1)
    
    if 'SPARKLORD_HOME' not in os.environ.keys():
        print('SPARKLORD_HOME is not in env')
        print(f'run export SPARKLORD_HOME=[sparklord dir] first')
        exit(1)
    
    path = sys.argv[1]
    if not os.path.exists(path) or os.path.isdir(path):
        print(f'file {path} is not exists or not a file')
        exit(1)

    main(path, sys.argv[2], sys.argv[3])
