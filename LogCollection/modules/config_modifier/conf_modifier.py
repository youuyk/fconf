import os, sys 

def main(filepath, conf_name, conf_val):
    
    with open(filepath, "w") as fw:
        fw.write(f"{conf_name} {conf_val}\n")
        # add 4 default value 
        if conf_name != "spark.master":
            fw.write("spark.master yarn\n") 
        if conf_name != "spark.eventLog.enabled":
            fw.write("spark.eventLog.enabled false\n")
        if conf_name != "spark.executor.memory":
            fw.write("spark.executor.memory 4g\n") 
        if conf_name != "spark.serializer":
            fw.write("spark.serializer org.apache.spark.serializer.JavaSerializer\n") 
    fw.close()
    
if __name__ == "__main__":
    
    # sys.argv = filename, config_file_path, conf_name, conf_val 
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