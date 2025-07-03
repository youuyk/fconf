import pickle, sys, os    
import multiprocessing
from make_template_sequence import make_template_main
import argparse
from itertools import repeat 

def strip(x):
    return x.strip()

def main(args):    
    
    app_name = args.app 
    log_p = args.log_p
    save_path = args.save_path
    # name of prepopulated template
    parser_p = args.parser_path 
    template_name = args.template_name
    
    log_p = f"{log_p}{app_name}/"
    save_path = f"{save_path}{app_name}/"
    # load template and reverse template 
    template_p = f"{parser_p}.pickle"
    template_r = f"{parser_p}_r.pickle"
    
    print(log_p, save_path)
    
    files = os.listdir(log_p) 
    #files.reverse()

    template_r_path, template_path, log_path, log_file_path, n_log_file_path, log_save_path, log_save_path_none, parser_path = [], [], [], [], [], [], [], [] 
    for idx, dirc in enumerate(files): 
        
        if os.path.isfile(f"{log_p}{dirc}"):
            continue 

        lfp = f"{dirc}/log"
        result_path = f"{log_p}{dirc}/sparkresult.txt" 
        
        # if combined log file is not exist 
        if not os.path.exists(f"{log_p}{dirc}/log") or not os.path.exists(result_path):
            continue 
        with open(result_path, "r") as fr:
            result = fr.readlines()
        result = " ".join(result)
        if "FINISHED SUCCEEDED 100%" in result:
            continue  
        #log_path.append(log_p)
        #log_file_path.append(lfp)
        s_path = f"{save_path}{lfp}_{template_name}"
        # if this log file is already templatized 
        if os.path.exists(s_path):
            continue
        
        parser_path.append(parser_p)
        template_path.append(template_p)
        template_r_path.append(template_r)
        log_path.append(log_p)
        log_file_path.append(lfp)
        log_save_path.append(s_path)
         
        
    print(len(parser_path), len(template_path), len(template_r_path), len(log_path), len(log_file_path), len(log_save_path))

    pool = multiprocessing.Pool()
    pool.starmap(make_template_main, zip(parser_path, template_path, template_r_path, log_path, log_file_path, log_save_path))
    pool.close()
    pool.join()
    
    #make_template_main(parser_path, log_path, log_file_path, log_save_path, app_name)
     
    '''with open('make_template_pipeline_app_name', "a") as fw:
        fw.write(str(app_name) + "\n")
    fw.close()    '''    

parser = argparse.ArgumentParser()
parser.add_argument("--app")
parser.add_argument("--log_p", required = False)
# save templatized log into same directory with raw logs 
parser.add_argument("--save_path", required = False)
parser.add_argument("--parser_path", default = "saved_log_template_3")
parser.add_argument("--template_name", default = "template_2")
parser.add_argument("--spark_apps")
args = parser.parse_args()

with open(args.spark_apps, "r") as fr:
    spark_apps = fr.readlines()
spark_apps = list(map(strip, spark_apps))
for s_app in spark_apps:
    args.app = s_app 
    main(args)
