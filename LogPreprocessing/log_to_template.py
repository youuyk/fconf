import pickle, sys, os  
import argparse

def strip(x):
    return x.strip()

def read_text(path):
    
    with open(path, "r") as fr:
        data = fr.readlines()
    data = list(map(strip, data))
    return data 

def main(args):
    
    logPath = args.logPath  
    appPath = args.appPath 
    step = args.step
    app_index = args.app_index
    
    apps = read_text(appPath)

    for index, app in enumerate(apps):
        
        print(f"Now Running: {app}")
        files = os.listdir(f"{logPath}{app}/")
        for file in files:  
            filePath = f"{logPath}{app}/{file}/log"
            resultPath = f"{logPath}{app}/{file}/sparkresult.txt"
            if not os.path.exists(resultPath):
                continue  
            with open(resultPath, "r") as fr:
                result = fr.readlines()
            result = " ".join(result)
            # run the lognroll with failed case 
            if "FINISHED SUCCEEDED 100%" in result:
                continue  
            if not os.path.exists(filePath):
                continue 
            with open(filePath, "r") as fr:
                logs = fr.readlines()
            logLines = len(logs)
            for line in range(0, logLines, step):
                start_index = line 
                end_index = line + step
                os.system(f"python3 lognroll_actual.py --logfile {filePath} --start_index {start_index} --end_index {end_index}")

parser = argparse.ArgumentParser()
parser.add_argument("--logPath")
parser.add_argument("--appPath")
parser.add_argument("--step", default = 10000, type = int)
parser.add_argument("--app_index", type = int)
args = parser.parse_args()

main(args)
