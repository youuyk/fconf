import os  
import pickle 
import random 
import pandas as pd 
from numpy import dot 
from numpy.linalg import norm
from itertools import repeat 
import numpy as np 
import torch 
import dataset_seq_embed
from sklearn.manifold import TSNE 
import seaborn as sns 
import matplotlib.pyplot as plt 

def get_conf_name_in_conf_dict(conf_index_dict, ans):
    
    for conf_name, index in conf_index_dict.items():
        if index == ans:
            return conf_name 

def write_result_2tr(answer, conf_index_dict, rank, prob):
    
    for idx, ans in enumerate(answer):
        conf = get_conf_name_in_conf_dict(conf_index_dict, ans)
        print(f"Result of {conf}: {rank[idx]}, {prob[idx]}")

def load_all_config_of_train_app(train_app_path, app_config_path):
    
    with open(train_app_path, "r") as fr:
        apps = fr.readlines()
    apps = list(map(strip, apps))

    config_files = os.listdir(app_config_path)
    
    config_set = []
    for app in apps:
        app = app.split("_")[0]
        for config in config_files:
            config_file_name = config.split("_")[0]
            config_file_name = config_file_name.split(".")
            if len(config_file_name) == 1:
                config_file_name = config_file_name[0]
            else:
                config_file_name = config_file_name[1]
            if app == config_file_name:
                configPath = f"{app_config_path}{config}"
                configs = read_text(configPath)
                config_set.extend(configs) 
                break 
    config_set = list(set(config_set))
    return config_set

# balancing dataset
def balance_dataset_over(train_files_dict):

    maxDataSize = 0
    for conf, logList in train_files_dict.items():
        maxDataSize = max(maxDataSize, len(logList))

    print(f"Max number of data: {maxDataSize}")
    tmp_files_dict = {}
    for conf, logList in train_files_dict.items():
        if maxDataSize == len(logList):
            tmp_files_dict[conf] = logList
        else:
            tmpList = logList
            while len(tmpList) == maxDataSize:
                tmpSet = random.choice(logList)
                tmpList.append(tmpSet)
            tmp_files_dict[conf] = tmpList
    
    return tmp_files_dict  

def balance_dataset_under(train_files_dict):
    
    minDataSize = float('inf')
    for conf, logList in train_files_dict.items():
        minDataSize = min(minDataSize, len(logList))
    
    print(f"Min number of data: {minDataSize}")
   
    tmp_files_dict = {}
    for conf, logList in train_files_dict.items():
        tmpList = random.sample(logList, minDataSize)
        tmp_files_dict[conf] = tmpList 
    
    return tmp_files_dict         

def get_log_set(log):
    
    log_set = []
    for l in log:
        log_set.extend(l)
    log_set = set(log_set)

    return len(log_set) + 1 

def load_pickle(path):
    with open(path, "rb") as fr:
        data = pickle.load(fr)
    return data 

def save_pickle(data, path):
    with open(path, "wb") as fw:
        pickle.dump(data, fw) 

def make_target_conf(p):
    
    with open(p, "rb") as fr:
        f = pickle.load(fr)
    t = [] 
    for k, v in f.items():
        print(k, len(v), len(set(v)))
        t.extend(v)
    return t 

def get_num_conf(target_conf, conf_number):
    random.shuffle(target_conf)
    return target_conf[:conf_number]

def strip(text):
    return text.strip()

def get_data(target_conf_path):
    
    with open(target_conf_path, "r") as fr:
        confs = fr.readlines()
    confs = list(map(strip, confs))
    return confs 

def make_conf_df(conf_path):
    conf_df = pd.read_excel(conf_path)
    conf_df = conf_df.loc[conf_df.file_name == 'core-site.xml', 'config_name']
    return list(conf_df)

def get_hadoop(conf_df):
    hadoop = [] 
    random.shuffle(conf_df)
    for conf in conf_df:
        name = conf.split(".")[0]
        nd_name = conf.split(".")[1]
        if name == "hadoop" and len(hadoop) < 20:
            hadoop.append(conf)
            
    return hadoop     

def get_conf_name(info_path):
    with open(info_path, "r") as fr:
        info = fr.readlines()
    app_name = info[0].strip().split("=")[1].split(".")[-1]
    conf_file = info[5].strip().split("=")[-1]
    conf_name = info[6].strip().split("=")[-1]
    conf_val = info[7].strip().split("=")[-1]
    return app_name, conf_file, conf_name, conf_val

def get_file_path(file, app_path):
    return f"{app_path}/{file}"

def get_file_path_preprocess_template(file, file_name, app_path):
    f_path = f"{app_path}/{file}/{file_name}"
    if not os.path.exists(f_path):
        return "None"
    return f"{app_path}/{file}/{file_name}"

def get_all_files(path, apps):
    
    files = []
    for app in apps:
        file = os.listdir(f"{path}/{app}")
        app_path = f"{path}/{app}"
        file_path = list(map(get_file_path, file, repeat(app_path)))
        files.extend(file_path)
         
    return files

def get_all_files_preprocess_template(path, apps, file_name):
    
    files = [] 
    for app in apps:
        file = os.listdir(f"{path}/{app}")
        app_path = f"{path}/{app}"
        file_path = list(map(get_file_path_preprocess_template, file, repeat(file_name), repeat(app_path)))
        files.extend(file_path)
    return files 

def remove_none_file(train_files):
    
    new_files = [] 
    for file in train_files:
        if file == "None":
            continue   
        new_files.append(file)
    return new_files 

def make_conf_cluster(c_path, raw_path, app):
    
    with open(c_path, "rb") as fr:
        c = pickle.load(fr)
    
    c_dict = {}
    for label, labeled_data in c.items():
        for conf in labeled_data:
            c_dict[conf] = label
            
    return c_dict, c       

def get_data_dict_cluster_conf_val(repr_conf, run_type, files, target_conf, target_app, raw_path, conf_cluster):
    
    result = {}
    for file in files:
        # get log files 
        #print(file)
        with open(file, "rb") as fr:
            logs = pickle.load(fr)
        # if no logs printed 
        if len(logs) == 0:
            continue
        # get app and file name 
        app_name = file.split("/")[4]
        file_name = file.split("/")[5]
        # load information of this case 
        if run_type == "train":
            if app_name not in target_app:
                continue  
        raw_file_path = f"{raw_path}{app_name}/{file_name}/LogInformation.txt"
        if not os.path.exists(raw_file_path):
            #print(raw_file_path)
            continue  
        app_name, _, conf_name, conf_val = get_conf_name(raw_file_path)
        
        if conf_name in repr_conf:
            c_name = conf_name
        else:
            c_name = conf_name + " " + conf_val
            
        if c_name not in target_conf:
            continue  
        
        if c_name not in conf_cluster:
            print(c_name)
            continue 
        '''if c_name  == "ipc.ping.interval" and conf_val == '1':
            print(f'skip {c_name}')
            continue 
        if c_name == "file.stream-buffer-size" and conf_val == '2147483647':
            print(f'skip {c_name}')
            continue
        if c_name == "file.bytes-per-checksum" and conf_val == "2147483647":
            print(f'skip {c_name}')
            continue '''
        label = conf_cluster[c_name]
        if label not in result:
            result[label] = [] 
        tmp = result[label]
        tmp.append(logs)
        result[label] = tmp 
        
    return result 

def get_data_dict_cluster(run_type, files, target_conf, target_app, raw_path, conf_cluster):
    
    result = {}
    for file in files:
        # get log files 
        #print(file)
        with open(file, "rb") as fr:
            logs = pickle.load(fr)
        # if no logs printed 
        if len(logs) == 0:
            continue
        # get app and file name 
        app_name = file.split("/")[4]
        file_name = file.split("/")[5]
        # load information of this case 
        if run_type == "train":
            if app_name not in target_app:
                continue  
        raw_file_path = f"{raw_path}{app_name}/{file_name}/LogInformation.txt"
        if not os.path.exists(raw_file_path):
            #print(raw_file_path)
            continue  
        app_name, _, conf_name, conf_val = get_conf_name(raw_file_path)
        #print(app_name, conf_name, conf_val)
        # if only using hadoop configuration 

        if conf_name not in target_conf:
            continue  
        label = conf_cluster[conf_name]
        if label not in result:
            result[label] = [] 
        tmp = result[label]
        tmp.append(logs)
        result[label] = tmp 
        
    return result 

def get_app_result(result_path):
    result = get_text_data(result_path)
    timeout = int(result[1].split(" ")[1])
    result = result[-1] 
    #result = result[-1].split(" ")

    '''finished_result = "FINISHED" 
    accepted_result = "ACCEPTED"
    failed_result = "FAILED"
    if finished_result in result:
        result_text = finished_result
    elif accepted_result in result:
        result_text = accepted_result
    elif failed_result in result:
        result_text = failed_result
    else:
        result_text = "None"
    
    if timeout == 0 and finished_result in result:
        return True, result_text
    return False, result_text'''
    
    if timeout == 0 and "FINISHED SUCCEEDED 100%" in result:
        return True 
    else:
        return False

def get_data_dict(label_type, data_type, files, system ):
    
    result = {}
    result_val = {}
    result_filename = {}
    totFiles = 0 
    print(f"# of files: {len(files)}")
    conftime = {}
    for f_idx, file in enumerate(files):
        if f_idx % 1000 == 0:
            print(f"running: {f_idx}")
        # get log files 
    
        if not os.path.exists(file):
            print(f"{file} is not exists")
            continue  
        
        with open(file, "rb") as fr:
            logs = pickle.load(fr)
        # if no logs printed 
        if len(logs) == 0:
            continue
    
        f_path = "/".join(file.split("/")[:-1])
        
        if system == "redis":
            conf_path = f"{f_path}/redis-master-1-revised.conf"
            with open(conf_path, "r") as fr:
                confs = fr.readlines()
            confs=  confs[-1]
            conf_name = confs.split(" ")[0]
            conf_val = " ".join(confs.split(" ")[1:])
            conf_s = f"{conf_name}_{conf_val}"
            if conf_s not in conftime:
                conftime[conf_s] = 0 
            tmp = conftime[conf_s]
            if tmp == 20:
                continue 
            conftime[conf_s] = tmp + 1 
        
        if system == "spark":
            raw_file_path = f"{f_path}/LogInformation.txt"
            result_path = f"{f_path}/sparkresult.txt"

            #raw_file_path = f"{raw_path}/{app_name}/{file_name}/LogInformation.txt"
            #result_path = f"{raw_path}/{app_name}/{file_name}/sparkresult.txt"
            # LogInformation.txt & sparkresult.txt not exists -> continue 
            if not os.path.exists(raw_file_path):
                #print(raw_file_path)
                continue  
            if not os.path.exists(result_path):
                continue 
        
            _, _, conf_name, conf_val = get_conf_name(raw_file_path)
            result_flag = get_app_result(result_path)
        
            # testing with normal logs data, after training with abnormal log data 
            if data_type == "normal":
                if result_flag == False:
                    continue   
            # usually training with abnormal log data, and testing 
            if data_type == "abnormal":
                # if result of application is succeed, don't add this data
                if result_flag == True:
                    continue 
            # if data_type == "both", then use both result_flag = False, True data 
            # if data_type == "both", change the conf_name of result_flag = True data as normal 
            if data_type == "both":
                if result_flag == True:
                    # collect the normal data at one key in dictionary 
                    conf_name = "normal"
                
            if label_type == "label_with_value":
                conf_name = conf_name + "_" + conf_val
            
        if conf_name not in result:
            result[conf_name] = [] 
            result_val[conf_name] = [] 
            result_filename[conf_name] =[] 
            
        tmp = result[conf_name]
        tmp_val = result_val[conf_name]
        tmp_filename = result_filename[conf_name]
        
        '''if system == "redis":
            if len(tmp) >= 30:
                continue '''
            
        tmp.append(logs)
        tmp_val.append(conf_val)
        tmp_filename.append(f_path)
        
        result[conf_name] = tmp 
        result_val[conf_name] = tmp_val 
        result_filename[conf_name] = tmp_filename
        
        totFiles += 1 
    return result, result_val, result_filename, totFiles 

def get_data_dict_sim(run_type, files, target_app, raw_path, threshold):
    result = {}
    for file in files:
        with open(file, "rb") as fr:
            logs = pickle.load(fr)
        if len(logs) == 0:
            continue  
        app_name = file.split("/")[4] 
        file_name = file.split("/")[5]
        
        if run_type == "train":
            if app_name not in target_app:
                continue  
        raw_file_path = f"{raw_path}{app_name}/{file_name}/LogInformation.txt"
        if not os.path.exists(raw_file_path):
            continue  
        app_name, _, conf_name, conf_val = get_conf_name(raw_file_path)
        #flag = check_similarity(conf_name, file, result, logs, threshold)
        # cosine similarity with all others configuration is below 0.2
        flag = True 
        if flag == True:
            #print(file, conf_name, conf_val)
            if conf_name not in result:
                result[conf_name] = [] 
            tmp = result[conf_name]
            tmp.append(logs)
            result[conf_name] = tmp 
    return result 

def check_similarity(conf_name, file_name, log_dict, a_logs, threshold):
    
    max_len = 46420 
    if len(a_logs) < max_len:
        a_logs.append(-1)
    
    for k, v in log_dict.items():
        for b_logs in v:
            '''max_len = max(len(a_logs), len(b_logs))
            if len(a_logs) < max_len:
                a_logs = a_logs + [-1 for i in range(max_len - len(a_logs))]'''
            if len(b_logs) < max_len:
                b_logs = b_logs + [-1 for i in range(max_len - len(b_logs))]
            cos_sim = dot(a_logs, b_logs) / (norm(a_logs) * norm(b_logs))
            if cos_sim == 1 and k == conf_name:
                return True  
            if cos_sim >= threshold:
                #print(file_name)
                return False
            #print(cos_sim, threshold)  
    return True         
    
def make_sequence(logs, window_size, if_sliding):
    
    sequence = [] 
    l_idx = 0 
    while True:
        if l_idx + window_size > len(logs):
            seq = logs[l_idx:]
            sequence.append(seq)
            break 
        seq = logs[l_idx:l_idx+window_size]
        sequence.append(seq)
        if if_sliding == False:
            l_idx = l_idx + window_size
        else: 
            l_idx += 1  
    return sequence 

def get_dataset(logs, conf, window_size, log_data, conf_data, if_sliding):
    
    sequence = make_sequence(logs, window_size, if_sliding)
    log_data.extend(sequence)
    conf_data.extend([conf for i in range(len(sequence))])
    return log_data, conf_data 

def get_dict_dataset_train(log, conf):
    
    log_dict, conf_dict, conf_dict_reverse = {}, {}, {}
    train_log, train_conf = [], [] 
    conf_key = 0 
    log_key = 2
    for s_idx, sequence in enumerate(log):
        seq_conf = conf[s_idx]
        if seq_conf not in conf_dict:
            conf_dict[seq_conf] = conf_key 
            conf_dict_reverse[conf_key] = seq_conf
            conf_key += 1 
        log_seq = [] 
        # convert each log index in sequence into dict index, or create new 
        for word in sequence:
            if word not in log_dict:
                log_dict[word] = log_key 
                log_key += 1
            log_seq.append(log_dict[word])
        '''if s_idx % 100000 == 0:
            print(sequence, seq_conf)
            print(log_seq, conf_dict[seq_conf])'''
        train_log.append(log_seq)
        train_conf.append(conf_dict[seq_conf])

    # log_key == input_dim 
    # conf_key == number of labels 
    return train_log, train_conf, log_dict, conf_dict, conf_dict_reverse, log_key, conf_key 

def get_dict_dataset_test(log, conf, log_dict, conf_dict):
    
    test_log, test_conf = [], [] 
    for s_idx, sequence in enumerate(log):
        seq_conf = conf[s_idx]
        # if training set does not have this kind of configuration (not trained), then do not add to test dataset 
        if seq_conf not in conf_dict:
            # for normal configuration testing, temporary setting index of normal configuration as -1 
            conf_dict[seq_conf] = -1  
            #continue  
        log_seq = [] 
        for word in sequence:
            # unknown_token = 1
            if word not in log_dict:
                log_seq.append(1)
            else:
                log_seq.append(log_dict[word])
        
        test_log.append(log_seq)
        test_conf.append(conf_dict[seq_conf])
    
    return test_log, test_conf 

# if conf is not in conf_dict, validation dataset doesn't include that configuration 
def get_dict_dataset_val(log, conf, log_dict, conf_dict):
    val_conf_set = set()    
    val_log, val_conf = [], []
    for s_idx, sequence in enumerate(log):
        
        seq_conf = conf[s_idx]
        # if seq_conf(configuration name is not in conf_dict, skip this configuration)
        if seq_conf not in conf_dict:
            continue   
        log_seq = [] 
        for word in sequence:
            if word not in log_dict:
                log_seq.append(1)
            else:
                log_seq.append(log_dict[word])
        val_log.append(log_seq)
        val_conf.append(conf_dict[seq_conf])
        val_conf_set.add(conf_dict[seq_conf])
    return val_log, val_conf, len(val_conf_set)

def padding(log, max_seq_length:int, pad_token:int):
    
    input = []
    for l in log:
        input.append(l) 
    if len(input) < max_seq_length:
        while len(input) < max_seq_length:
            input.append(pad_token)
    # truncation 
    elif len(input) >= max_seq_length:
        input = input[:max_seq_length]
    return input

def shuffle(log, conf):
    
    d_list = [] 
    for idx, l in enumerate(log):
        n_dict = {}
        n_dict['log'] = l
        n_dict['conf'] = conf[idx] 
        d_list.append(n_dict)
        
    random.Random(0).shuffle(d_list)

    log_list, conf_list= [], []
    for d in d_list:
        l = d['log']
        c = d['conf']
        log_list.append(l)
        conf_list.append(c)
        
    return log_list, conf_list

def make_balanced_dataset(log, conf, data_per_conf):
    
    data_dict = {}
    for i, l in enumerate(log):
        if conf[i] == 32:
            continue  
        if conf[i] not in data_dict:
            data_dict[conf[i]] = [] 
        tmp = data_dict[conf[i]] 
        tmp.append(l)
        data_dict[conf[i]] = tmp 
        
    n_logs, n_conf = [], [] 
    min_data = min(data_per_conf)  
    for conf, logs in data_dict.items():
        # if this conf has imbalanced dataset 
        #print(conf, len(data_dict[conf]))
        if len(logs) > min_data:
            tmp_logs = [random.choice(logs) for i in range(min_data)]
            data_dict[conf] = tmp_logs
        #print(conf, len(data_dict[conf]))
        n_logs.extend(data_dict[conf]) 
        n_conf.extend([conf for i in range(len(data_dict[conf]))])
        #print(len(n_logs), len(n_conf))
    return n_logs, n_conf            
                
def get_text_data(path):
    if not os.path.exists(path):
        return None 
    with open(path, "r") as fr:
        data = fr.readlines()
    data = list(map(strip, data))
    return data 

def data_per_conf(train_conf):
    
 
    conf_set = list(set(train_conf))
    conf_number = [0 for i in range(len(conf_set))]
    for c in conf_set:
        c_count = train_conf.count(c)
        conf_number[c] = c_count 
    return conf_number 

def padding_embedding(log, embed_length, max_seq_length):
    
    padded_log = log.copy()
    pad_index = [0 for i in range(max_seq_length)] 
    # padding token 
    pad_token = [0 for i in range(embed_length)]
    pad_token = np.array(pad_token)
    
    while True:
        if len(padded_log) >= max_seq_length:
            break 
        padded_log.append(pad_token)
        index = len(padded_log) - 1 
        pad_index[index] = 1 
    
    # truncation
    # item of pad_index is all 0 
    if len(padded_log) > max_seq_length:
         padded_log = padded_log[:max_seq_length]  
    
    return padded_log, pad_index 

def to_list(conf):
    c = conf[0]
    return c 

def strip(x):
    return x.strip()

def write_text(path, data):
    
    with open(path, "w") as fw:
        for d in data:
            fw.write(str(d) + "\n")
    fw.close()

def read_text(path):
    with open(path, "r") as fr:
        data = fr.readlines()
    data = list(map(strip, data))
    return data 
            
def read_pickle(path):
    with open(path, "rb") as fr:
        data = pickle.load(fr)
    return data 
    
# open pickle file 
# padding 
# make batched dataset
# max_seq_length is fixed with arguments 
def make_embed_dataset(data_path, conf_path, max_seq_length):
    
    # key: configuration, value: sequence of embedding for each sequence  
    data = read_pickle(data_path)
    conf = read_pickle(conf_path)
    
    # padding data with max_length 
    # length of embedding for one sequence -> need for padding 
    embed_length = len(data[0][0])
    
    padded_logs, padded_index = [], [] 
    for log in data:
        pad_log, pad_index = padding_embedding(log, embed_length, max_seq_length)
        padded_logs.append(pad_log)
        padded_index.append(pad_index)
    
    # get the number of labels     
    conf = list(map(to_list, conf))
    num_labels = len(list(set(conf)))
    
    # convert the list to numpy.ndarray 
    padded_logs = np.array(padded_logs)
    padded_index = np.array(padded_index)
    
    '''train_size = int(len(padded_logs) * (1 - validation_prop))
    train_log = padded_logs[:train_size]
    train_index = padded_index[:train_size] 
    train_conf = conf[:train_size]
    
    val_log = padded_logs[train_size:]
    val_index = padded_index[train_size:]
    val_conf = conf[train_size:]'''
    # size of padded_logs: (# of train_data, max_seq_length, # features)
    # size of padded_index: (# of train_data, max_seq_length)
    #train_dataset = dataset.LogSequenceDataset(torch.tensor(padded_logs, dtype=torch.long), torch.tensor(padded_index, dtype = torch.long), torch.tensor(conf))
    train_dataset = dataset_seq_embed.LogSequenceDataset(torch.tensor(padded_logs, dtype = torch.float32), torch.tensor(padded_index, dtype = torch.long), torch.tensor(conf))
    #train_dataset = dataset_seq_embed.LogSequenceDataset(torch.tensor(train_log, dtype = torch.float32), torch.tensor(train_index, dtype = torch.long), torch.tensor(train_conf))
    #val_dataset = dataset_seq_embed.LogSequenceDataset(torch.tensor(val_log, dtype = torch.float32), torch.tensor(val_index, dtype = torch.long), torch.tensor(val_conf)) 
    
    return train_dataset, num_labels, embed_length
    #return train_dataset, val_dataset, num_labels, embed_length
    
def make_embed_dataset_per_conf(data, max_seq_length, conf_index):
    
    padded_logs, padded_index=  [], []
    pad_log, pad_index = padding_embedding(data, len(data[0]), max_seq_length) 
    padded_logs.append(pad_log)
    padded_index.append(pad_index)
    
    padded_logs = np.array(padded_logs)
    padded_index = np.array(padded_index)
    conf_index = [conf_index]
    
    dataset = dataset_seq_embed.LogSequenceDataset(torch.tensor(padded_logs, dtype = torch.float32), torch.tensor(padded_index, dtype = torch.long), torch.tensor(conf_index))
    return dataset 

def make_dataset_per_data(data, conf, max_seq_length, conf_dict):
    
    pad_log, pad_index = padding_embedding(data, len(data[0]), max_seq_length)
    pad_log = np.array(pad_log)
    pad_index = np.array(pad_index)

    c = conf.split(" ")[0]
    v = conf.split(" ")[1]
    conf_index = [conf_dict[c]]
    train_dataset = dataset_seq_embed.LogSequenceDataset(torch.tensor(pad_log, dtype = torch.float32), torch.tensor(pad_index), torch.tensor(conf_index))
    return train_dataset, c, v

def conf_to_list(label):
    
    conflist = [] 
    for l in label:
        conflist.append(l.item())
    
    return conflist 

def limit_conf(hs, label, predict, predict_max_conf, n_conf):
    
    n_hs, n_label, n_predict, n_max_conf = [], [], [], [] 
    for idx, l in enumerate(label):
        
        if l <= n_conf:
            n_hs.append(hs[idx])
            n_label.append(label[idx])
            n_predict.append(predict[idx])
            n_max_conf.append(predict_max_conf[idx])
        
    return n_hs, n_label, n_predict, n_max_conf
    
def make_tsne_predict_label_flag(label, predict_max_conf):

    flag = [] 
    for idx, l in enumerate(label):
        if l == predict_max_conf[idx]:
            flag.append(1)
        else:
            flag.append(0)
        
    return flag 

def eval_mrr(ranks):

    tot_rank = 0
    totCase = 0
    for rank in ranks:
        if rank == -1:
            continue 
        tot_rank += 1/rank 
        totCase += 1 
    mrr = tot_rank / totCase
    return mrr 
        
def eval_successrate(sorted_index, label, top_n):
    
    label = label.item()
    top_conf = sorted_index[:top_n]
    if label in top_conf:
        return True  
    return False  

def eval_rk(rank, k):
    
    tot_case, n_case = 0, 0
    for r in rank:
        if r == -1:
            continue  
        if r <= k:
            n_case += 1 
        tot_case += 1 
    return n_case / tot_case, n_case 

def eval_max_conf(sorted_index, model_predict):
    
    top_conf = sorted_index[0]
    prob = model_predict[top_conf]
    return prob 
    
def eval_ranking(label, sorted_index):
    
    label = label.item()
    r = 1 
    flag = False 
    for idx in sorted_index:
        if idx == label:
            flag = True 
            break 
        r += 1 
    # if labes is not in sorted_index 
    if flag == False:
        return -1 
    return r 
        
def eval_get_prob(label, model_predict):
    
    #print(label)
    #print(model_predict)
    label = label.item()
    if label == -1:
        return -1 
    prob = model_predict[label]
    return prob 

def sorted_configuration_index(model_predict):

    #print("-" * 30)
    #print(f"Conf: {conf_answer}")
    model_predict_index = np.argsort(model_predict, axis = 0)
    #print(model_predict)
    #print(model_predict_index)
    model_predict_index = np.flip(model_predict_index)
    #print(model_predict_index)
    return model_predict_index

# rank: ranking of label 
# prob: probability of label 
def save_excel(train_app_to_name, test_app_to_name, val_app_to_name, rep, n_epochs, max_seq_length_1tr, max_seq_length, rank, prob, highest_prob, data_type):
    
    df = pd.DataFrame(columns = ['index', 'rank', 'prob_answer', 'highest_prob'])
    for idx, r in enumerate(rank):
        p = prob[idx]
        hp = highest_prob[idx]
        df.loc[idx] = [idx, r, p, hp]
        
    df.to_excel(f'./result/2ndTR_train_{train_app_to_name}_val_{val_app_to_name}_test_{test_app_to_name}_rep_{rep}_epoch2nd_{n_epochs}_max_seq_length_1st_{max_seq_length_1tr}_max_seq_length_{max_seq_length}_model_result_{data_type}.xlsx')
    
def divide_test_data(result):
    
    model_predict, answer, model_predict_embed, conf, val = [], [], [], [], []
    
    for r in result:
        model_predict.append(r[0])
        answer.append(r[1])
        model_predict_embed.append(r[2])
        conf.append(r[3])
        val.append(r[4])
        
    return model_predict, answer, model_predict_embed, conf, val

def eval_get_highest_prob(predict):
    
    max_prob = -1
    for idx, item in enumerate(predict):
        if item > max_prob:
            max_prob = item 
    return max_prob 

def make_embedding_plot(label, embeddings, n_components, save_name):
    
    n_labels = len(list(set(label)))
    model = TSNE(n_components = n_components)
    
    embedded = model.fit_transform(embeddings)
    
    df = pd.DataFrame(columns = ['x', 'y', 'label'])
    x = embedded[:, 0]
    y = embedded[:, 1]
    df['x'] = x 
    df['y'] = y 
    df['label'] = label 
    
    palette = sns.color_palette("bright", n_labels) 
    
    sns.scatterplot(x = 'x', y = 'y', hue = 'label', data = df)
    
    plt.savefig(f"{save_name}.png")
    
def limit_normal_data(files_dict, files_dict_val):
    
    avg_len = 0 
    
    # get the average number of data of abnormal configurations 
    for k, v in files_dict.items():
        if k == "normal":
            continue   
        avg_len += len(v)
    avg_len = avg_len / len(files_dict)
    print(f"# of conf(key): {len(files_dict)}, # of avg data: {avg_len}")
    
    normal_data = files_dict['normal']
    normal_data_val = files_dict_val['normal']
    normal_data_length = [i for i in range(0, len(normal_data))]
    # randomly sample the index of normal data 
    normal_data_sample = random.sample(normal_data_length, int(avg_len))
    
    sampled_data, sampled_data_val = [], [] 
    for item_idx, item in enumerate(normal_data):
        if item_idx in normal_data_sample:
            sampled_data.append(item)
            sampled_data_val.append(normal_data_val[item_idx])

    files_dict['normal'] = sampled_data 
    files_dict_val['normal'] = sampled_data_val 
    
    print(f"# of normal data: {len(files_dict['normal'])}")
    totFiles = 0 
    for k, v in files_dict.items():
        totFiles += len(v)

    return files_dict, files_dict_val, totFiles 

def get_normal_case(answer, conf_index_dict, rank):
    
    normal_rank = [] 
    for idx, ans in enumerate(answer):
        conf_name = get_conf_name_in_conf_dict(conf_index_dict, ans)
        if conf_name== "normal":
            normal_rank.append(rank[idx])
    
    return normal_rank

def get_abnormal_case(answer, conf_index_dict, rank):
    
    abnormal_rank= [] 
    for idx, ans in enumerate(answer):
        conf_name = get_conf_name_in_conf_dict(conf_index_dict, ans)
        if conf_name != "normal":
            abnormal_rank.append(rank[idx])
    return abnormal_rank
