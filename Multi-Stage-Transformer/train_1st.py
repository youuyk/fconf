from dataset import ConfLogDataset
import torch.nn as nn 
import torch 
from itertools import repeat 
from model import Encoder, TransformerEncoder
import time, os, sys, pickle  
import numpy as np   
import util 
from sklearn.model_selection import train_test_split
import pandas as pd 
from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def train(epoch, model, dataloader, optimizer, criterion, clip, device):
    #model.to(f'cuda:{model.device_ids[0]}')
    model.to(device)
    model.train()
    epoch_loss = 0
    train_loss, x = [], [] 
    
    for i, batch in enumerate(dataloader):
        log = batch[0]
        conf = batch[1]
        log = log.to(device)
        conf = conf.to(device)
        
        #log = log.to(f'cuda:{model.device_ids[0]}')
        #conf = conf.to(log.device)
        optimizer.zero_grad()
        
        output, predict = model(log)
        # log.shape: [batch_size, max_seq_length]
        # conf.shape: [batch_size, 1]
        # output = embedding of batched input 
        # output.shape(before limit to first token): [batch_size, max_seq_length, hidden_dim] (e.g.[32, 128, 512])
        # output.shape(after limit to first token): [batch_size, hidden_dim] (e.g.[32, 128])
        # label: predicted label 
        # label.shape: [batch_size, num_labels] (32, 9)   
        
        loss = criterion(predict, conf)
        '''if i % 5000 == 0:
            print(f"(During training) loss_{i}: {loss}")'''
        train_loss.append(loss.item())
        x.append(i)
        loss.backward()
    
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss/ len(dataloader), train_loss, x    

def validation(model, dataloader, criterion, device):
    
    model.to(device)
    model.eval()
    total_loss = 0 
    val_loss = [] 
    with torch.no_grad():
        
        for i, batch in enumerate(dataloader):
            log = batch[0]
            conf = batch[1]
            log = log.to(device)
            conf = conf.to(device)
            output, predict = model(log)
            
            loss = criterion(predict, conf)
            loss = loss.item()
            val_loss.append(loss)
            total_loss += loss 
    
    return total_loss / len(dataloader)                    
            

def evaluate(model, test_dataloader, n_top, device):
    
    embeddings, confs = [], [] 
    model.eval()
    softmax = nn.Softmax(dim = 1)
    
    tot_result = [] 
    with torch.no_grad():
        
        for i, batch in enumerate(test_dataloader):
            log = batch[0]
            conf = batch[1]
            log = log.to(device)
            # output : embedding of each log sequence     
            output, predict = model(log)
            #print(output.shape)
            output = output.cpu().numpy()
            conf = conf.cpu().numpy()
            embeddings.extend(output)
            confs.extend(conf)
            
            '''if i == 0:
                print(f"Number of labels: {len(predict[0])}")'''
                
            s = softmax(predict)
            s = s.detach().cpu().numpy()
            #s_sort = np.argsort(s, axis = 1)
            
            tot_result.extend(s)
            
    # tot_result: [# of sequence, num_labels]
    # embeddings: [# of sequence, # of features]
    # conf: [# of sequence, 1]
    return tot_result, embeddings, confs
    
def get_case_score(tot_result, num_labels):
    
    tot_score = [0 for i in range(num_labels)]
    for result in tot_result:
        for s_idx, score in enumerate(result):
            tot_score[s_idx] += score 
    return tot_score 

def get_tmp_case_result(tot_result, ans_conf):

    rankList= []     
    for result in tot_result:
        tmp_result = result.copy()
        result_sort = np.argsort(tmp_result)
        rank = 1
        for n in range(len(result_sort) -1 , -1, -1):
            if result_sort[n] == ans_conf:
                break 
            rank += 1 
        rankList.append(rank)
        
    return rankList
        
def get_type_of_conf(conf, configuration_file_path):

    df = pd.read_excel(configuration_file_path)
    conf_type = df[df.config_name == conf]['config_type']
    return conf_type

def get_frequency(log, train_log, train_conf):
    conf_list = [] 
    for idx, train_l in enumerate(train_log):
        if train_l == log:
            conf_list.append(train_conf[idx])
    return len(set(conf_list))

def get_data_frequency(test_log, train_log, train_conf):
    
    # get the frequency of log in test_log 
    # return list of frequency, answer of each log(test_conf) 
    frequency = list(map(get_frequency, test_log, repeat(train_log), repeat(train_conf)))
    return frequency 
    
def get_data_per_conf(log, conf):
    
    conf_number = [] 
    conf_set = list(set(conf))
    for c in conf_set:
        conf_number.append(conf.count(c))
    return conf_number, conf_set

def get_max_prob(tot_result, num_labels):
    
    max_result = [0 for i in range(num_labels)]
    for result in tot_result:
        for s_idx, score in enumerate(result):
            s_max = max_result[s_idx]
            if s_max < score:
                max_result[s_idx] = score 
    # maximum score of each label 
    return max_result 

def data_save(embed_save_path, model, data_type, train_data_type, files_dict, files_dict_val, label_type, log_index_dict, conf_index_dict, rep, n_epochs, n_top, batch_size, device, pad_token, if_sliding, train_app_to_name, val_app_to_name, target_conf_to_name, max_seq_length, hidden_dim, layers, attention_heads, ff_dim):
    
    log, embed, conf, conf_name, conf_val, conf_name_str =[], [], [], [], [], [] 
    seq_dict = {}
    print(f"Saving {data_type}")
    for t_idx, (test_conf, test_logs) in enumerate(files_dict.items()):
        
        #print(f"{data_type} configuration: {test_conf}, {data_type} logs: {len(test_logs)}")
        for test_logs_tmp_index, test_logs_tmp in enumerate(test_logs):
            # injected value to this configuration 
            test_log_data, test_conf_data = util.get_dataset(test_logs_tmp, test_conf, max_seq_length, [], [], if_sliding)
            # convert it to index of log and conf 
            if data_type == "val":
                test_log_data, test_conf_data, _ = util.get_dict_dataset_val(test_log_data, test_conf_data, log_index_dict, conf_index_dict)
            else:
                test_log_data, test_conf_data = util.get_dict_dataset_test(test_log_data, test_conf_data, log_index_dict, conf_index_dict)
            test_log_data = list(map(util.padding, test_log_data, repeat(max_seq_length), repeat(pad_token)))
            
            if len(test_log_data) == 0 and len(test_conf_data) == 0:
                continue  
            
            test_dataset = ConfLogDataset(torch.tensor(test_log_data, dtype = torch.long), torch.tensor(test_conf_data))
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
            
            # tot_result: [# of sequence, num_labels]
            # embeddings: [# of sequence, # of features]
            # conf: [# of sequence, 1]
            result, embeddings, confs = evaluate(model, test_dataloader, n_top, device)
            
            #print(test_log_data)
            #print(test_conf_data)
            #print(len(test_log_data), len(test_log_data[0]))
            #print(len(embeddings), len(embeddings[0]))
            
            log.append(test_log_data)
            conf.append(test_conf_data)
            # embedding of each log sequence 
            embed.append(embeddings)
            
            if data_type == "test":
                # the faulty value which generate this faulty logs (test_logs_tmp)
                test_conf_val = files_dict_val[test_conf][test_logs_tmp_index]
                conf_val.append(test_conf_val)
                conf_name.append(test_conf)
                conf_name_str.append(test_conf)
                #print(test_conf, test_conf_val) 

    log_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_log.pickle"
    conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_conf.pickle"
    embed_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_embed.pickle"
    
    if ff_dim == 128 or ff_dim == 256:
        log_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{ff_dim}_{rep}_{data_type}_{train_data_type}_log.pickle"
        conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{ff_dim}_{rep}_{data_type}_{train_data_type}_conf.pickle"
        embed_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{ff_dim}_{rep}_{data_type}_{train_data_type}_embed.pickle"
    #log_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_log.pickle"
    #conf_path =f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_conf.pickle"
    #embed_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_embed.pickle"
    
    if label_type == "label_with_value":
        log_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_log.pickle"
        conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_conf.pickle"
        embed_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_embed.pickle"

        #log_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_log_with_value.pickle"
        #conf_path =f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_conf_with_value.pickle"
        #embed_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_embed_with_value.pickle"

    if data_type == "test":
        
        if label_type == "label_with_value":
            conf_val_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_conf_val_with_value.pickle"
            conf_name_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_conf_name_with_value.pickle"
            conf_name_str_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_conf_name_str_with_value.pickle"

            #conf_val_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_conf_val_with_value.pickle"
            #conf_name_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_conf_name_with_value.pickle"
            #conf_name_str_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_conf_name_str_with_value.pickle"
        else:
            conf_val_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_conf_val.pickle"
            conf_name_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_conf_name.pickle"
            conf_name_str_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_{data_type}_{train_data_type}_conf_name_str.pickle"

            if ff_dim == 256 or ff_dim == 128:
                conf_val_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{ff_dim}_{rep}_{data_type}_{train_data_type}_conf_val.pickle"
                conf_name_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{ff_dim}_{rep}_{data_type}_{train_data_type}_conf_name.pickle"
                conf_name_str_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{ff_dim}_{rep}_{data_type}_{train_data_type}_conf_name_str.pickle"

            #conf_val_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_conf_val.pickle"
            #conf_name_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_conf_name.pickle"
            #conf_name_str_path = f"{save_path}{app_file_name}_{conf_file_name}_mslength_{max_seq_length}_epochs_{n_epochs}_{rep}_conf_name_str.pickle"
    util.save_pickle(log, log_path)
    util.save_pickle(conf, conf_path)
    util.save_pickle(embed, embed_path)
    if data_type == "test":
        util.save_pickle(conf_val , conf_val_path)
        util.save_pickle(conf_name, conf_name_path) 
        util.save_pickle(conf_name_str, conf_name_str_path)
           
    #print(f"Length of data({data_type}): {len(log)}, {len(conf)}, {len(embed)}, {len(seq_dict)}")

def loss_fig(tr_loss, val_loss, loss_save_dirc, train_app_to_name, val_app_to_name, target_conf_to_name, max_seq_length, n_epochs, hidden_dim, layers, attention_heads, rep):
    
    plt.plot(tr_loss, color = 'blue')
    plt.plot(val_loss, color = 'red')
    plt.xticks(np.arange(0, n_epochs, 10))
    plt.grid(visible = True)
    
    plt.savefig(f"{loss_save_dirc}/1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}.png")

def count_item(file_dict):
    
    totFiles = 0 
    for k, v in file_dict.items():
        totFiles += len(v)
    return totFiles 
    
def train_main(args, target_app, test_apps, val_apps):
    
    system = args.system 
    rep = args.rep
    raw_path = args.raw_path 
    train_file_path = args.train_file_path 
    validation_file_path = args.validation_file_path
    test_file_path = args.test_file_path 
    train_file_name = args.train_file_name 
    train_app_path = args.train_app_path
    validation_app_path = args.validation_app_path 
    test_app_path = args.test_app_path 
    # path to save used configuration 
    target_conf_path = args.target_conf_path
    log_file_name = args.log_file_name 
    
    model_save_dirc = args.model_save_dirc
    result_save_dirc = args.result_save_dirc
    loss_save_dirc = args.loss_save_dirc
    
    app_config_path = args.app_config_path
    
    batch_size = args.batch_size 
    # number of training epochs 
    n_epochs = args.epochs 
    # hidden dimension of encoder output
    hidden_dim = args.hidden_dim
    # maximum sequence length of transformer 
    max_seq_length = args.max_seq_length 
    if_sliding = args.if_sliding
    # number of attention heads in transformer 
    attention_heads = args.attention_heads  
    # number of layers in transformer 
    layers = args.layers
    return_cls = args.return_cls 
    feedforward_dim = args.feedforward_dim 
    
    n_top = args.n_top
    test_type = args.test_type
    embed_save_path =args.embed_save_path
    label_type = args.label_type
    
    setting_number_of_labels = args.setting_number_of_labels
    
    train_data_type = args.train_data_type 
    limit_test_normal = args.limit_test_normal
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')     
    
    print(f"Test App: {test_apps}, validation app: {val_apps}") 

    pad_token = 0
    
    train_app_to_name = train_app_path.split("/")[-1].replace(".txt", "")
    target_conf_to_name = target_conf_path.split("/")[-1].replace(".txt", "")

    train_file_path_files = train_file_name + f"{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_train_files"
    train_file_path_log = train_file_name + f"{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_train_log"
    train_file_path_conf = train_file_name + f"{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_train_conf"
    train_file_path_conf_val = train_file_name + f"{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_train_conf_val"
    train_file_path_log_index = train_file_name + f"{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_train_log_index"
    train_file_path_conf_index = train_file_name + f"{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_train_conf_index"
                
    if label_type == "label_with_value":
        train_file_path_files = train_file_path_files + "_with_value"
        train_file_path_log = train_file_path_log + "_with_value"
        train_file_path_conf = train_file_path_conf + "_with_value"
        train_file_path_conf_val = train_file_path_conf_val + "_with_value"
        train_file_path_log_index = train_file_path_log_index + "_with_value"
        train_file_path_conf_index = train_file_path_conf_index + "_with_value"

    print(train_file_path_log)
    
    if not os.path.exists(train_file_path_log) or not os.path.exists(train_file_path_conf):
        print("Make new dataset")
        train_files = util.get_all_files_preprocess_template(train_file_path, target_app, log_file_name)
        train_files = util.remove_none_file(train_files)
        train_files_dict, train_files_dict_val, _, totFiles = util.get_data_dict(label_type, train_data_type, train_files, system)
        #train_files_dict, train_files_dict_val, _, totFiles = util.get_data_dict(label_type, "abnormal", train_files)
        # slicing with size of max_sequence(same as window_size)
        if train_data_type == "both":
            train_files_dict, train_files_dict_val, totFiles = util.limit_normal_data(train_files_dict, train_files_dict_val)
        
        conf_set = set()
        train_log, train_conf = [], [] 
        for t_idx, (t_conf, t_logs) in enumerate(train_files_dict.items()):
            conf_set.add(t_conf)
            print(t_conf, len(t_logs))
            for idx,t_log in enumerate(t_logs):
                train_log, train_conf = util.get_dataset(t_log, t_conf, max_seq_length, train_log, train_conf, if_sliding) 
    
        # save used configuration (types of labels)
        conf_set = list(conf_set)
        conf_set.sort()
        fw = open(f'{target_conf_path}', "w")
        for c in conf_set:
            fw.write(str(c) + "\n")
        fw.close()

        train_log, train_conf, log_index_dict, conf_index_dict, conf_index_dict_reverse, input_dim, num_labels = util.get_dict_dataset_train(train_log, train_conf) 
                        
        train_log = list(map(util.padding, train_log, repeat(max_seq_length), repeat(pad_token)))
    
        util.save_pickle(train_files_dict, train_file_path_files)
        util.save_pickle(train_log, train_file_path_log)
        util.save_pickle(train_conf, train_file_path_conf)
        util.save_pickle(train_files_dict_val, train_file_path_conf_val)
        util.save_pickle(log_index_dict, train_file_path_log_index)
        util.save_pickle(conf_index_dict, train_file_path_conf_index)
        
    else:
        print("Use pre-made dataset")
        train_files_dict = util.load_pickle(train_file_path_files) 
        train_files_dict_val = util.load_pickle(train_file_path_conf_val)
        train_log = util.load_pickle(train_file_path_log)
        train_conf = util.load_pickle(train_file_path_conf)
        log_index_dict = util.load_pickle(train_file_path_log_index)
        conf_index_dict = util.load_pickle(train_file_path_conf_index)
        
        totFiles = count_item(train_files_dict)
        
        input_dim = util.get_log_set(train_log)
        #num_labels = len(target_conf)
        num_labels = len(conf_index_dict)
        
        #for k, v in conf_index_dict.items():
        #    print(k, v)
   
     
    print(f"# of used files: {totFiles}")
    train_log, train_conf = util.shuffle(train_log, train_conf)
    
    # split train log, conf into validation log and conf 
    #train_log, val_log, train_conf, val_conf = train_test_split(train_log, train_conf, test_size = validation_prop, random_state = 321)
    train_dataset = ConfLogDataset(torch.tensor(train_log, dtype = torch.long), torch.tensor(train_conf))
    # train dataset should be shuffled 
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle = True)
    
    val_app_to_name = validation_app_path.split("/")[-1].replace(".txt", "")

    val_file_path_files = train_file_name + f"{val_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_val_files_dict"
    val_file_path_log = train_file_name + f"{val_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_val_log"
    val_file_path_conf = train_file_name + f"{val_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_val_conf"

    if label_type == "label_with_value":
        val_file_path_files = val_file_path_files + "_with_value"
        val_file_path_log = val_file_path_log + "_with_value"
        val_file_path_conf = val_file_path_conf + "_with_value"

    if not os.path.exists(val_file_path_log) or not os.path.exists(val_file_path_conf):
         
        print("Making Validation dataset")
        val_files = util.get_all_files_preprocess_template(validation_file_path, val_apps, log_file_name)
        val_files = util.remove_none_file(val_files)
        val_files_dict, val_files_dict_val, _, totValFiles = util.get_data_dict(label_type, train_data_type, val_files, system)

        if train_data_type == "both":
            val_files_dict, _, totValFiles = util.limit_normal_data(val_files_dict, val_files_dict_val)
            
        val_log, val_conf = [], [] 
        for v_conf, v_logs in val_files_dict.items():
            for v_log in v_logs:
                val_log, val_conf = util.get_dataset(v_log, v_conf, max_seq_length, val_log, val_conf, if_sliding)
        val_log, val_conf, val_conf_n = util.get_dict_dataset_val(val_log, val_conf, log_index_dict, conf_index_dict)
        val_log = list(map(util.padding, val_log, repeat(max_seq_length), repeat(pad_token)))

        util.save_pickle(val_log, val_file_path_log)
        util.save_pickle(val_conf, val_file_path_conf)
        util.save_pickle(val_files_dict, val_file_path_files)
        
    else:
        print("Use pre-made validation dataset")
        val_log = util.load_pickle(val_file_path_log)
        val_conf = util.load_pickle(val_file_path_conf) 
        val_files_dict = util.load_pickle(val_file_path_files)
        
        val_conf_copy = val_conf.copy()
        val_conf_n = len(set(val_conf_copy))
        
        totValFiles = count_item(val_files_dict)
    
    val_dataset = ConfLogDataset(torch.tensor(val_log, dtype = torch.long), torch.tensor(val_conf))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    # extending the number of labels 
    if setting_number_of_labels == True:
        new_config_set = util.load_all_config_of_train_app(train_app_path, app_config_path)
        if train_data_type == "both":
            new_config_set.append("normal")
        pre_num_labels = num_labels
        num_labels = len(new_config_set)
        
        new_config_set.sort()
        fw = open(f'{target_conf_path}', "w")
        for c in new_config_set:
            fw.write(str(c) + "\n")
        fw.close()
        print(f"Number of labels is changed! Before:{pre_num_labels}, New: {num_labels}")

    
    print(f"# of used validation files: {totValFiles}")
    
    print(f"input dimension: {input_dim}")
    print(f"number of labels: {num_labels}")
    print(f"train data, validation data: {len(train_dataset)}, {len(val_dataset)}")
    print(f"# of configuration in train_data, validation_data: {len(train_files_dict)}, {len(val_files_dict)}")
    
    if len(val_files_dict) > len(train_files_dict):        
        print("Number of configuration in validation is larger than train data!")
        print(f"Number of configuration in validation dataset: {val_conf_n}")
    
    # setting hyperparameters 
    enc_layers = layers
    enc_heads = attention_heads
    enc_pf_dim = feedforward_dim
    enc_dropout = 0.1 
    clip = 1 
    best_train_loss = float('inf')
    
    # parameter for model name
    # 1. train app, val app, test app
    # 2. configuration file (number of labels are included)
    # 3. max sequence length
    # 4. model info (epochs, layer, size of hidden dim, attention heads)
    # 5. number of repetitions 
    model_name = f"{model_save_dirc}/1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}"
    if feedforward_dim == 128 or feedforward_dim == 256:
        model_name = f"{model_save_dirc}/1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{feedforward_dim}_{rep}"

    # setting model     
    enc = Encoder(input_dim, hidden_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device, max_seq_length, return_cls)
    model = TransformerEncoder(hidden_dim, num_labels, enc, device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)
    #model = nn.DataParallel(model, device_ids = [1, 2, 4])
    
    # define optimizer and loss function 
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    tr_loss_list, val_loss_list = [], [] 
    # training  
    for epoch in range(n_epochs):
        start_time = time.time()
        
        train_loss, _, _ = train(epoch, model, train_dataloader, optimizer, criterion, clip, device)
        valid_loss  = validation(model, val_dataloader, criterion, device)
        print(f"{epoch} Train Loss: {train_loss}")
        print(f"{epoch} Validation Loss: {valid_loss}")
        # save the model when train loss is minumum
        if train_loss < best_train_loss:
            best_train_loss = train_loss  
            print(f"Save model: {train_loss}")
            torch.save(model.state_dict(), model_name)
        else:
            print(f"The validation loss is increased! Not save this model!")
        tr_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)

    loss_fig(tr_loss_list, val_loss_list, loss_save_dirc, train_app_to_name, val_app_to_name, target_conf_to_name, max_seq_length, n_epochs, hidden_dim, layers, attention_heads, rep)
    
    model.load_state_dict(torch.load(model_name))
        
    test_app_to_name = test_app_path.split("/")[-1].replace(".txt", "")
    test_data_val_path = train_file_name + f"{test_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_test_data_val_pickle"
    test_data_path = train_file_name + f"{test_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_test_data.pickle"
    test_files_path = train_file_name + f"{test_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length}_{train_data_type}_test_files.pickle"
    
    if not os.path.exists(test_data_path):
        print("Making test dataset")
        test_files = util.get_all_files_preprocess_template(test_file_path, test_apps, log_file_name)
        test_files = util.remove_none_file(test_files)
        test_files_dict, test_files_dict_val, _, totTestFiles = util.get_data_dict(label_type, train_data_type, test_files, system)
        
        if train_data_type == "both":
            if limit_test_normal == 1:
                test_files_dict, test_files_dict_val, totTestFiles = util.limit_normal_data(test_files_dict, test_files_dict_val) 
            
        with open(test_data_path, "wb") as fw:
            pickle.dump(test_files_dict, fw)
        
        with open(test_data_val_path, "wb") as fw:
            pickle.dump(test_files_dict_val, fw)
            
        with open(test_files_path, "wb") as fw:
            pickle.dump(test_files, fw)
        
        print(f"# of used test files: {totTestFiles}")
    else:
        print("Test dataset is already generated!")
        with open(test_files_path, "rb") as fr:
            test_files = pickle.load(fr)
        with open(test_data_path, "rb") as fr:
            test_files_dict = pickle.load(fr)
        with open(test_data_val_path, "rb") as fr:
            test_files_dict_val = pickle.load(fr)
        totTestFiles = count_item(test_files_dict)
        print(f"# of used test files: {totTestFiles}")
                    
    top_1_case, top_5_case, top_10_case, tot_case, mrr = 0, 0, 0, 0, 0 
    max_top_1_case, max_top_5_case, max_top_10_case, max_mrr = 0, 0, 0, 0
    
    tot_test_log_data, tot_test_conf_data, tot_test_log_embedding = [], [], [] 
    print(f"Test type: {test_type}")
    
    # save embedding data 
    if test_type == "data_save" or test_type == "abnormal" or test_type == "both":
        #normal_target_conf_path = "./param/target_conf_normal.txt"
       
        #target_conf_to_name_normal = normal_target_conf_path.split("/")[-1].replace(".txt", "")
        #normal_target_conf = util.get_text_data(normal_target_conf_path)
        #normal_test_files_dict, normal_test_files_dict_val, normal_test_files_filename = util.get_data_dict("normal", "test", test_files, normal_target_conf, target_app, raw_path)
        data_save(embed_save_path, model, "train", train_data_type, train_files_dict, train_files_dict_val, label_type, log_index_dict, conf_index_dict, rep, n_epochs, n_top, batch_size, device, pad_token, if_sliding, train_app_to_name, val_app_to_name, target_conf_to_name, max_seq_length, hidden_dim, layers, attention_heads, feedforward_dim)
        data_save(embed_save_path, model, "val", train_data_type, val_files_dict, [], label_type, log_index_dict, conf_index_dict, rep, n_epochs, n_top, batch_size, device, pad_token, if_sliding, train_app_to_name, val_app_to_name, target_conf_to_name, max_seq_length, hidden_dim, layers, attention_heads, feedforward_dim)
        data_save(embed_save_path, model, "test", train_data_type, test_files_dict, test_files_dict_val, label_type, log_index_dict, conf_index_dict, rep, n_epochs, n_top, batch_size, device, pad_token, if_sliding, train_app_to_name, val_app_to_name, target_conf_to_name, max_seq_length, hidden_dim, layers, attention_heads, feedforward_dim)
    
        #data_save(rep, n_epochs, label_type, train_app_to_name, target_conf_to_name, train_files_dict, train_files_dict_val, log_index_dict, conf_index_dict, max_seq_length, n_top, device, model, pad_token, if_sliding, batch_size, "train", embed_save_path)
        #data_save(rep, n_epochs, label_type, val_app_to_name, target_conf_to_name, val_files_dict, [], log_index_dict, conf_index_dict, max_seq_length, n_top, device, model, pad_token, if_sliding, batch_size, "val", embed_save_path)
        #data_save(rep, n_epochs, label_type, test_app_to_name, target_conf_to_name, test_files_dict, test_files_dict_val, log_index_dict, conf_index_dict, max_seq_length, n_top, device, model, pad_token, if_sliding, batch_size, "test", embed_save_path)
        #data_save(test_app_to_name, target_conf_to_name_normal, normal_test_files_dict, normal_test_files_dict_val, log_index_dict, conf_index_dict, max_seq_length, n_top, device, model, pad_token, if_sliding, batch_size, "test", embed_save_path, len(target_app), "normal")
        
    if test_type == "abnormal" or test_type == "data_save" or test_type == "both":
        print("Testing abnormal configuration!")
        print(f"# of configuration in testset: {len(test_files_dict)}")
        #result_per_conf = []
        #tot_index, tot_result = [], [] 
        #columns = [i for i in range(num_labels)]
        #columns.insert(0, 0)
        #tot_df = pd.DataFrame(columns =columns)
        #tot_df_index = 0 
        test_all_case_rank= [] 
        test_cases = 0
        tot_test_cases = 0 
        for test_conf, test_logs in test_files_dict.items():
            #test_conf_type = get_type_of_conf(test_conf, configuration_file_path)
            #test_conf_type = test_conf_type.to_string().split()[1]
            tmp_max_mrr, tmp_avg_mrr = 0, 0
            tmp_avg_r1, tmp_avg_r5, tmp_avg_r10, tmp_max_r1, tmp_max_r5, tmp_max_r10 = 0, 0, 0, 0, 0, 0  
            for test_logs_tmp_index, test_logs_tmp in enumerate(test_logs):
                tot_frequency, tot_test_log, tot_test_prob = [], [], [] 
                test_conf_val = test_files_dict_val[test_conf][test_logs_tmp_index]
                if "/" in test_conf_val:
                    test_conf_val = test_conf_val.replace("/", "%")
                test_log_data, test_conf_data = util.get_dataset(test_logs_tmp, test_conf, max_seq_length, [], [], if_sliding)
                test_log_data, test_conf_data = util.get_dict_dataset_test(test_log_data, test_conf_data, log_index_dict, conf_index_dict)
                test_log_data = list(map(util.padding, test_log_data, repeat(max_seq_length), repeat(pad_token)))

                test_conf_data_flag = test_conf_data.copy()
                test_conf_data_flag = list(set(test_conf_data_flag))
                if test_conf_data_flag[0] == -1:
                    print(f"SKIP!!! Configuration ({test_conf}) is not included in training dataset!")
                    continue 
                
                # if test_conf was not in training dataset, then skip 
                # if more than one data contain in test_conf_data, then error 
                if len(test_log_data) == 0 and len(test_conf_data) == 0:
                    continue
                #test_log_data_frequency = get_data_frequency(test_log_data, train_log, train_conf)
                test_dataset = ConfLogDataset(torch.tensor(test_log_data, dtype=torch.long), torch.tensor(test_conf_data))
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
                result, embeddings, confs = evaluate(model, test_dataloader, n_top, device)
                #tot_embeddings.extend(embeddings)
                #tot_confs.extend(confs)
                test_cases += len(result)
                
                tot_score = get_case_score(result, num_labels)
                tot_prob = get_max_prob(result, num_labels)
                            
                #tot_frequency.extend(test_log_data_frequency)
                tot_test_log.extend(test_log_data) 
                tot_test_prob.extend(result)
                        
                # sorting as ascending order (if tot_score = [0.5, 0.3, 0.2] -> s_sort = [2, 1, 0])
                test_conf_data = list(set(test_conf_data))
                if len(test_conf_data) != 1:
                    print("check test configuration data!!!!")
                    sys.exit(0)
                test_conf_data = test_conf_data[0] 
                tot_test_cases += 1 
                # save index, prob of each configuration as dataframe 
                prob = tot_prob.copy()
                prob.insert(0, test_conf_data) 
                #tot_df.loc[tot_df_index] = prob
                #tot_df_index += 1 

                # sorted index of score 
                s_sort = np.argsort(tot_score, axis = 0)
                max_sort = np.argsort(tot_prob, axis = 0)
                            
                rank = 0
                for idx in range(len(s_sort) - 1, -1, -1):
                    rank += 1
                    if s_sort[idx] == test_conf_data:
                        mrr += (1 / rank)
                        tmp_avg_mrr += (1 / rank)
                        break 
                    
                max_rank = 0
                for idx in range(len(max_sort) - 1, -1, -1):
                    max_rank += 1 
                    if max_sort[idx] == test_conf_data:
                        max_mrr += (1 / max_rank)
                        tmp_max_mrr += (1 / max_rank) 
                        break  
                
                '''os.makedirs(f"./result_{test_type}/", exist_ok=True)
                f_name = f"./result_{test_type}/{test_conf}_{test_conf_data}_{test_conf_val}_{rep}"
                fw_result = open(f'{f_name}.txt', "w")
                # test_conf: name of configuration, test_conf_data: answer_configuration, rank 
                with open(f'{f_name}.pickle', "wb") as fw:
                    pickle.dump(tot_test_log, fw)
                
                

                columns = [i for i in range(num_labels)]
                df = pd.DataFrame(columns = columns)
                for t_idx, t_prob in enumerate(tot_test_prob):
                    #print(t_prob)
                    fw_result.write(f"{t_idx}: {str(tot_test_log[t_idx])}\n")
                    fw_result.write(f"{t_idx}: {t_prob}\n")

                    df = df.append(pd.Series(t_prob, index = df.columns), ignore_index = True)
                
                #print(df)
                df.to_excel(f'{f_name}.xlsx')
                fw_result.close()'''
                
                #print(test_conf, test_conf_type, test_conf_data, rank)
                #print(s_sor            
                tmp_a = [] 
                for l in test_log_data:
                    tmp_a.extend(l)
                #print(test_conf, test_conf_data, rank, max_rank, len(target_app), test_logs_tmp_index)
                #print(test_logs_tmp)
                #print(tmp_a)
                #print(s_sort)
                #print(max_sort)
                    
                #tot_result = [] 
                top_1 = s_sort[-1:]
                top_5 = s_sort[-5:]
                top_10 = s_sort[-10:]
                max_top_1 = max_sort[-1:]
                max_top_5 = max_sort[-5:]
                max_top_10 = max_sort[-10:]
                tot_case += 1 

                if test_conf_data in top_1:
                    top_1_case += 1 
                    tmp_avg_r1 += 1 
                if test_conf_data in top_5:
                    top_5_case += 1
                    tmp_avg_r5 += 1 
                if test_conf_data in top_10:
                    top_10_case += 1
                    tmp_avg_r10 += 1      
                if test_conf_data in max_top_1:
                    max_top_1_case += 1
                    tmp_max_r1 += 1  
                if test_conf_data in max_top_5:
                    max_top_5_case += 1 
                    tmp_max_r5 += 1 
                if test_conf_data in max_top_10:
                    max_top_10_case += 1
                    tmp_max_r10 += 1  
                
                print(f"Testing configuration: {test_conf_data}, length of log: {len(test_log_data)}")                
                test_all_case_rank.extend(get_tmp_case_result(result, test_conf_data))
                
            print(f"length of all rank: {len(test_all_case_rank)} (compare: {test_cases})")
            r_tmp = {}
            r_tmp['conf'] = test_conf
            r_tmp['tot_case'] = len(test_logs)
            r_tmp['mrr'] = tmp_avg_mrr / len(test_logs)
            r_tmp['mrr_max'] = tmp_max_mrr / len(test_logs)
            #r_tmp['mrr'] = sum(tmp_avg_mrr) / len(tmp_avg_mrr)
            #r_tmp['top_mrr'] = sum(tmp_max_mrr) / len(tmp_max_mrr)
            r_tmp['R@1'] = tmp_avg_r1
            r_tmp['R@1_max'] = tmp_max_r1    
            r_tmp['R@5'] = tmp_avg_r5
            r_tmp['R@5_max'] = tmp_max_r5
            r_tmp['R@10'] = tmp_avg_r10
            r_tmp['R@10_max'] = tmp_max_r10
            
            #result_per_conf.append(r_tmp)
        '''os.makedirs("./result/", exist_ok=True)
        tot_df.to_excel(f'./result/{test_type}_{rep}.xlsx')'''
    
    '''if test_type == "normal":

        if not os.path.exists(f"./result_{test_type}/"):
            os.mkdir(f"./result_{test_type}/")
        fw_predict = open('predicted_as_abnormal.txt', "w")
        print("Testing normal configuration!")
        normal_target_conf_path = "./param/target_conf_normal.txt"
        normal_target_conf = util.get_text_data(normal_target_conf_path)
        normal_test_files_dict, normal_test_files_dict_val, normal_test_files_filename, _ = util.get_data_dict(label_type, "normal", "test", test_files, normal_target_conf, target_app, raw_path)
        columns = [i for i in range(num_labels)]
        columns.insert(0, 0)
        normal_tot_df = pd.DataFrame(columns = columns)
        normal_tot_df_index = 0 
        for test_conf, normal_logs in normal_test_files_dict.items():
            print(f"Configuration: {test_conf}, # of logs: {len(normal_logs)}")
            for normal_log_index, normal_logs_tmp in enumerate(normal_logs):
                max_prob = 0 
                normal_filename = normal_test_files_filename[test_conf][normal_log_index]
                normal_conf_val = normal_test_files_dict_val[test_conf][normal_log_index]
                if "/" in normal_conf_val:
                    normal_conf_val = normal_conf_val.replace("/", "%")

                #f_name = f"./result_{test_type}/{test_conf}_{normal_conf_val}"
                f_name = f"./result_{test_type}/{normal_filename}"
                normal_log_data, normal_conf_data = util.get_dataset(normal_logs_tmp, test_conf, max_seq_length, [], [], if_sliding)
                normal_log_data, normal_conf_data = util.get_dict_dataset_test(normal_log_data, normal_conf_data, log_index_dict, conf_index_dict)
                normal_log_data = list(map(util.padding, normal_log_data, repeat(max_seq_length), repeat(pad_token)))
                
                if len(normal_log_data) == 0 and len(normal_conf_data) == 0:
                    continue   
                test_dataset = ConfLogDataset(torch.tensor(normal_log_data, dtype = torch.long), torch.tensor(normal_conf_data)) 
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
                # result of this normal configuration log, (result of all sequence)
                result, _, _ = evaluate(model, test_dataloader, n_top, device)
                normal_max_prob = get_max_prob(result, num_labels)
                
                normal_max_prob.insert(0, -1)
                normal_tot_df.loc[normal_tot_df_index] = normal_max_prob 
                normal_tot_df_index += 1 
                df = pd.DataFrame(columns = [i for i in range(num_labels)]) 
                for idx, r in enumerate(result):
                    df.loc[idx] = r
                    max_r = max(r)
                    if max_r > max_prob:
                        max_prob = max_r 
            
                df.to_excel(f'{f_name}.xlsx')
                
                fw = open(f'{f_name}.txt', "w")
                for n_idx, a in enumerate(normal_log_data):
                    fw.write(f"{str(n_idx)}: {a}\n")
                    fw.write(f"{str(n_idx)}: {result[n_idx]}\n")
                fw.close()
                with open(f'{f_name}.pickle', "wb") as fw:
                    pickle.dump(normal_log_data, fw)
                if max_prob > 0.7:
                    print(test_conf, normal_conf_val)
                    fw_predict.write(f"{str(test_conf)} {str(normal_conf_val)}\n")
        normal_tot_df.to_excel(f'./result/{test_type}.xlsx')'''
    
    mrr_test_case = util.eval_mrr(test_all_case_rank)
    r1_test_case, r5_test_case, r10_test_case = 0, 0, 0
    for r in test_all_case_rank:
        if r <= 1:
            r1_test_case += 1 
        if r <= 5:
            r5_test_case += 1 
        if r <= 10:
            r10_test_case += 1 
    print(mrr_test_case, r1_test_case / len(test_all_case_rank), r5_test_case / len(test_all_case_rank), r10_test_case/len(test_all_case_rank))
    
    
    if tot_case == 0:
        return 0, 0, 0, 0
    else:
        mrr /= tot_case 
        top_1_tot = top_1_case / tot_case
        top_5_tot = top_5_case / tot_case
        top_10_tot = top_10_case / tot_case
        max_mrr /= tot_case 
        max_top_1_tot = max_top_1_case / tot_case 
        max_top_5_tot = max_top_5_case / tot_case 
        max_top_10_tot = max_top_10_case / tot_case 
        
        print(f"{tot_case} org: {mrr}, {top_1_tot}, {top_5_tot}, {top_10_tot}\n")
        print(f"{tot_case} max: {max_mrr}, {max_top_1_tot}, {max_top_5_tot}, {max_top_10_tot}\n")
        
        result_dict = {}
        result_dict['sum'] = {}
        result_dict['max'] = {}
        result_dict['sum']['mrr'] = mrr 
        result_dict['sum']['top_1_tot'] = top_1_tot 
        result_dict['sum']['top_5_tot'] = top_5_tot
        result_dict['sum']['top_10_tot'] = top_10_tot 
        
        result_dict['max']['mrr'] = max_mrr
        result_dict['max']['top_1_tot'] = max_top_1_tot 
        result_dict['max']['top_5_tot'] = max_top_5_tot 
        result_dict['max']['top_10_tot'] = max_top_10_tot 

        if feedforward_dim == 128 or feedforward_dim == 256:
            result_path = f"{result_save_dirc}/1tr_{train_app_to_name}_{val_app_to_name}_{test_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{feedforward_dim}_{rep}_testresult_sum_max_mrr_r@k.pickle"
        else:
            result_path = f"{result_save_dirc}/1tr_{train_app_to_name}_{val_app_to_name}_{test_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_testresult_sum_max_mrr_r@k.pickle"
        with open(result_path, "wb") as fw:
            pickle.dump(result_dict, fw)
            
        test_dict = {'case_num': tot_test_cases, 'ranking': test_all_case_rank}
        if feedforward_dim == 128 or feedforward_dim == 256:
            result_path = f'{result_save_dirc}/1tr_{train_app_to_name}_{val_app_to_name}_{test_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{feedforward_dim}_{rep}_testresult_all_test_case_sum_max_mrr_r@k.pickle'
        else:
            result_path =f'{result_save_dirc}/1tr_{train_app_to_name}_{val_app_to_name}_{test_app_to_name}_{target_conf_to_name}_{max_seq_length}_{n_epochs}_{hidden_dim}_{layers}_{attention_heads}_{rep}_testresult_all_test_case_sum_max_mrr_r@k.pickle'
        print(result_path)
        with open(result_path, "wb") as fw:
            pickle.dump(test_dict, fw)
            
        return mrr, top_1_tot, top_5_tot, top_10_tot
    
    
