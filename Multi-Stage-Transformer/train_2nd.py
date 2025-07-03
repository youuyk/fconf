from dataset_seq_embed import ConfLogDataset
import torch.nn as nn 
import torch 
from itertools import repeat 
from model_seq_embed import Encoder, TransformerEncoder
import time, os, sys, pickle  
import numpy as np   
import pandas as pd 
from collections import OrderedDict
import matplotlib.pyplot as plt 
import warnings
import util
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
        seq_embed = batch[0]
        padded_index = batch[1]
        conf = batch[2]
        
        #print("="*30)
        #print(log)
        #print(padded_index)
        #print(conf)
        seq_embed = seq_embed.to(device)
        padded_index = padded_index.to(device)
        conf = conf.to(device)
        
        optimizer.zero_grad()
        
        output, predict = model(seq_embed, padded_index)
        
        loss = criterion(predict, conf)
        if i % 5000 == 0:
            print(f"loss_{i}: {loss}")
        train_loss.append(loss.item())
        x.append(i)
        loss.backward()
    
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss/ len(dataloader), train_loss, x    

def validation(model, dataloader, criterion, device):
    
    model.eval()
    total_loss = 0     
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            
            seq_embed = batch[0]
            padded_index = batch[1]
            conf = batch[2] 
            conf = conf.to(device)
            
            seq_embed = seq_embed.to(device) 
            padded_index = padded_index.to(device)
            
            output, predict = model(seq_embed, padded_index)
            
            loss = criterion(predict, conf)
            loss = loss.item()
            total_loss += loss  
    
    return total_loss / len(dataloader)        
    
def evaluate(model, test_dataloader, device):
    
    model_predicts, answer =[], [] 
    model.eval()
    softmax = nn.Softmax(dim = 1)
    model_predicts_embed = [] 
    
    with torch.no_grad():
        
        for i, batch in enumerate(test_dataloader):
            seq_embed = batch[0]
            padded_index = batch[1]
            conf = batch[2]
            
            seq_embed = seq_embed.to(device)
            padded_index = padded_index.to(device)
            
            # output : embedding of each log sequence     
            output, predict = model(seq_embed, padded_index)
            
            if i == 0:
                print(f"Number of labels: {len(predict[0])}")
            
            #print(output)
            # shape of output: (batch_size, hidden_dim)
            output = output.cpu().numpy()
            model_predicts_embed.extend(output)

            #print(output.shape)
            #output = output.cpu().numpy()
            #conf = conf.cpu().numpy()
            #embeddings.extend(output)
            #confs.extend(conf)
            
            s = softmax(predict)
            s = s.detach().cpu().numpy()
            #s_sort = np.argsort(s, axis = 1)
            
            model_predicts.extend(s)
            conf = conf.detach().cpu().numpy()
            answer.extend(conf)
            
    # tot_result: [# of sequence, num_labels]
    # embeddings: [# of sequence, # of features]
    # conf: [# of sequence, 1]
    return model_predicts, answer, model_predicts_embed
    
def evaluate_pre(conf_index_dict, result_save_dirc, rep, train_data_type, test_data_path, test_conf_path, max_seq_length, max_seq_length_1tr, n_epochs_1tr, n_epochs, hidden_dim_1tr, hidden_dim, layers_1tr, layers, attention_heads_1tr, attention_heads, rep_1tr, batch_size, model, device, target_conf_to_name, train_app_to_name, val_app_to_name, test_app_to_name, ff_dim):
    
    test_dataset, _, _ = util.make_embed_dataset(test_data_path, test_conf_path, max_seq_length)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    # model_predict: probability of each configuration 
    # answer: labels
    # model_predict_embed: hidden_state from transformer for each log_sequence 
    model_predict, answer, model_predict_embed = evaluate(model, test_dataloader, device)
    #util.make_embedding_plot(answer, model_predict_embed, 2, fig_save_name)
        
    sorted_index = list(map(util.sorted_configuration_index, model_predict))
    # if test_type == "normal", then rank is all -1 
    rank = list(map(util.eval_ranking, answer, sorted_index))
    # if test_type == "normal", then prob is all -1 
    prob = list(map(util.eval_get_prob, answer, model_predict))
    highest_prob = list(map(util.eval_get_highest_prob, model_predict))

    conf_result = accuracy_by_conf(answer, rank)

    util.write_result_2tr(answer, conf_index_dict, rank, prob)
    
    if train_data_type == "both":
        rank_of_normal_case = util.get_normal_case(answer, conf_index_dict, rank)
        normal_mrr = util.eval_mrr(rank_of_normal_case)
        normal_r1, normal_r1_count = util.eval_rk(rank_of_normal_case, 1)
        normal_r5, normal_r5_count = util.eval_rk(rank_of_normal_case, 5)
        normal_r10, normal_r10_count = util.eval_rk(rank_of_normal_case, 10)

        rank_of_abnormal_case = util.get_abnormal_case(answer, conf_index_dict, rank)
        abnormal_mrr = util.eval_mrr(rank_of_abnormal_case)
        abnormal_r1, abnormal_r1_count = util.eval_rk(rank_of_abnormal_case, 1)
        abnormal_r5, abnormal_r5_count = util.eval_rk(rank_of_abnormal_case, 5)
        abnormal_r10, abnormal_r10_count = util.eval_rk(rank_of_abnormal_case, 10)
        
    mrr = util.eval_mrr(rank)
    r1, r1_count = util.eval_rk(rank, 1)
    r5, r5_count = util.eval_rk(rank, 5)
    r10, r10_count = util.eval_rk(rank, 10)
    #r1 = list(map(util.eval_successrate, sorted_index, answer, repeat(1)))
    #r5 = list(map(util.eval_successrate, sorted_index, answer, repeat(5)))
    #r10 =list(map(util.eval_successrate, sorted_index, answer, repeat(10)))
    #max_conf = list(map(util.eval_max_conf, sorted_index, model_predict))
    # number of cases for each metric 
    #r1_count = r1.count(True)
    #r5_count = r5.count(True)
    #r10_count = r10.count(True)
    #r1 = r1.count(True) / len(r1)
    #r5 = r5.count(True) / len(r5)
    #r10 = r10.count(True) / len(r10)

    r_dict = {}
    r_dict['case_num'] = len(rank)
    r_dict['mrr'] = mrr  
    r_dict['top_1'] = r1 
    r_dict['top_5'] = r5 
    r_dict['top_10'] = r10 
    
    result_path = f'{result_save_dirc}2tr_{train_app_to_name}_{val_app_to_name}_{test_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{max_seq_length}_{n_epochs_1tr}_{n_epochs}_{hidden_dim_1tr}_{hidden_dim}_{layers_1tr}_{layers}_{attention_heads_1tr}_{attention_heads}_{rep_1tr}_{rep}_testresult_mrr_r@k.pickle'
    if ff_dim == 128 or ff_dim == 256:
        result_path = f'{result_save_dirc}2tr_{train_app_to_name}_{val_app_to_name}_{test_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{max_seq_length}_{n_epochs_1tr}_{n_epochs}_{hidden_dim_1tr}_{hidden_dim}_{layers_1tr}_{layers}_{attention_heads_1tr}_{attention_heads}_{ff_dim}_{rep_1tr}_{rep}_testresult_mrr_r@k.pickle'
    with open(result_path, "wb") as fw:
        pickle.dump(r_dict, fw)
    
    print(f"Test type: {train_data_type} ({len(rank)})")
    print(f"MRR: {mrr}, R@1: {r1}({r1_count}), R@5: {r5}({r5_count}), R@10: {r10}({r10_count})")
    
    if train_data_type== "both":
        r_dict_normal = {}
        r_dict_normal['mrr']= normal_mrr
        r_dict_normal['top_1'] = normal_r1 
        r_dict_normal['top_5'] = normal_r5 
        r_dict_normal['top_10'] = normal_r10 
        normal_path = f"{result_save_dirc}2tr_{train_app_to_name}_{val_app_to_name}_{test_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{max_seq_length}_{n_epochs_1tr}_{n_epochs}_{hidden_dim_1tr}_{hidden_dim}_{layers_1tr}_{layers}_{attention_heads_1tr}_{attention_heads}_{rep_1tr}_{rep}_normal_testresult_mrr_r@k.pickle"
        with open(normal_path, "wb") as fw:
            pickle.dump(r_dict_normal, fw) 
            
        print(f"Result of Normal: {len(rank_of_normal_case)}")
        print(f"MRR: {normal_mrr}, R@1: {normal_r1}({normal_r1_count}), R@5: {normal_r5}({normal_r5_count}), R@10: {normal_r10}({normal_r10_count})")
        print(f"Result of Abnormal: {len(rank_of_abnormal_case)}")
        print(f"MRR: {abnormal_mrr}, R@1: {abnormal_r1}({abnormal_r1_count}), R@5:{abnormal_r5}({abnormal_r5_count}), R@10: {abnormal_r10}({abnormal_r10_count})")

    
    '''conf_index_dict_path = f"./data/{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length_1tr}_train_conf_index"    
    answer_prob = get_answer_prob(conf_index_dict_path, answer, prob)
    rank_dict = get_answer_prob(conf_index_dict_path, answer, rank)
    
    os.makedirs("./result/", exist_ok=True)
    answer_path = f"./result/2ndTR_train_{train_app_to_name}_val_{val_app_to_name}_test_{test_app_to_name}_rep_{rep}_epoch2nd_{n_epochs}_max_seq_length_1st_{max_seq_length_1tr}_max_seq_length_{max_seq_length}_label_{data_type}_{rep}.pickle" 
    predict_embed_path = f"./result/2ndTR_train_{train_app_to_name}_val_{val_app_to_name}_test_{test_app_to_name}_rep_{rep}_epoch2nd_{n_epochs}_max_seq_length_1st_{max_seq_length_1tr}_max_seq_length_{max_seq_length}_hidden_state_{data_type}_{rep}.pickle"
    predict_path = f"./result/2ndTR_train_{train_app_to_name}_val_{val_app_to_name}_test_{test_app_to_name}_rep_{rep}_epoch2nd_{n_epochs}_max_seq_length_1st_{max_seq_length_1tr}_max_seq_length_{max_seq_length}_predict_{data_type}_{rep}.pickle"
    max_conf_path = f"./result/2ndTR_train_{train_app_to_name}_val_{val_app_to_name}_test_{test_app_to_name}_rep_{rep}_epoch2nd_{n_epochs}_max_seq_length_1st_{max_seq_length_1tr}_max_seq_length_{max_seq_length}_predict_max_conf_{data_type}_{rep}.pickle"
    ranking_path = f"./result/2ndTR_train_{train_app_to_name}_val_{val_app_to_name}_test_{test_app_to_name}_rep_{rep}_epoch2nd_{n_epochs}_max_seq_length_1st_{max_seq_length_1tr}_max_seq_length_{max_seq_length}_predict_ranking_{data_type}_{rep}.pickle"
    ansprob_path = f"./result/2ndTR_train_{train_app_to_name}_val_{val_app_to_name}_test_{test_app_to_name}_rep_{rep}_epoch2nd_{n_epochs}_max_seq_length_1st_{max_seq_length_1tr}_max_seq_length_{max_seq_length}_answer_prob_{data_type}_{rep}.pickle"
    rankans_path = f"./result/2ndTR_train_{train_app_to_name}_val_{val_app_to_name}_test_{test_app_to_name}_rep_{rep}_epoch2nd_{n_epochs}_max_seq_length_1st_{max_seq_length_1tr}_max_seq_length_{max_seq_length}_answer_rank_{data_type}_{rep}.pickle"

    util.save_pickle(answer, answer_path)
    util.save_pickle(model_predict_embed, predict_embed_path)
    util.save_pickle(model_predict, predict_path)
    util.save_pickle(max_conf, max_conf_path)
    util.save_pickle(rank, ranking_path)
    util.save_pickle(answer_prob, ansprob_path)
    util.save_pickle(rank_dict, rankans_path)
    
    # save_excel as result 
    if data_type == "abnormal":
        util.save_excel(train_app_to_name, test_app_to_name, val_app_to_name, rep, n_epochs, max_seq_length_1tr, max_seq_length, rank, prob, highest_prob, data_type)'''
        
def get_answer_prob(conf_index_dict_path, answer, prob):
    
    with open(conf_index_dict_path, "rb") as fr:
        conf_index_dict = pickle.load(fr)
    
    ansDict = {}
    for idx, ans in enumerate(answer):
        p = prob[idx]
        for k, v in conf_index_dict.items():
            if v == ans:
                conf_name = k 
                break 
        #print(conf_name, p)
        if conf_name not in ansDict:
            ansDict[conf_name] = []
        tmp = ansDict[conf_name]
        tmp.append(p)
        ansDict[conf_name] = tmp 
    return ansDict
            
def accuracy_by_conf(answer, rank):
    
    conf_set = {}
    for idx, ans in enumerate(answer):
        ans = ans.item()
        if ans not in conf_set:
            conf_set[ans] = []
        tmp = conf_set[ans]
        tmp.append(rank[idx])
        conf_set[ans] = tmp 
    
    conf_set = OrderedDict(sorted(conf_set.items(), key = lambda item:item[0],reverse = False))
    df = pd.DataFrame(columns = ['Configuration', 'MRR'])
    # result of each configuration 
    for idx, (conf, rank_list) in enumerate(conf_set.items()):
        mrr = util.eval_mrr(rank_list)
        df.loc[idx] = [conf, mrr]
    #df.to_excel('./result/mrr_per_conf.xlsx')

def loss_fig(tr_loss_list, val_loss_list, loss_save_dirc, train_app_to_name, val_app_to_name, target_conf_to_name, max_seq_length_1tr, max_seq_length, n_epochs_1tr, n_epochs, hidden_dim_1tr, hidden_dim, layers_1tr, layers, attention_heads_1tr, attention_heads, rep_1tr, rep):

    plt.plot(tr_loss_list, color = 'blue')
    plt.plot(val_loss_list, color = 'red')
    plt.xticks(np.arange(0, n_epochs, 10))
    plt.grid(visible=True)

    plt.savefig(f"{loss_save_dirc}2tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{max_seq_length}_{n_epochs_1tr}_{n_epochs}_{hidden_dim_1tr}_{hidden_dim}_{layers_1tr}_{layers}_{attention_heads_1tr}_{attention_heads}_{rep_1tr}_{rep}.png")

# input: sequence of embedding, labels (configuration)
# output: probability of each configuration (after training linear layer)
def train_main(args):
    
    # train_data, test_data(save as pickle) = dict(key: configuration name, value: sequence of embedding)
    '''train_data_path = args.train_data_path
    train_conf_path = args.train_conf_path
    test_abnormal_data_path = args.test_abnormal_data_path  
    test_abnormal_conf_path = args.test_abnormal_conf_path
    test_normal_data_path = args.test_normal_data_path 
    test_normal_conf_path = args.test_normal_conf_path '''
    embed_save_path = args.embed_save_path 
    train_app_path = args.train_app_path 
    val_app_path = args.validation_app_path
    test_app_path = args.test_app_path 
    target_conf_path = args.target_conf_path 
    
    rep = args.rep
    rep_1tr = args.rep_1tr
    
    model_save_dirc = args.model_save_dirc
    loss_save_dirc = args.loss_save_dirc
    result_save_dirc = args.result_save_dirc 
    
    param_data_path = args.param_data_path 
        
    # parameter for training 
    batch_size = args.batch_size
    # parameters for model setting  
    n_epochs = args.n_epochs 
    n_epochs_1tr = args.n_epochs_1tr
    #hidden_dim = args.hidden_dim 
    max_seq_length = args.max_seq_length 
    max_seq_length_1tr = args.max_seq_length_1tr
    attention_heads_1tr = args.attention_heads_1tr 
    attention_heads = args.attention_heads
    layers = args.layers 
    layers_1tr = args.layers_1tr
    hidden_dim_1tr = args.hidden_dim_1tr
    feedforward_dim = args.feedforward_dim 
    
    train_data_type = args.train_data_type 
    
    setting_number_of_labels = args.setting_number_of_labels 

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    '''train_data_path = f"{train_data_path}_{n_apps}.pickle"
    train_conf_path = f"{train_conf_path}_{n_apps}.pickle"
    test_abnormal_data_path = f"{test_abnormal_data_path}_{n_apps}.pickle"
    test_abnormal_conf_path = f"{test_abnormal_conf_path}_{n_apps}.pickle"
    test_normal_data_path = f"{test_normal_data_path}_{n_apps}.pickle"
    test_normal_conf_path = f"{test_normal_conf_path}_{n_apps}.pickle"'''
    
    train_app_to_name = train_app_path.split("/")[-1].replace(".txt", "")
    target_conf_to_name = target_conf_path.split("/")[-1].replace(".txt", "")
    val_app_to_name = val_app_path.split("/")[-1].replace(".txt", "")
    test_app_to_name = test_app_path.split("/")[-1].replace(".txt", "")
    
    train_data_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{rep_1tr}_train_{train_data_type}_embed.pickle"
    train_conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{rep_1tr}_train_{train_data_type}_conf.pickle"
    val_data_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{rep_1tr}_val_{train_data_type}_embed.pickle"
    val_conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{rep_1tr}_val_{train_data_type}_conf.pickle"
    test_abnormal_data_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{rep_1tr}_test_{train_data_type}_embed.pickle"
    test_abnormal_conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{rep_1tr}_test_{train_data_type}_conf.pickle"
    
    if feedforward_dim == 128 or feedforward_dim == 256:
        train_data_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{feedforward_dim}_{rep_1tr}_train_{train_data_type}_embed.pickle"
        train_conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{feedforward_dim}_{rep_1tr}_train_{train_data_type}_conf.pickle"
        val_data_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{feedforward_dim}_{rep_1tr}_val_{train_data_type}_embed.pickle"
        val_conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{feedforward_dim}_{rep_1tr}_val_{train_data_type}_conf.pickle"
        test_abnormal_data_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{feedforward_dim}_{rep_1tr}_test_{train_data_type}_embed.pickle"
        test_abnormal_conf_path = f"{embed_save_path}1tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{n_epochs_1tr}_{hidden_dim_1tr}_{layers_1tr}_{attention_heads_1tr}_{feedforward_dim}_{rep_1tr}_test_{train_data_type}_conf.pickle"
    
    conf_index_dict_path = f"{param_data_path}{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length_1tr}_{train_data_type}_train_conf_index"
    conf_index_dict = util.read_pickle(conf_index_dict_path)
    #train_data_path = data_path + f"{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length_1tr}_epochs_{n_epochs_1st}_{rep}_embed.pickle"
    #train_conf_path = data_path + f"{train_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length_1tr}_epochs_{n_epochs_1st}_{rep}_conf.pickle"
    #val_data_path = data_path + f"{val_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length_1tr}_epochs_{n_epochs_1st}_{rep}_embed.pickle"
    #val_conf_path = data_path + f"{val_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length_1tr}_epochs_{n_epochs_1st}_{rep}_conf.pickle"
    
    #test_abnormal_data_path = data_path + f"{test_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length_1tr}_epochs_{n_epochs_1st}_{rep}_embed.pickle"
    #test_abnormal_conf_path = data_path + f"{test_app_to_name}_{target_conf_to_name}_mslength_{max_seq_length_1tr}_epochs_{n_epochs_1st}_{rep}_conf.pickle"

    # make train, test dataset (padding is necessary)
    # But after the padding, the padded part is masked at self-attention stage, so it causes no influence  
    # check the padded part is masked!!!                
    # hidden_dim is # of feature in input embedding  
    train_dataset, num_labels, hidden_dim = util.make_embed_dataset(train_data_path, train_conf_path, max_seq_length)
    #train_dataset, val_dataset, num_labels, hidden_dim = util.make_dataset(train_data_path, train_conf_path, max_seq_length, validation_prop)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle = True)
    #val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    val_dataset, _, _ = util.make_embed_dataset(val_data_path, val_conf_path, max_seq_length)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    if setting_number_of_labels == True:

        new_config_set = util.read_text(target_conf_path)
        num_labels = len(new_config_set)        
        
    print(f"number of labels: {num_labels}")
    print(f"train data, validation data: {len(train_dataset)}, {len(val_dataset)}")
    
    model_name = f"{model_save_dirc}2tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{max_seq_length}_{n_epochs_1tr}_{n_epochs}_{hidden_dim_1tr}_{hidden_dim}_{layers_1tr}_{layers}_{attention_heads_1tr}_{attention_heads}_{rep_1tr}_{rep}"
    if feedforward_dim == 128 or feedforward_dim == 256:
        model_name = f"{model_save_dirc}2tr_{train_app_to_name}_{val_app_to_name}_{target_conf_to_name}_{max_seq_length_1tr}_{max_seq_length}_{n_epochs_1tr}_{n_epochs}_{hidden_dim_1tr}_{hidden_dim}_{layers_1tr}_{layers}_{attention_heads_1tr}_{attention_heads}_{feedforward_dim}_{rep_1tr}_{rep}"
    # setting hyperparameters 
    enc_layers = layers
    enc_heads = attention_heads
    enc_pf_dim = feedforward_dim 
    enc_dropout = 0.1 
    clip = 1 
    best_valid_loss = float('inf')
    
    # setting model     
    enc = Encoder(hidden_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device, max_seq_length)
    model = TransformerEncoder(hidden_dim, num_labels, enc, device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)
    #model = nn.DataParallel(model, device_ids = [1, 2, 4])
    
    # define optimizer and loss function 
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # training  
    tr_loss_list, val_loss_list =[], [] 
    for epoch in range(n_epochs):
        start_time = time.time()
        
        train_loss, _, _ = train(epoch, model, train_dataloader, optimizer, criterion, clip, device)
        val_loss = validation(model, val_dataloader, criterion, device)
        print(f"Training {epoch} Loss: {train_loss}")
        print(f"Validation {epoch} Loss: {val_loss}")
        tr_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        # save model when validation loss is minimum 
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss 
            print(f"Save model :{val_loss}") 
            torch.save(model.state_dict(), model_name)
    
    loss_fig(tr_loss_list, val_loss_list, loss_save_dirc, train_app_to_name, val_app_to_name, target_conf_to_name, max_seq_length_1tr, max_seq_length, n_epochs_1tr, n_epochs, hidden_dim_1tr, hidden_dim, layers_1tr, layers, attention_heads_1tr, attention_heads, rep_1tr, rep)
    #loss_fig(rep, num_labels, tr_loss_list, val_loss_list, n_epochs, max_seq_length, max_seq_length_1tr, train_app_to_name, val_app_to_name)
    
    model.load_state_dict(torch.load(model_name))

    evaluate_pre(conf_index_dict, result_save_dirc, rep, train_data_type, test_abnormal_data_path, test_abnormal_conf_path, max_seq_length, max_seq_length_1tr, n_epochs_1tr, n_epochs, hidden_dim_1tr, hidden_dim, layers_1tr, layers, attention_heads_1tr, attention_heads, rep_1tr, batch_size, model, device, target_conf_to_name, train_app_to_name, val_app_to_name, test_app_to_name, feedforward_dim)
