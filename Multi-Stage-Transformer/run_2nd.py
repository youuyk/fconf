import argparse
import train_2nd 
import util
import matplotlib.pyplot as plt 
import sys, os
import pandas as pd  
import random 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # running_type 
    parser.add_argument("--rep", default = 0, type = int)
    parser.add_argument("--param_data_path", default = "./data/")
    parser.add_argument("--embed_save_path", default = "./train_embed_data/")
    parser.add_argument("--train_app_path", default = "./param/spark_train_app_4.txt")
    parser.add_argument("--validation_app_path", default = "./param/spark_val_app_4.txt")
    parser.add_argument("--test_app_path", default = "./param/spark_test_app_4.txt")
    parser.add_argument("--target_conf_path", default = "./param/spark_abnorm_conf_9.txt")

    parser.add_argument("--result_save_dirc", default = "./result/")
    parser.add_argument("--loss_save_dirc", default = "./loss_graph/")
    parser.add_argument("--model_save_dirc", default = "./model/")
    parser.add_argument("--rep_1tr", default = 0)
    # argument for training 
    parser.add_argument("--batch_size", type = int, default=512)
    # argument for model setting
    parser.add_argument("--n_epochs", type = int, default=100)
    parser.add_argument("--n_epochs_1tr", type = int)
    # if the input is embedding(or sequence of embedding), we don't need hidden_dim, 
    # the hidden_dim is fixed as # of feature in input embedding 
    #parser.add_argument("--hidden_dim", type=int, default =128)
    parser.add_argument("--max_seq_length", type = int, default=128)
    parser.add_argument("--max_seq_length_1tr", type = int, default = 128)
    parser.add_argument("--hidden_dim_1tr", type = int, default = 128) 
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--attention_heads_1tr", type = int, default = 8)
    parser.add_argument("--layers_1tr", type = int, default=1)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--cuda", default = 0)
    
    parser.add_argument("--setting_number_of_labels", action = 'store_true')
    parser.add_argument("--no_setting_number_of_labels", action = 'store_false', dest = 'setting_number_of_labels')
    parser.set_defaults(setting_number_of_labels = True)
    parser.add_argument("--feedforward_dim", default = 512, type= int)

    # train_data_type should be same as 1tr 
    parser.add_argument("--train_data_type", default = "abnormal")
    args = parser.parse_args()
        
    train_2nd.train_main(args)   
