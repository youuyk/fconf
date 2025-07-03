import argparse
from train_1st import train_main
import util
import matplotlib.pyplot as plt 
import sys, os
import pandas as pd  
import random 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", default = 0, type = int)
    parser.add_argument("--raw_path")
    #parser.add_argument("--configuration_file_path")
    # spark application for training 
    parser.add_argument("--train_app_path", default = "./param/spark_train_app_6.txt")
    # spark application for testing 
    parser.add_argument("--validation_app_path", default = "./param/spark_val_app_6.txt")
    parser.add_argument("--test_app_path", default = "./param/spark_test_app_6.txt")
    parser.add_argument("--target_conf_path", default = "./param/spark_abnorm_conf_6.txt")
    
    parser.add_argument("--train_file_name", default ="./data/")
    parser.add_argument("--train_file_path", required=False)
    parser.add_argument("--test_file_path", required=False,)
    parser.add_argument("--validation_file_path", required = False)
    
    # log_template.pickle: log_template created with prepopulated_template (lognroll)
    # log_template_2.pickle: log_template created with saved_log_template_3 (lognroll)
    parser.add_argument("--log_file_name", default = "log_template_2.pickle")
    
    parser.add_argument("--model_save_dirc", default = "./model")
    parser.add_argument("--result_save_dirc", default= "./result")
    parser.add_argument("--loss_save_dirc", default = "./loss_graph")
    
    # argument for transformer training 
    parser.add_argument("--batch_size", type = int, default=512)
    parser.add_argument("--epochs", type = int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    # depending on the input sequence length 
    parser.add_argument("--max_seq_length", type = int, default=512)
    parser.add_argument("--if_sliding", action = 'store_true')
    parser.add_argument("--no_if_sliding", action = 'store_false')
    parser.set_defaults(if_sliding = False)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--validation_prop", type=float, default=0.2)
    parser.add_argument("--return_cls", action='store_true')
    parser.add_argument("--no_return_cls", dest = 'return_cls', action='store_false')
    parser.set_defaults(return_cls=False)
    parser.add_argument("--val_prop", type = float, default = 0.2)
    parser.add_argument("--system", default = "spark")
    
    # argument for evaluation 
    parser.add_argument("--check_top", action='store_true')
    parser.add_argument("--no_check_top", action='store_false', dest='check_top')
    parser.set_defaults(check_top =True)
    parser.add_argument("--n_top", type=int, default=10)
    parser.add_argument("--use_window", action='store_true')
    parser.add_argument("--no_use_window", action='store_false', dest='use_window')
    parser.set_defaults(use_window=False)
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--limit_label", action='store_true')
    parser.add_argument("--no_limit_label", action='store_false', dest='limit_label')
    parser.set_defaults(limit_label=False)
    parser.add_argument("--split_random", action='store_true')
    parser.add_argument("--no_split_random", action='store_false', dest='split_random')
    parser.set_defaults(split_random=False)
    # if limit_label is True, then should define number of limit label as well 
    parser.add_argument("--number_of_limit_label", type=int, default=-1)
    parser.add_argument("--append_cls_token", action='store_true')
    parser.add_argument("--no_append_cls_token", action='store_false', dest='append_cls_token')
    parser.set_defaults(append_cls_token=False)
    parser.add_argument("--iter_parameters", default="None")
    parser.add_argument("--cuda", default = 0)
    parser.add_argument("--log_length_limit", default=-1, type=int)
    parser.add_argument("--iter", type = int, default = 0)
    parser.add_argument("--data_type")
    # limit the number of configuration to train 
    parser.add_argument("--conf_filter", action='store_true')
    parser.add_argument("--no_conf_filter", action='store_false', dest = 'conf_filter')
    parser.set_defaults(conf_filter=False)
    # test_type: normal, abnormal 
    parser.add_argument("--test_type", default="abnormal")

    # path for saving embedding from 1st model 
    parser.add_argument("--embed_save_path", default="./train_embed_data/")
    parser.add_argument("--label_type", default = "label_without_value")
    
    parser.add_argument("--setting_number_of_labels", action = 'store_true')
    parser.add_argument("--no_setting_number_of_labels", action = 'store_false', dest = 'setting_number_of_labels')
    parser.set_defaults(setting_number_of_labels = False) 
    parser.add_argument("--app_config_path")
    
    # type of train_data_type == "abnormal", "normal", "both" (both when collecting both normal and abnormal)
    parser.add_argument("--train_data_type", default = "abnormal")
    parser.add_argument("--limit_test_normal", default = 1, type = int)
    parser.add_argument("--feedforward_dim", default = 512, type= int)
    args = parser.parse_args()
            
    # spark apps for training 
    target_apps = util.get_text_data(args.train_app_path)
    #target_conf = util.get_text_data(args.target_conf_path)
    test_apps = util.get_text_data(args.test_app_path)
    val_apps = util.get_text_data(args.validation_app_path)
    
    #train_main(args, target_conf, target_apps, test_apps, val_apps)
    train_main(args, target_apps, test_apps, val_apps)

