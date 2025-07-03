# Falconf


## 📑 Overview 
Falconf: Learning to Identify Misconfiguration via Log-based Deep Learning Model
<img src="https://github.com/user-attachments/assets/17b78d3e-535d-464c-bc94-d264e71a77e7">


## Organization
---
```
|--Dataset			#Directory of log dataset generated from 'LogCollection'.
|--LogCollection 		#Collecting logs of system (e.g., Spark, Redis, and etc). 
|--LogPreprocessing		#Preprocess log and generating log templates
|--Multi-Stage-Transformer	#Training multi-stage Transformer model 
```

## 🖥 Running Falconf 
---
```
git clone https://github.com/youuyk/Falconf.git
cd Falconf 
pip install -r requirements.txt
```

### Step1. Collecting Logs

In this step, we employ fault injection to induce errors in the target system (e.g., Spark) and collect configuration error logs by running the system in distributed manner. 

To run this step, 
```
cd LogCollection 
./start.sh 
```

For more information, please check the ```LogCollection``` directory.

The outcome:
```
|--(application name) 		#Directory for storing logs 
	|--(file name) 		#Directory for storing logs of each configuration error case 
```

### Step 2. Preprocessing Logs 

In this step, we preprocess logs and generate log templates.

Then, we convert each log entry into its corresponding template. 

To run this step, 
```
cd LogPreprocessing
python3 make_template_pipeline.py 
python3 log_to_template.py
```

### Step 3. Training Multi-Stage Transformer 

In this step, we train the multi-stage transformer. 

To run this step, 
```python
cd Multi-Stage-Transformer
python3 run_1st.py --train_app_path (path for training log data) --validation_app_path (path for validation log data) --test_app_path (path for test log data) --target_conf_path (path of configuration file) --batch_size 512 --epochs 30 --hidden_dim 512 --max_seq_length 128 --attnetion_heads 8 --layers 1 --system spark 
python3 run_2nd.py --train_app_path (path for training embedding data) --validation_app_path (path for validation embedding data) --test_app_path (path for test embedding data) --n_epochs 100 --max_seq_length 512 --attention_heads 8 --layers 1 
```

The outcome:
```
|--model			#Directory where trained model is stored 
    |--(name of model) 		#Directory of model 
```
