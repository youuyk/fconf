# Falconf
Falconf: Learning to Identify Misconfiguration via Log-based Deep Learning Model

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
