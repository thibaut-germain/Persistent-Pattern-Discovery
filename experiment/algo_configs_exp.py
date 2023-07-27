import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt 

import json

import sys
sys.path.append("./src")

from methods import BasePersistentPattern
from competitors import Baseline,LatentMotif,MatrixProfile,Valmod,Grammarviz
from experiment import Experiment


algorithms = {
    "BasePersistentPattern" : BasePersistentPattern,
    "Baseline" : Baseline,
    "LatentMotif" : LatentMotif,
    "MatrixProfile" : MatrixProfile,
    "Valmod" : Valmod,
    "Grammarviz" : Grammarviz,
}

algorithm_class = algorithms[sys.argv[1]]
#print("### LOOK HERE ### \n")
with open(sys.argv[2],"r") as f:
    configs = json.load(f)

algo_configs = json.loads(configs[sys.argv[1]])

DATASET_PATH = sys.argv[3]
LABEL_PATH = sys.argv[4]
RESULT_FOLDER = sys.argv[5]
EXP_ID = sys.argv[6]
BACKUP_PATH =  RESULT_FOLDER + f'{algorithm_class.__name__}_{EXP_ID}.csv'
EXPERIMENT_PATH = RESULT_FOLDER + f'{algorithm_class.__name__}_{EXP_ID}.pickle'
LOG_PATH = RESULT_FOLDER + f'logs_{algorithm_class.__name__}_{EXP_ID}.txt'

#load dataset and labels
with open(DATASET_PATH,"rb") as f: 
    dataset = pickle.load(f)

with open(LABEL_PATH,"rb") as f: 
    labels = pickle.load(f)


algorithm = [algorithm_class]
configurations = [algo_configs]

if __name__ == "__main__": 

    ese = Experiment(algorithm,configurations,njobs=1)
    ese.run_experiment(dataset,labels,backup_path=BACKUP_PATH,batch_size=10,logs_path=LOG_PATH)
    with open(EXPERIMENT_PATH, 'wb') as filehandler: 
        pickle.dump(ese,filehandler)

