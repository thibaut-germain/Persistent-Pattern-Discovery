import numpy as np
import json
#from sklearn.model_selection import ParameterGrid
import subprocess

import sys
sys.path.append("./src")

CONFIG_PATH = "./experiment/REFIT/configs/base_configs.json"
DATASET_PATH = "./dataset/REFIT/dataset.pkl"
LABEL_PATH = "./dataset/REFIT/labels.pkl"
RESULT_FOLDER = "./experiment/REFIT/results/"
EXP_ID = 0

algorithms = [
    "BasePersistentPattern",
    "MatrixProfile",
    "Valmod",
    "Baseline",
    "LatentMotif",
    "Grammarviz",
]

if __name__ == "__main__": 

    command = ""
    for algo in algorithms: 
        indv_command = f"python ./experiment/algo_configs_exp.py {algo} {CONFIG_PATH} {DATASET_PATH} {LABEL_PATH} {RESULT_FOLDER} {EXP_ID} & "
        command += indv_command

    subprocess.run(command, shell=True)

