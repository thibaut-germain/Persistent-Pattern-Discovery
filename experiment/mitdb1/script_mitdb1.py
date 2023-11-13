import subprocess

import sys
sys.path.append("./src")

CONFIG_PATH = "./experiment/mitdb1/configs/base_configs.json"
DATASET_PATH = "./dataset/mitdb1/dataset.pkl"
LABEL_PATH = "./dataset/mitdb1/labels.pkl"
RESULT_FOLDER = "./experiment/mitdb1/results/"
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


