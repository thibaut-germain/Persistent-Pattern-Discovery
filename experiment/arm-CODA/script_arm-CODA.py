import subprocess

import sys
sys.path.append("./src")

CONFIG_PATH = "./experiment/arm-CODA/configs/test_bpp_configs.json"
DATASET_PATH = "./dataset/arm-CODA/dataset.pkl"
LABEL_PATH = "./dataset/arm-CODA/labels.pkl"
RESULT_FOLDER = "./experiment/arm-CODA/results/"
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

