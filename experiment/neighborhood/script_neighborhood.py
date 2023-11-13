import subprocess

import sys
sys.path.append("./src")
import os 
import json 

N_NEIGHBORS = [1,2,3,4,5,10,15,20,30,40,50]
EXP_ID = 0

algo= "BasePersistentPattern"
folders = os.listdir("./dataset/")
folders.remove(".DS_Store")
folders.remove("computation-time")

for f in folders: 
    config_path = "./experiment/"+ f + "/configs/base_configs.json"
    with open(config_path,"r") as fl: 
        configs = json.load(fl)
    lst = []
    for config in json.loads(configs["BasePersistentPattern"]):
        for i in N_NEIGHBORS: 
            t_config = config.copy()
            t_config['n_neighbors'] = int(i)
            lst.append(t_config)
    configs = {"BasePersistentPattern": json.dumps(lst)}
    save_path = "./experiment/neighborhood/configs/" + f"{f}_configs.json"
    with open(save_path, "w") as fl: 
        json.dump(configs,fl)
    

if __name__ == "__main__": 

    command = ""
    for f in folders: 
        data_path = "./dataset/" + f +"/dataset.pkl"
        label_path = "./dataset/" + f +"/labels.pkl"
        config_path = "./experiment/neighborhood/configs/" + f"{f}_configs.json"
        result_path = f"./experiment/neighborhood/results/{f}_"
        indv_command = f"python ./experiment/algo_configs_exp.py {algo} {config_path} {data_path} {label_path} {result_path} {EXP_ID} & "
        command += indv_command

    subprocess.run(command, shell=True)

