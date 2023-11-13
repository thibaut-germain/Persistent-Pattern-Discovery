import subprocess

import sys
sys.path.append("./src")
import os 
import json 

N_JUMP = range(1,4)
EXP_ID = 0

algo= "BasePersistentPattern"
folders = os.listdir("./dataset/")
folders.remove(".DS_Store")
folders.remove("computation-time")

for f in folders: 
    config_path = "./experiment/"+ f + "/configs/base_configs.json"
    with open(config_path,"r") as fl: 
        configs = json.load(fl)
    config = json.loads(configs["BasePersistentPattern"])[0]
    lst = []
    for i in N_JUMP: 
        t_config = config.copy()
        del t_config['n_patterns']
        t_config["jump"] = int(i)
        lst.append(t_config)
    configs = {"BasePersistentPattern": json.dumps(lst)}

    save_path = "./experiment/n_patterns/configs/" + f"{f}_configs.json"
    with open(save_path, "w") as fl: 
        json.dump(configs,fl)
    

if __name__ == "__main__": 

    command = ""
    for f in folders: 
        data_path = "./dataset/" + f +"/dataset.pkl"
        label_path = "./dataset/" + f +"/labels.pkl"
        config_path = "./experiment/n_patterns/configs/" + f"{f}_configs.json"
        result_path = f"./experiment/n_patterns/results/{f}_"
        indv_command = f"python ./experiment/algo_configs_exp.py {algo} {config_path} {data_path} {label_path} {result_path} {EXP_ID} & "
        command += indv_command

    subprocess.run(command, shell=True)

