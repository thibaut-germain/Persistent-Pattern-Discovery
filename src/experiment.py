import numpy as np 
import pandas as pd
import time
from metric import SampleScore,EventScore, AdjustedMutualInfoScore
from joblib import Parallel, delayed


class Experiment: 

    def __init__(self,algorithms:list, configurations:list, thresholds = np.linspace(0,1,101),njobs=1,verbose = True) -> None:
   


        """Initialization

        Args:
            algorithms (list): list of algorithm classes
            configurations (list): list of list of configurations as dictionnaries for each algorithms classes
            thresholds (np.ndarray, optional): numpy array of thresholds to consider for the event based metric. Defaults to numpy.linspace(0,1,101).
        """
        self.algorithms = algorithms
        self.configurations = configurations
        self.thresholds = thresholds
        self.njobs = njobs
        self.verbose = verbose

    def compute_scores(self,label,prediction): 

        single_pred = np.clip(np.sum(prediction,axis=0),0,1).reshape(1,-1) 
        single_label = np.clip(np.sum(label,axis=0),0,1).reshape(1,-1)

        scores = []

        #single sample score
        p,r,f = SampleScore().score(single_label,single_pred)
        scores.append(["sss-precision",p])
        scores.append(["sss-recall",r])
        scores.append(["sss-fscore",f])

        #sample score 
        p,r,f = SampleScore().score(label,prediction)
        scores.append(["ss-precision",p])
        scores.append(["ss-recall",r])
        scores.append(["ss-fscore",f])

        # weigthed sample score 
        p,r,f = SampleScore(averaging="weighted").score(label,prediction)
        scores.append(["w-ss-precision",p])
        scores.append(["w-ss-recall",r])
        scores.append(["w-ss-fscore",f])

        #single event score
        lp,lr,lf = EventScore().score(single_label,single_pred,self.thresholds)
        for t,p,r,f in zip(self.thresholds,lp,lr,lf): 
            scores.append([f"ses-precision_{np.round(t,2)}",p])
            scores.append([f"ses-recall_{np.round(t,2)}",r])
            scores.append([f"ses-fscore_{np.round(t,2)}",f])
        scores.append(["ses-auc-precision",np.mean(lp)])
        scores.append(["ses-auc-recall",np.mean(lr)])
        scores.append(["ses-auc-fscore",np.mean(lf)])

        #event score
        lp,lr,lf = EventScore().score(label,prediction,self.thresholds)
        for t,p,r,f in zip(self.thresholds,lp,lr,lf): 
            scores.append([f"es-precision_{np.round(t,2)}",p])
            scores.append([f"es-recall_{np.round(t,2)}",r])
            scores.append([f"es-fscore_{np.round(t,2)}",f])
        scores.append(["es-auc-precision",np.mean(lp)])
        scores.append(["es-auc-recall",np.mean(lr)])
        scores.append(["es-auc-fscore",np.mean(lf)])

        # weighted event score
        lp,lr,lf = EventScore(averaging="weighted").score(label,prediction,self.thresholds)
        for t,p,r,f in zip(self.thresholds,lp,lr,lf): 
            scores.append([f"w-es-precision_{np.round(t,2)}",p])
            scores.append([f"w-es-recall_{np.round(t,2)}",r])
            scores.append([f"w-es-fscore_{np.round(t,2)}",f])
        scores.append(["w-es-auc-precision",np.mean(lp)])
        scores.append(["w-es-auc-recall",np.mean(lr)])
        scores.append(["w-es-auc-fscore",np.mean(lf)])

        #ajusted mutual information
        scores.append(["amis",AdjustedMutualInfoScore().score(label,prediction)])

        return scores
    
    def signal_algo_class_experiement(self,signal_idx,signal,label,algo_class,config,config_idx): 
        "Return a DF"
        #keep only labels row that are activated by the signal 
        label = label[label.sum(axis=1)>0]

        #update the number of patterns to predict if required
        t_config = config.copy()
        if ("n_patterns" in t_config.keys()):
            if (isinstance(t_config["n_patterns"],int)):
                t_config["n_patterns"] = label.shape[0]
            else:
                t_config["n_patterns"] = None

        
        try:
            #get predictions
            algo = algo_class(**t_config)
            start = time.time()
            algo.fit(signal)
            end = time.time()


            #compute scores
            scores = self.compute_scores(label,algo.prediction_mask_)

            tdf = pd.DataFrame(scores,columns=["metric","score"])
            tdf["algorithm"] = algo_class.__name__
            tdf["config_idx"] = config_idx
            tdf["execution_time"] = end - start
            tdf["signal_idx"] = signal_idx
            tdf["n_patterns"] = label.shape[0]
            tdf["predicted_n_patterns"] = algo.prediction_mask_.shape[0]

            if self.verbose: 
                s1 = np.round(tdf[tdf["metric"] == "es-auc-fscore"].score.values[0],2)
                s2 = np.round(tdf[tdf["metric"] == "amis"].score.values[0],2)
                print(f"signal_id: {signal_idx}, algo: {algo_class.__name__}, config_id: {config_idx}, f-auc: {s1}, ami: {s2}")
            
            return tdf 

        except: 
            s= f"signal_id: {signal_idx}, algo: {algo_class.__name__}, config_id: {config_idx} failed to fit."
            if self.verbose: 
                print(s)
            if self.logs_path_ is not None:
                with open(self.logs_path_,"a") as f: 
                    f.write(s +"\n")
            

         

    def run_experiment(self,dataset:np.ndarray,labels:np.ndarray,backup_path = None,batch_size=10,logs_path = None,verbose = True)->np.ndarray:
        """_summary_

        Args:
            dataset (np.ndarray): array of signals, signal shape (L,), variable length allowed
            labels (np.ndarray): array of labels, label shape (L,), variable length allowed
            signal_configs (pd.DataFrame, optional): Dataframe containing the configuration of the synthetic generator for each signals.
            backup_path (str, optional): Path to store df in case of big experiment. If None no saving. Defaults to None.
            batch_size (int, optional)
            verbose (bool, optional): verbose. Defaults to True.

        Returns:
            pd.DataFrame: scores_df
        """
        self.logs_path_ = logs_path
        
        n_signals = len(dataset)
        n_configs = np.sum([len(conf) for conf in self.configurations])
        total = n_signals*n_configs

        if backup_path != None: 
            n_batches  = n_signals//batch_size
            if n_batches >0:
                batches =[zip(dataset[i*batch_size:(i+1)*batch_size],labels[i*batch_size:(i+1)*batch_size]) for i in range(n_batches)]
            else: 
                batches = []
            if n_signals % batch_size !=0: 
                batches.append(zip(dataset[n_batches*batch_size:],labels[n_batches*batch_size:]))
        else:
            batches = [zip(dataset,labels)]

        self.df_ = pd.DataFrame()

        counts = 0
        for batch in batches: 
            results = Parallel(n_jobs=self.njobs)(
                delayed(self.signal_algo_class_experiement)(counts+id_s,signal,label,algo,config,id_c) 
                for id_s,(signal,label) in enumerate(batch) 
                for id_a,algo in enumerate(self.algorithms)
                for id_c,config in enumerate(self.configurations[id_a])
                )
            counts = min(counts+batch_size,n_signals)
            self.df_= pd.concat((self.df_,*results)).reset_index(drop = True)
            self.df_ = self.df_.astype({'metric':str, "score":float, "algorithm":str,'config_idx':int,"signal_idx":int, "n_patterns":int, "predicted_n_patterns":int})

            if backup_path != None: 
                self.df_.to_csv(backup_path)

            if verbose:
                print(f"Achieved [{counts*n_configs}/{total}]")

        return self.df_       