import numpy as np 
import pandas as pd
import time

from metric import SingleSampleScore,SampleScore,SingleEventScore,EventScore
from synthetic_signal import SignalGenerator

def generate_dataset(config_lst:list,n_signals:int)->tuple:
    """Generate a dataset

    Args:
        config_lst (list): list of dictionnary, each dictionnary is a configuration
        n_signals (int): the number of signals is evenly distributed among all configurations

    Returns:
        tuple: list of signals, associated labels, associated configurations
    """
    df = pd.DataFrame.from_dict(config_lst)
    n_configs = df.shape[0]
    n_sig_config = np.ceil(n_signals/n_configs).astype(int)
    idxs = np.repeat(np.arange(n_configs),n_sig_config)
    df = df.loc[idxs]
    df = df.reset_index(drop=True)

    configs = df.to_dict(orient="records")
    dataset = []
    labels = []
    for i,config in enumerate(configs): 
        np.random.seed(i)
        signal, label = SignalGenerator(**config).generate()
        dataset.append(signal)
        labels.append(label)
    return dataset,labels,df

class ExperimentSampleEventOld(object): 

    def __init__(self,algorithms:list, configurations:list, nbins = 101) -> None:
        """Initialization

        Args:
            algorithms (list): list of algorithm classes
            configurations (list): list of list of configurations as dictionnaries for each algorithms classes
            nbins (int, optional): number of bins for thresholdings. Defaults to 101.
        """
        self.algorithms = algorithms
        self.configurations = configurations
        self.nbins = nbins

    def _get_algo_class_predictions(self,signal:np.ndarray,algo_class:object, configs:dict)->np.ndarray: 
        """Get predictions for one signal one algorithm and all its configurations

        Args:
            signal (np.ndarray): signal, shape: (L,)
            algo_class (object): algortihm class
            configs (dict): list of algorithm configurations

        Returns:
            np.ndarray: predictions, exectution times
        """
        predictions = []
        execution_time = []
        for config in configs:
            algo = algo_class(**config)
            start = time.time()
            algo.fit(signal)
            end = time.time()
            predictions.append(algo.prediction_mask_)
            execution_time.append(end-start)

        return np.array(predictions),np.array(execution_time)

    def _compute_single_sample_score(self,single_label:np.ndarray,single_preds:np.ndarray)->list:
        """Compute single sample score

        Args:
            single_label (np.ndarray): label, shape: (L,)
            single_preds (np.ndarray): prediction, shape: (L,)

        Returns:
            list: score
        """
        sss = SingleSampleScore() 
        sss_scores = []
        for single_pred in single_preds: 
            sss_scores.append(sss.score(single_label,single_pred))
        sss_scores = np.array(sss_scores)
        config_idx = np.argmax(sss_scores[:,-1])
        score = ['sss',np.nan,config_idx] + sss_scores[config_idx].tolist()
        return score

    def _compute_sample_score(self,label:np.ndarray,predictions:np.ndarray)->list: 
        """Compute sampling score

        Args:
            label (np.ndarray): label, shape (N,L)
            predictions (np.ndarray): prediction, shape (N,L)

        Returns:
            list: score
        """
        ss = SampleScore()
        ss_scores = []
        for prediction in predictions: 
            ss_scores.append(ss.score(label,prediction))
        ss_scores = np.array(ss_scores)
        config_idx = np.argmax(ss_scores[:,-1])
        score = ['ss',np.nan,config_idx] + ss_scores[config_idx].tolist()
        return score

    def _compute_single_event_score(self,single_label:np.ndarray,single_preds:np.ndarray)->np.ndarray: 
        """Compute single event score

        Args:
            single_label (np.ndarray): label, shape: (L,)
            single_preds (np.ndarray): prediction, shape: (L,)

        Returns:
            np.ndarray: score
        """
        ses = SingleEventScore(self.nbins)
        thresholds = np.linspace(0,1,self.nbins)

        #compute scores
        scores = []
        for single_pred in single_preds: 
            scores.append(np.array(ses.all_score(single_label,single_pred)).T)
        scores = np.array(scores)

        #compute scores per threshold
        config_idxs = np.argmax(scores[:,:,2],axis=0)
        t_scores = scores[config_idxs,np.arange(self.nbins),:]
        t_scores = np.hstack((np.full((self.nbins,1),'ses'),thresholds.reshape(-1,1),config_idxs.reshape(-1,1),t_scores))

        #compute AUC
        mean_score = np.mean(scores,axis=1)
        config_idx = np.argmax(mean_score[:,-1])
        auc_score = ['ses_auc', np.nan, config_idx] + mean_score[config_idx].tolist()

        scores = np.vstack((t_scores,auc_score))
        return scores

    
    def _compute_event_score(self,label:np.ndarray,predictions:np.ndarray)->np.ndarray:
        """Compute event score

        Args:
            label (np.ndarray): label, shape (N,L)
            predictions (np.ndarray): prediction, shape (N,L)

        Returns:
            np.ndarray: score
        """ 
        es = EventScore(self.nbins)
        thresholds = np.linspace(0,1,self.nbins)

        #compute scores
        scores = []
        for prediction in predictions: 
            scores.append(np.array(es.all_score(label,prediction)).T)
        scores = np.array(scores)

        #compute scores per threshold
        config_idxs = np.argmax(scores[:,:,2],axis=0)
        t_scores = scores[config_idxs,np.arange(self.nbins),:]
        t_scores = np.hstack((np.full((self.nbins,1),'es'),thresholds.reshape(-1,1),config_idxs.reshape(-1,1),t_scores))

        #compute AUC
        mean_score = np.mean(scores,axis=1)
        config_idx = np.argmax(mean_score[:,-1])
        auc_score = ['es_auc', np.nan, config_idx] + mean_score[config_idx].tolist()

        scores = np.vstack((t_scores,auc_score))
        return scores

    def _compute_scores(self,label:np.ndarray,predictions:np.ndarray)->np.ndarray: 
        """Compute score

        Args:
            label (np.ndarray): label, shape (N,L)
            predictions (np.ndarray): prediction, shape (N,L)

        Returns:
            np.ndarray: score
        """
        scores = []

        single_preds = np.clip(np.sum(predictions,axis=1),0,1) 
        single_label = np.clip(np.sum(label,axis=0),0,1)

        #single sample score
        sss_score = self._compute_single_sample_score(single_label,single_preds) 
        scores.append(sss_score)

        #sample score
        ss_score = self._compute_sample_score(label,predictions)
        scores.append(ss_score)

        #single event score 
        ses_score = self._compute_single_event_score(single_label,single_preds)
        scores.append(ses_score)

        #event score
        es_score = self._compute_event_score(label,predictions)
        scores.append(es_score)

        scores = np.vstack(scores)

        return scores            

    def run_experiment(self,dataset:np.ndarray,labels:np.ndarray,backup_path = None,verbose = True)->np.ndarray:
        """_summary_

        Args:
            dataset (np.ndarray): array of signals, signal shape (L,), variable length allowed
            labels (np.ndarray): array of labels, label shape (L,), variable length allowed
            backup_path (str, optional): Path to store df in case of big experiment. If None no saving. Defaults to None.
            verbose (bool, optional): verbose. Defaults to True.

        Returns:
            pd.DataFrame: scores_df
        """
        n_signals = len(dataset)
        n_configs = np.sum([len(conf) for conf in self.configurations])
        total = n_signals*n_configs

        self.df_ = pd.DataFrame()

        counter = 0
        for s_idx,(signal,label) in enumerate(zip(dataset,labels)): 
            for algo_class,configs in zip(self.algorithms,self.configurations): 
                predictions,execution_times = self._get_algo_class_predictions(signal,algo_class,configs)
                scores = self._compute_scores(label,predictions)
                t_df = pd.DataFrame(scores, columns=['metric', 'threshold', 'config_idx', 'precision', 'recall', 'f1-score'])
                t_df['algorithm'] = algo_class.__name__
                t_df['signal_idx'] = s_idx
                t_df = t_df.astype({'config_idx' : int})
                other = pd.DataFrame({'ex_time' : execution_times})
                t_df = t_df.join(other=other, on='config_idx')
                self.df_= pd.concat((self.df_,t_df)).reset_index(drop = True)
                self.df_ = self.df_.astype({'metric':str, 'threshold':float, 'config_idx':int, 'precision':float, 'recall':float, 'f1-score':float})

                counter += len(configs)
                if verbose: 
                    print(f"Prog: {np.around(100*counter/total,2)}%, signal: {s_idx+1}/{n_signals}, algo: {algo_class.__name__}, ss_precision: {np.around(float(scores[1][3]),2)}, ss_recall: {np.around(float(scores[1][4]),2)}, ss_f1_score: {np.around(float(scores[1][5]),2)}")
        
            if backup_path != None: 
                self.df_.to_csv(backup_path)

        return self.df_
    

class ExperimentSampleEvent(object): 

    def __init__(self,algorithms:list, configurations:list, nbins = 101) -> None:
        """Initialization

        Args:
            algorithms (list): list of algorithm classes
            configurations (list): list of list of configurations as dictionnaries for each algorithms classes
            nbins (int, optional): number of bins for thresholdings. Defaults to 101.
        """
        self.algorithms = algorithms
        self.configurations = configurations
        self.nbins = nbins

    def _get_algo_class_predictions(self,signal:np.ndarray,algo_class:object, configs:dict)->np.ndarray: 
        """Get predictions for one signal one algorithm and all its configurations

        Args:
            signal (np.ndarray): signal, shape: (L,)
            algo_class (object): algortihm class
            configs (dict): list of algorithm configurations

        Returns:
            np.ndarray: predictions, exectution times
        """
        predictions = []
        execution_time = []
        for config in configs:
            algo = algo_class(**config)
            start = time.time()
            algo.fit(signal)
            end = time.time()
            predictions.append(algo.prediction_mask_)
            execution_time.append(end-start)

        return np.array(predictions),np.array(execution_time)

    def _compute_single_sample_score(self,single_label:np.ndarray,single_preds:np.ndarray)->list:
        """Compute single sample score

        Args:
            single_label (np.ndarray): label, shape: (L,)
            single_preds (np.ndarray): prediction, shape: (L,)

        Returns:
            list: score
        """
        sss = SingleSampleScore() 
        sss_scores = []
        for single_pred in single_preds: 
            sss_scores.append(sss.score(single_label,single_pred))
        sss_scores = np.array(sss_scores)
        n_configs = sss_scores.shape[0]
        score = np.hstack((np.full((n_configs,1),'sss'),np.full((n_configs,1),np.nan),np.arange(n_configs).reshape(-1,1),np.full((n_configs,1),np.nan),sss_scores))
        return score

    def _compute_sample_score(self,label:np.ndarray,predictions:np.ndarray)->list: 
        """Compute sampling score

        Args:
            label (np.ndarray): label, shape (N,L)
            predictions (np.ndarray): prediction, shape (N,L)

        Returns:
            list: score
        """
        ss = SampleScore()
        ss_scores = []
        for prediction in predictions: 
            ss_scores.append(ss.score(label,prediction))
        ss_scores = np.array(ss_scores)
        n_configs = ss_scores.shape[0]
        score = np.hstack((np.full((n_configs,1),'ss'),np.full((n_configs,1),np.nan),np.arange(n_configs).reshape(-1,1),np.full((n_configs,1),np.nan),ss_scores))
        return score

    def _compute_single_event_score(self,single_label:np.ndarray,single_preds:np.ndarray)->np.ndarray: 
        """Compute single event score

        Args:
            single_label (np.ndarray): label, shape: (L,)
            single_preds (np.ndarray): prediction, shape: (L,)

        Returns:
            np.ndarray: score
        """
        ses = SingleEventScore(self.nbins)
        thresholds = np.linspace(0,1,self.nbins)

        #compute scores
        scores = []
        for idx,single_pred in enumerate(single_preds): 
            t_scores = np.array(ses.all_score(single_label,single_pred)).T
            mean_score = np.mean(t_scores,axis=0)
            auc_score = np.array(['ses_auc', np.nan, idx, np.nan] + mean_score.tolist()).reshape(1,-1)
            t_scores = np.hstack((np.full((self.nbins,1),'ses'),thresholds.reshape(-1,1),np.full((self.nbins,1),idx),np.full((self.nbins,1),np.nan),t_scores))
            t_scores = np.vstack((auc_score,t_scores))
            scores.append(t_scores)
        scores = np.array(scores)
        scores = np.vstack((scores))
        return scores

    
    def _compute_event_score(self,label:np.ndarray,predictions:np.ndarray)->np.ndarray:
        """Compute event score

        Args:
            label (np.ndarray): label, shape (N,L)
            predictions (np.ndarray): prediction, shape (N,L)

        Returns:
            np.ndarray: score
        """ 
        es = EventScore(self.nbins)
        thresholds = np.linspace(0,1,self.nbins)

        #compute scores
        scores = []
        for idx,prediction in enumerate(predictions): 
            t_scores = np.array(es.all_score(label,prediction)).T
            mean_score = np.mean(t_scores,axis=0)
            auc_score = np.array(['es_auc', np.nan, idx,np.nan] + mean_score.tolist()).reshape(1,-1)
            t_scores = np.hstack((np.full((self.nbins,1),'es'),thresholds.reshape(-1,1),np.full((self.nbins,1),idx),np.full((self.nbins,1),np.nan),t_scores))
            t_scores = np.vstack((auc_score,t_scores))
            scores.append(t_scores)
        scores = np.array(scores)
        scores = np.vstack((scores))
        return scores
    

    def _compute_adjusted_mutual_info_score(self,label:np.ndarray,predictions:np.ndarray)->np.ndarray:
        """Compute adjusted mutual info score

        Args:
            label (np.ndarray): label, shape (N,L)
            predictions (np.ndarray): prediction, shape (N,L)

        Returns:
            np.ndarray: score
        """ 
        amis = AdjustedMutualInfoScore()
        amis_scores = []
        for prediction in predictions: 
            amis_scores.append([amis.score(label,prediction),np.nan,np.nan,np.nan])
        amis_scores = np.array(amis_scores)
        n_configs = amis_scores.shape[0]
        score = np.hstack((np.full((n_configs,1),'amis'),np.full((n_configs,1),np.nan),np.arange(n_configs).reshape(-1,1),amis_scores))
        return score



    def _compute_scores(self,label:np.ndarray,predictions:np.ndarray)->np.ndarray: 
        """Compute score

        Args:
            label (np.ndarray): label, shape (N,L)
            predictions (np.ndarray): prediction, shape (N,L)

        Returns:
            np.ndarray: score
        """
        scores = []

        single_preds = np.clip(np.sum(predictions,axis=1),0,1) 
        single_label = np.clip(np.sum(label,axis=0),0,1)

        #single sample score
        sss_score = self._compute_single_sample_score(single_label,single_preds) 
        scores.append(sss_score)

        #sample score
        ss_score = self._compute_sample_score(label,predictions)
        scores.append(ss_score)

        #single event score 
        ses_score = self._compute_single_event_score(single_label,single_preds)
        scores.append(ses_score)

        #event score
        es_score = self._compute_event_score(label,predictions)
        scores.append(es_score)

        #amis score 
        amis_score = self._compute_adjusted_mutual_info_score(label,predictions)
        scores.append(amis_score)

        scores = np.vstack(scores)

        return scores            

    def run_experiment(self,dataset:np.ndarray,labels:np.ndarray,signal_configs=None,backup_path = None,verbose = True)->np.ndarray:
        """_summary_

        Args:
            dataset (np.ndarray): array of signals, signal shape (L,), variable length allowed
            labels (np.ndarray): array of labels, label shape (L,), variable length allowed
            signal_configs (pd.DataFrame, optional): Dataframe containing the configuration of the synthetic generator for each signals.
            backup_path (str, optional): Path to store df in case of big experiment. If None no saving. Defaults to None.
            verbose (bool, optional): verbose. Defaults to True.

        Returns:
            pd.DataFrame: scores_df
        """
        if signal_configs: 
            self.signal_configs_ = signal_configs
        
        n_signals = len(dataset)
        n_configs = np.sum([len(conf) for conf in self.configurations])
        total = n_signals*n_configs

        self.df_ = pd.DataFrame()

        counter = 0
        for s_idx,(signal,label) in enumerate(zip(dataset,labels)): 
            for algo_class,configs in zip(self.algorithms,self.configurations): 
                predictions,execution_times = self._get_algo_class_predictions(signal,algo_class,configs)
                scores = self._compute_scores(label,predictions)
                t_df = pd.DataFrame(scores, columns=['metric', 'threshold', 'config_idx', 'score', 'precision', 'recall', 'f1-score'])
                t_df['algorithm'] = algo_class.__name__
                t_df['signal_idx'] = s_idx
                t_df = t_df.astype({'config_idx' : int})
                other = pd.DataFrame({'ex_time' : execution_times})
                t_df = t_df.join(other=other, on='config_idx')
                self.df_= pd.concat((self.df_,t_df)).reset_index(drop = True)
                self.df_ = self.df_.astype({'metric':str, 'threshold':float, 'config_idx':int, "score":float ,'precision':float, 'recall':float, 'f1-score':float})

                ss_score = scores[scores[:,0] == "ss"]
                best_config = np.argmax(ss_score[:,-1])
                ss_scores = ss_score[best_config]
                amis_score = scores[scores[:,0] == "amis"]
                best_config = np.argmax(ss_score[:,-1])
                amis_scores = amis_score[best_config]
                counter += len(configs)
                if verbose: 
                    print(f"Prog: {np.around(100*counter/total,2)}%, signal: {s_idx+1}/{n_signals}, algo: {algo_class.__name__}, amis : {np.around(float(amis_scores[3]),2)}, ss_precision: {np.around(float(ss_scores[4]),2)}, ss_recall: {np.around(float(ss_scores[5]),2)}, ss_f1_score: {np.around(float(ss_scores[6]),2)}")
        
            if backup_path != None: 
                self.df_.to_csv(backup_path)

        return self.df_