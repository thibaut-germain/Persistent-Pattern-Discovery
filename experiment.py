import numpy as np 
import pandas as pd

from metric import SingleSampleScore,SampleScore,SingleEventScore,EventScore

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
            np.ndarray: predictions
        """
        predictions = []
        for config in configs:
            algo = algo_class(**config)
            algo.fit(signal)
            predictions.append(algo.prediction_mask_)

        return np.array(predictions)

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
                predictions = self._get_algo_class_predictions(signal,algo_class,configs)
                scores = self._compute_scores(label,predictions)
                t_df = pd.DataFrame(scores, columns=['metric', 'threshold', 'config_idx', 'precision', 'recall', 'f1-score'])
                t_df['algorithm'] = algo_class.__name__
                t_df['signal_idx'] = s_idx
                self.df_= pd.concat((self.df_,t_df)).reset_index(drop = True)
                self.df_ = self.df_.astype({'metric':str, 'threshold':float, 'config_idx':int, 'precision':float, 'recall':float, 'f1-score':float})

                counter += len(configs)
                if verbose: 
                    print(f"Prog: {np.around(100*counter/total,2)}%, signal: {s_idx+1}/{n_signals}, algo: {algo_class.__name__}, ss_precision: {np.around(float(scores[1][3]),2)}, ss_recall: {np.around(float(scores[1][4]),2)}, ss_f1_score: {np.around(float(scores[1][5]),2)}")
        
            if backup_path != None: 
                self.df_.to_csv(backup_path)

        return self.df_