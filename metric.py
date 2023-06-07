import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from utils import transfrom_label

####################################################################################################################
####################################################################################################################
# Sample Based
####################################################################################################################
####################################################################################################################

def b_s_precision_(R:np.ndarray,P:np.ndarray)->float: 
    """compute precision

    Args:
        R (np.ndarray): Real Label array, shape: (L,)
        P (np.ndarray): Predicted label array, shape: (L,) 

    Returns:
        float: precision score
    """
    if sum(P) != 0:
        return np.sum(R*P)/np.sum(P)
    else: 
        return 0.0

def b_s_recall_(R:np.ndarray,P:np.ndarray)->float: 
    """compute recall

    Args:
        R (np.ndarray): Real Label array, shape: (L,)
        P (np.ndarray): Predicted label array, shape: (L,) 

    Returns:
        float: recall score
    """
    if sum(R) != 0: 
        return np.sum(R*P)/np.sum(R)
    else: 
        return 0.0

def b_f1_score_(R:np.ndarray,P:np.ndarray)->float: 
    """compute f1score

    Args:
        R (np.ndarray): Real Label array, shape: (L,)
        P (np.ndarray): Predicted label array, shape: (L,) 

    Returns:
        float: f1 score
    """
    a = b_s_precision_(R,P)
    b = b_s_recall_(R,P)
    if a+b == 0:
        return 0
    else:
        return 2*a*b/(a+b)

class SingleSampleScore: 

    def __init__(self) -> None:
        pass

    def score(self,R:np.ndarray,P:np.ndarray)->tuple: 
        """Compute the scoring

        Args:
            R (np.ndarray): Real Label array, shape: (L,)
            P (np.ndarray): Predicted label array, shape: (L,) 

        Returns: 
        tuple: precision, recall, F1-Score
        """
        self.precision_ = b_s_precision_(R,P)
        self.recall_ = b_s_recall_(R,P)
        self.f1_score_ = b_f1_score_(R,P)
        return self.precision_,self.recall_,self.f1_score_


class SampleScore:

    def __init__(self) -> None:
        pass

    def score(self,R:np.ndarray,P:np.ndarray)->None: 
        """Compute the scoring

        Args:
            R (np.ndarray): Real Label array, shape: (N,L)
            P (np.ndarray): Predicted label array, shape: (M,L) 
        """
        N = R.shape[0]
        M = P.shape[0]

        precisions = np.zeros((N,M))
        recalls = np.zeros((N,M))

        #Compute precision and recall
        for i,r in enumerate(R):
            for j, p in enumerate(P): 
                precisions[i,j] = b_s_precision_(r,p)
                recalls[i,j] = b_s_recall_(r,p)
   
        #compute f1scores
        f1_scores = np.divide(2*precisions*recalls,precisions+recalls,out=np.zeros((N,M)), where= (precisions+recalls)!=0)

        if N<=M: 
            cbs = list(it.permutations(range(M),N))
            t_f1_scores = np.zeros(len(cbs))
            for i,cb in enumerate(cbs): 
                for j,idx in enumerate(cb): 
                    t_f1_scores[i] += f1_scores[j,idx]
            self.best_cb_ = (np.arange(N),np.array(cbs[np.argmax(t_f1_scores)]))
        else: 
            cbs = list(it.permutations(range(N),M))
            t_f1_scores = np.zeros(len(cbs))
            for i,cb in enumerate(cbs): 
                for j,idx in enumerate(cb): 
                    t_f1_scores[i] += f1_scores[idx,j]
            self.best_cb_ = (np.array(cbs[np.argmax(t_f1_scores)]),np.arange(M))

        # compute score: 
        self.precision_ = np.sum(precisions[self.best_cb_])/max(N,M)
        self.recall_ = np.sum(recalls[self.best_cb_])/max(N,M)
        self.f1_score_ = np.sum(f1_scores[self.best_cb_])/max(N,M)

        return self.precision_, self.recall_, self.f1_score_


####################################################################################################################
####################################################################################################################
# Event Based
####################################################################################################################
####################################################################################################################

def event_assignement(R_lst:list,P_lst:list)->np.ndarray:
    """Assigned predicted events to real events

    Args:
        R_lst (list): Real label start end list
        P_lst (list): Predicted label start end list
        threshold (float, optional): _description_. Defaults to 0.9.

    Returns:
        np.ndarray : pairing
    """ 
    #compute TP Matrix
    n_r = len(R_lst)
    n_p = len(P_lst)
    arr = np.zeros((n_r,n_p))
    for i,(rs,re) in enumerate(R_lst): 
        for j, (ps,pe) in enumerate(P_lst): 
            s = max(rs,ps)
            e = min(re,pe)
            if e>=s: 
                arr[i,j] = e-s+1
                
    count = min(n_r,n_p)
    pairing = []
    while count>0: 
        i,j = np.unravel_index(np.argmax(arr),(n_r,n_p))
        pairing.append([i,j])
        arr[i,:] = -np.inf
        arr[:,j] = -np.inf
        count -=1
    return np.array(pairing)


def b_e_precision_(R_lst:list,P_lst:list,threshold=0.9)->float: 
    """Event based precision

    Args:
        R_lst (list): Real label start end list
        P_lst (list): Predicted label start end list
        threshold (float, optional): _description_. Defaults to 0.9.

    Returns:
        float: precision score
    """
    if len(R_lst)*len(P_lst) !=0:
        pairing = event_assignement(R_lst,P_lst)
        Rp = R_lst[pairing[:,0]]
        Pp = P_lst[pairing[:,1]]
        score = 0 
        for (rs,re),(ps,pe) in zip(Rp,Pp): 
            s = max(rs,ps)
            e = min(re,pe)
            if e-s>=threshold*(pe-ps): 
                score+=1
        return score/len(P_lst) 
    else: 
        return 0.0

def b_e_recall_(R_lst:list,P_lst:list,threshold=0.9)->float: 
    """Event based precision

    Args:
        R_lst (list): Real label start end list
        P_lst (list): Predicted label start end list
        threshold (float, optional): _description_. Defaults to 0.9.

        Must have a one to one correspondance
    Returns:
        float: recall score
    """
    if len(R_lst)*len(P_lst) != 0:
        pairing = event_assignement(R_lst,P_lst)
        Rp = R_lst[pairing[:,0]]
        Pp = P_lst[pairing[:,1]]
        score = 0 
        for (rs,re),(ps,pe) in zip(Rp,Pp): 
            s = max(rs,ps)
            e = min(re,pe)
            if e-s>=threshold*(re-rs): 
                score+=1
        return score/len(R_lst)
    else: 
        return 0.0

def b_e_f1_score_(R_lst:list,L_lst:list,threshold=0.9)->float: 
    """Event based precision

    Args:
        R_lst (list): Real label start end list
        P_lst (list): Predicted label start end list
        threshold (float, optional): _description_. Defaults to 0.9.

        Must have a one to one correspondance
    Returns:
        float: precision score
    """
    a = b_e_precision_(R_lst,L_lst,threshold)
    b = b_e_recall_(R_lst,L_lst,threshold)
    if a+b == 0:
        return 0
    else:
        return 2*a*b/(a+b)

class SingleEventScore(object): 

    def __init__(self,nbins=101) -> None:
        """Intialization

        Args:
            nbins (int, optional): number of bins to compute the AUC. Defaults to 100.
        """
        self.nbins = nbins
        
    def all_score(self,R:np.ndarray,P:np.ndarray)->tuple: 
        """Compute precision, recall, f1_score for a single event. 

        Args:
            R (np.ndarray): Real label mask
            P (np.ndarray): Predicted label mask
            
        Returns:
            tuple: precision, recall, f1_score
        """
        R_lst = transfrom_label([R])[0]
        P_lst = transfrom_label([P])[0]

        thresholds = np.linspace(0,1,self.nbins)
        self.precision_lst_ = []
        self.recall_lst_ = []
        self.f1_score_lst_ = []

        for threshold in thresholds: 
            t_precision = b_e_precision_(R_lst,P_lst,threshold)
            self.precision_lst_.append(t_precision)
            t_recall = b_e_recall_(R_lst,P_lst,threshold)
            self.recall_lst_.append(t_recall)
            if t_recall + t_precision != 0: 
                self.f1_score_lst_.append(2*t_precision*t_recall/(t_precision + t_recall))
            else: 
                self.f1_score_lst_.append(0.)

        self.precision_lst_ = np.array(self.precision_lst_)
        self.recall_lst_ = np.array(self.recall_lst_)
        self.f1_score_lst_ = np.array(self.f1_score_lst_)

        return self.precision_lst_, self.recall_lst_,self.f1_score_lst_

    def score(self,R:np.ndarray,P:np.ndarray,threshold=0.1)->tuple: 
        """Compute precision, recall, f1_score for a single event. 

        Args:
            R (np.ndarray): Real label mask
            P (np.ndarray): Predicted label mask
            threshold (float, optional): threshold. Defaults to 0.1.
            compute_intervals (bool, optional): compute the list for threshold in intervals.

        Returns:
            tuple: precision, recall, f1_score
        """
        R_lst = transfrom_label([R])[0]
        P_lst = transfrom_label([P])[0]
        
        self.precision_ = b_e_precision_(R_lst,P_lst,threshold)
        self.recall_ = b_e_recall_(R_lst,P_lst,threshold)
        self.f1_score_ = b_e_f1_score_(R_lst,P_lst,threshold)

        return self.precision_,self.recall_,self.f1_score_

    def plot_score(self): 
        line = np.linspace(0,1,self.nbins)

        fig,axs = plt.subplots(1,3,figsize = (10,5),sharey=True,sharex=True)
        axs[0].plot(line,self.precision_lst_)
        axs[0].set_title('Precision')
        axs[1].plot(line,self.recall_lst_)
        axs[1].set_title('Recall')
        axs[2].plot(line,self.f1_score_lst_)
        axs[2].set_title('F1-score')

        axs[0].set_ylabel('Score')
        axs[0].set_ylim(-0.02,1.02)
        axs[1].set_xlabel('Threshold')
        return fig,axs






class EventScore(object):

    def __init__(self,nbins=101) -> None:
        """Intialization

        Args:
            nbins (int, optional): number of bins to compute the AUC. Defaults to 100.
        """
        self.nbins = nbins

    def _find_best_permutation(self,R_lst:np.ndarray,P_lst:np.ndarray)->None: 
        """Compute the scoring

        Args:
            R_lst (np.ndarray): Real Label start end list for each prediction line
            P_lst (np.ndarray): Predicted label start end list for each prediction line
        """

        N = len(R_lst)
        M = len(P_lst)

        thresholds = np.linspace(0,1,self.nbins)
        precisions = np.zeros((self.nbins,N,M))
        recalls = np.zeros((self.nbins,N,M))


        #Compute precision and recall
        for i,r in enumerate(R_lst):
            for j, p in enumerate(P_lst): 
                for k,t_threshold in enumerate(thresholds):
                    precisions[k,i,j] = b_e_precision_(r,p,t_threshold)
                    recalls[k,i,j] = b_e_recall_(r,p,t_threshold)
   
        #compute f1-score AUC
        f1_scores= np.divide(2*precisions*recalls,precisions+recalls,out=np.zeros((self.nbins,N,M)), where= (precisions+recalls)!=0)
        f1_auc = np.sum(f1_scores,axis=0)

        #find the best permutation
        if N<=M: 
            cbs = list(it.permutations(range(M),N))
            t_f1_aucs = np.zeros(len(cbs))
            for i,cb in enumerate(cbs): 
                for j,idx in enumerate(cb): 
                    t_f1_aucs[i] += f1_auc[j,idx]
            self.best_cb_ = np.vstack((np.arange(N),np.array(cbs[np.argmax(t_f1_aucs)]))).T
        else: 
            cbs = list(it.permutations(range(N),M))
            t_f1_aucs = np.zeros(len(cbs))
            for i,cb in enumerate(cbs): 
                for j,idx in enumerate(cb): 
                    t_f1_aucs[i] += f1_auc[idx,j]
            self.best_cb_ = np.vstack((np.array(cbs[np.argmax(t_f1_aucs)]),np.arange(M))).T

        #compute score intervals for best permutation
        self.precision_lst_ = np.sum(precisions[:,self.best_cb_[:,0],self.best_cb_[:,1]],axis=1)/max(N,M)
        self.recall_lst_ = np.sum(recalls[:,self.best_cb_[:,0],self.best_cb_[:,1]],axis=1)/max(N,M)
        self.f1_score_lst_ = np.sum(f1_scores[:,self.best_cb_[:,0],self.best_cb_[:,1]],axis=1)/max(N,M)

    def all_score(self,R:np.ndarray,P:np.ndarray)->tuple: 

        R_lst = transfrom_label(R)
        P_lst = transfrom_label(P)
        self._find_best_permutation(R_lst,P_lst)

        return self.precision_lst_,self.recall_lst_,self.f1_score_lst_

    def score(self,R:np.ndarray,P:np.ndarray,threshold=0.1)->tuple: 

        R_lst = transfrom_label(R)
        P_lst = transfrom_label(P)

        N = len(R_lst)
        M = len(P_lst)

        self._find_best_permutation(R_lst,P_lst)

        # compute score: 
        self.precision_ = 0
        self.recall_ = 0 
        self.f1_score_ = 0
        for i,j in self.best_cb_:
            r = R_lst[i]
            p = P_lst[j]
            t_precision = b_e_precision_(r,p,threshold)
            t_recall = b_e_recall_(r,p,threshold)
            self.precision_ += t_precision
            self.recall_ += t_recall
            if t_precision + t_recall != 0: 
                self.f1_score_ += 2*t_precision*t_recall / (t_precision + t_recall)

        self.precision_ /= max(N,M)
        self.recall_ /= max(N,M)
        self.f1_score_ /= max(N,M)

        return self.precision_, self.recall_, self.f1_score_

    def plot_score(self): 
        line = np.linspace(0,1,self.nbins)

        fig,axs = plt.subplots(1,3,figsize = (10,5),sharey=True,sharex=True)
        axs[0].plot(line,self.precision_lst_)
        axs[0].set_title('Precision')
        axs[1].plot(line,self.recall_lst_)
        axs[1].set_title('Recall')
        axs[2].plot(line,self.f1_score_lst_)
        axs[2].set_title('F1-score')

        axs[0].set_ylabel('Score')
        axs[0].set_ylim(-0.02,1.02)
        axs[1].set_xlabel('Threshold')
        return fig,axs

