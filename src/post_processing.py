import numpy as np
from utils import transfrom_label
    
class PostProcessingBis(object): 

    def __init__(self,wlen:int,min_wlen = None, similar_length = False, similarity = 0.25) -> None:
        """Initialization

        Args:
            wlen (int): window length used to compute KNN
            min_wlen (int, optional): minimum length of an occurences. Defaults to wlen//2.
            similar_length (bool, optional): Keep only occurences with similar length. Defaults to False.
            similarity (float, optional): similarity to average length. Defaults to 0.25.
        """
        self.wlen = wlen
        if min_wlen is None: 
            self.min_wlen = wlen
        else:
            self.min_wlen = min_wlen
        self.similar_length = similar_length
        self.similarity = similarity

    def _initialize(self,idx_lst:list,mp:np.ndarray): 
        start_lst = []
        end_lst = []
        birth_lst = []
        for lst in idx_lst: 
            idxs = np.sort(lst)
            rpt_idx = np.where(np.diff(idxs)>1)[0]
            starts = np.hstack(([idxs[0]],idxs[rpt_idx+1]))
            ends = np.hstack((idxs[rpt_idx],[idxs[-1]]))
            births = []
            for s,e in zip(starts,ends):
                if s != e: 
                    births.append(np.min(mp[s:e]))
                else: 
                    births.append(mp[s])
            start_lst.append(starts)
            end_lst.append(ends)
            birth_lst.append(births)
        label_lst = [np.full(len(lst),i) for i,lst in enumerate(start_lst)]
        starts = np.hstack(start_lst)
        time_sort = np.argsort(starts)
        self.starts_ = starts[time_sort]
        self.ends_ = np.hstack(end_lst)[time_sort]
        self.births_ = np.hstack(birth_lst)[time_sort]
        self.labels_ = np.hstack(label_lst)[time_sort]
        self.sort_idx_ = np.argsort(self.births_)[::-1]
        self.keep_mask_ = np.full_like(self.sort_idx_,True).astype(bool)

    def _next_valid_start(self,idx:int)->int: 
        try: 
            t_s = self.starts_[idx+1:]
            t_mask = self.keep_mask_[idx+1:]
            return t_s[t_mask][0]
        except: 
            return np.inf  
        
    def _previous_valid_start(self,idx:int)->int: 
        try: 
            t_s = self.starts_[:max(idx-1,0)]
            t_mask = self.keep_mask_[:max(idx-1,0)]
            return t_s[t_mask][-1]
        except: 
            return -np.inf 

    def _maximal_window(self,idx):
        r_s = self.starts_[idx]
        end = self.ends_[idx]
        r_e = min(self.n_samples_,end+self.wlen,self._next_valid_start(idx)-1)
        return r_s,r_e
    
    def _similar_length(self): 
        for label in np.unique(self.labels_[self.keep_mask_]):
           mask =  self.keep_mask_ * (self.labels_ == label)
           starts = self.starts_[mask]
           ends = self.ends_[mask]
           dists = ends - starts
           mdist = np.median(dists)
           t_mask = np.abs(dists - mdist) > self.similarity*mdist
           remonve_idx = np.where(mask == True)[0][t_mask]
           self.keep_mask_[remonve_idx] = False

    def fit(self,idx_lst:list,mp:np.ndarray)->None: 
        """Fit 

        Args:
            idx_lst (list): list of list of index per motif. 
            mp (np.ndarray): Birth profile
        """
        #initialize
        self.n_samples_ = mp.shape[0] + self.wlen -1
        self._initialize(idx_lst,mp)
        
        #compute windows
        for idx in self.sort_idx_: 
            r_s,r_e = self._maximal_window(idx)
            pr_s = self._previous_valid_start(idx)
            if (r_e - r_s >= self.min_wlen) * (r_s - pr_s >= self.min_wlen): 
                self.starts_[idx] = r_s
                self.ends_[idx] = r_e
            else:
                self.keep_mask_[idx] = False

        #limit to similar length if required
        if self.similar_length: 
            self._similar_length()

        # remove one reccurence pattern: 
        labels, counts = np.unique(self.labels_[self.keep_mask_],return_counts=True)
        for label in labels[counts<=1]: 
            self.keep_mask_[self.labels_ == label] = False

        return self
    
    @property
    def predictions_(self): 
        labels = self.labels_[self.keep_mask_]
        starts = self.starts_[self.keep_mask_]
        ends = self.ends_[self.keep_mask_]
        births = self.births_[self.keep_mask_]
        return labels,starts,ends,births
    

    @property
    def prediction_mask_(self): 
        labels,starts,ends,_ = self.predictions_
        label_repr = np.unique(labels)
        mask = np.zeros((len(label_repr),self.n_samples_))
        for i,label in enumerate(label_repr): 
            t_start = starts[labels == label]
            t_end = ends [labels == label]
            for s,e in zip(t_start,t_end): 
                mask[i,s:e] =1
        return mask
    
    @property
    def prediction_birth_list_(self): 
        labels,starts,ends,births = self.predictions_
        unique_labels = np.unique(labels)
        label_lst = []
        birth_lst = []
        for label in unique_labels: 
            mask = labels == label
            label_lst.append(np.vstack((starts[mask],ends[mask])).T)
            birth_lst.append(births[mask])
        return label_lst,birth_lst


#########################################################################################################################################
#########################################################################################################################################


class PostProcessing(object): 

    def __init__(self,wlen:int, min_wlen = None,similar_length = False, similarity = 0.25) -> None:
        """Initialization

        Args:
            wlen (int): window length used to compute KNN
            min_wlen (int, optional): minimum length of an occurences. Defaults to wlen//2.
            similar_length (bool, optional): Keep only occurences with similar length. Defaults to False.
            similarity (float, optional): similarity to average length. Defaults to 0.25.
        """
        self.wlen = wlen
        self.similar_length = similar_length
        self.similarity = similarity

    
    def _similar_length(self): 
        for label in np.unique(self.labels_[self.keep_mask_]):
           mask =  self.keep_mask_ * (self.labels_ == label)
           starts = self.starts_[mask]
           ends = self.ends_[mask]
           dists = ends - starts
           mdist = np.median(dists)
           t_mask = np.abs(dists - mdist) > self.similarity*mdist
           remove_idx = np.where(mask == True)[0][t_mask]
           self.keep_mask_[remove_idx] = False

    def fit(self,idx_lst:list,mp:np.ndarray)->None: 
        """Fit 

        Args:
            idx_lst (list): list of list of index per motif. 
            mp (np.ndarray): Birth profile
        """
        #initialize
        self.n_samples_ = mp.shape[0] + self.wlen -1

        cluster_id = 1
        label_lst = []
        cluster_lst = []
        sidx_lst = [] 
        for i,lst in enumerate(idx_lst): 
            idxs = np.sort(lst)
            p_idx = idxs[0]-1
            for idx in idxs: 
                if idx-1 == p_idx:
                    cluster_lst.append(cluster_id)
                else: 
                    cluster_id +=1
                    cluster_lst.append(cluster_id)
                sidx_lst.append(idx)
                label_lst.append(i)
                p_idx = idx

        sort_idx = np.argsort(mp[sidx_lst])
        cluster_lst = np.array(cluster_lst)[sort_idx]
        sidx_lst = np.array(sidx_lst)[sort_idx]
        label_lst = np.array(label_lst)[sort_idx]

        mask = np.zeros(self.n_samples_)
        prediction_mask = np.zeros((len(idx_lst),self.n_samples_))
        for c,idx,l in zip(cluster_lst,sidx_lst,label_lst):
             if np.all((mask[idx:min(idx+self.wlen,self.n_samples_)] == c) + (mask[idx:min(idx+self.wlen,self.n_samples_)] == 0)):
                mask[idx:min(idx+self.wlen,self.n_samples_)] = c
                prediction_mask[l,idx:min(idx+self.wlen,self.n_samples_)-1] =1

        #post traitement
        prediction_mask = prediction_mask[np.sum(prediction_mask,axis=1)>0]
        starts = []
        ends = []
        births = []
        labels = []
        label_idx = 0
        for line in prediction_mask:
            line = np.hstack(([0],line,[0]))
            diff = np.diff(line)
            start = np.where(diff==1)[0]+1
            end = np.where(diff==-1)[0]
            if len(start)>1:
                birth = [np.min(mp[s:e-self.wlen+3]) for s,e in zip(start,end)]
                label = [label_idx]*len(start)
                starts += start.tolist()
                ends += end.tolist()
                births += birth
                labels += label
                label_idx +=1

        self.starts_ = np.array(starts)
        self.ends_ = np.array(ends)
        self.births_ = np.array(births)
        self.labels_ = np.array(labels,dtype=int)
        self.keep_mask_ = np.full_like(self.labels_,True).astype(bool)
            

        #limit to similar length if required
        if self.similar_length: 
            self._similar_length()

        # remove one reccurence pattern: 
        labels, counts = np.unique(self.labels_[self.keep_mask_],return_counts=True)
        for label in labels[counts<=1]: 
            self.keep_mask_[self.labels_ == label] = False

        return self
    
    @property
    def predictions_(self): 
        labels = self.labels_[self.keep_mask_]
        starts = self.starts_[self.keep_mask_]
        ends = self.ends_[self.keep_mask_]
        births = self.births_[self.keep_mask_]
        return labels,starts,ends,births
    
    @property
    def prediction_mask_(self): 
        labels,starts,ends,_ = self.predictions_
        label_repr = np.unique(labels)
        mask = np.zeros((len(label_repr),self.n_samples_))
        for i,label in enumerate(label_repr): 
            t_start = starts[labels == label]
            t_end = ends [labels == label]
            for s,e in zip(t_start,t_end): 
                mask[i,s:e] =1
        return mask


    @property
    def prediction_birth_list_(self): 
        labels,starts,ends,births = self.predictions_
        unique_labels = np.unique(labels)
        label_lst = []
        birth_lst = []
        for label in unique_labels: 
            mask = labels == label
            label_lst.append(np.vstack((starts[mask],ends[mask])).T)
            birth_lst.append(births[mask])
        return label_lst,birth_lst

    
             