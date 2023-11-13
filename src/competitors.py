import numpy as np
from joblib import Parallel,delayed
import distance as distance
import itertools as it 
from functools import partial
from pathlib import Path
import subprocess 
import os

import warnings
warnings.filterwarnings('ignore')


#########################################################################################################################################################################
#########################################################################################################################################################################

class Baseline(object): 

    def __init__(self,n_patterns:int,radius:int,wlen:int,distance_name:str,distance_params = dict(),n_jobs = 1) -> None:
        """KNN initialization

        Args:
            n_neighbors (int): Number of neighbors
            wlen (int): Window length
            distance_name (str): name of the distance
            distance_params (_type_, optional): additional distance parameters. Defaults to dict().
            n_jobs (int, optional): number of processes. Defaults to 1.
        """
        self.n_patterns = n_patterns
        self.radius = radius
        self.wlen = wlen
        self.distance_name = distance_name
        self.distance_params = distance_params
        self.n_jobs = n_jobs


    def _search_neighbors(self,idx:int,line:np.ndarray)-> tuple: 
        """Find index and distance value of the non overlapping nearest neighbors under a radius.

        Args:
            idx (int): index of the considerded line in the crossdistance matrix
            line (np.ndarray): line of the crossdistance matrix. shape: (n_sample,)

        Returns:
            tuple: neighbor index np.ndarray, neighbor distance np.ndarray
        """

        #initilization
        neighbors = []
        dists = []
        idxs = np.arange(self.mdim_)
        remove_idx = np.arange(max(0,idx-self.wlen+1),min(self.mdim_,idx+self.wlen))
        idxs = np.delete(idxs,remove_idx)
        line = np.delete(line,remove_idx)

        #search loop
        t_distance = np.min(line)
        while t_distance < self.radius:
            try: 
                #local next neighbor
                t_idx = np.argmin(line)
                if line[t_idx] == np.inf:
                    break
                neighbors.append(idxs[t_idx])
                dists.append(line[t_idx])

                #remove window
                remove_idx = np.arange(max(0,t_idx-self.wlen+1),min(len(line),t_idx+self.wlen))
                idxs = np.delete(idxs,remove_idx)
                line = np.delete(line,remove_idx)

                t_distance = dists[-1]
            except: 
                break
            
        return neighbors,dists

    def _elementary_neighborhood(self,start:int,end:int)->tuple:
        """Find elementary neighborhood of a chunk of successive lines of the crossdistance matrix

        Args:
            start (int): chunk start
            end (int): chunck end

        Returns:
            tuple: neighborhood count, neighborhood std
        """
        #initialization
        neighbors =[]
        dists = []
        line = self.distance_.first_line(start)
        t_neighbors,t_dists = self._search_neighbors(start,line)
        neighbors.append(t_neighbors)
        dists.append(t_dists)
        
        #main loop
        for i in range(start+1,end): 
            line = self.distance_.next_line()
            t_neighbors,t_dists = self._search_neighbors(i,line)
            neighbors.append(t_neighbors)
            dists.append(t_dists)
        return neighbors,dists

    def neighborhood_(self)->None: 

        #divide the signal accordingly to the number of jobs
        set_idxs = np.linspace(0,self.mdim_,self.n_jobs+1,dtype=int)
        set_idxs = np.vstack((set_idxs[:-1],set_idxs[1:])).T

        # set the parrallel computation
        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._elementary_neighborhood)(*set_idx)for set_idx in set_idxs
        )

        idxs,dists = list(zip(*results))
        self.idxs_ = list(it.chain(*idxs))
        self.dists_ = list(it.chain(*dists))
        return self

    def find_patterns_(self): 

        self.counts_ = np.array([len(lst) for lst in self.idxs_])
        stds = []
        stds = []
        for lst in self.dists_: 
            if len(lst)>0: 
                stds.append(np.std(lst))
            else: 
                stds.append(np.inf)
        self.stds_ = np.array(stds)
        self.sort_idx_ = np.lexsort((self.stds_,-self.counts_))
        patterns = [self.sort_idx_[0]]

        for idx in self.sort_idx_[1:]: 
            if len(patterns) <self.n_patterns: 
                dist_to_patten = np.array([self.distance_.individual_distance(idx,p_idx) for p_idx in patterns])
                if np.all(dist_to_patten > 2*self.radius): 
                    patterns.append(idx)
            else: 
                break

        self.patterns_ = patterns

    def fit(self,signal:np.ndarray)->None:
        """Compute the best patterns

        Args:
            signal (np.ndarray): Univariate time-series, shape: (L,)
        """
        
        #initialisation
        self.signal_ = signal
        self.mdim_ = len(signal)-self.wlen+1 
        self.distance_ = getattr(distance,self.distance_name)(self.wlen,**self.distance_params)
        self.distance_.fit(signal)

        #Compute neighborhood
        self.neighborhood_()
        #find patterns
        self.find_patterns_()

        return self

    @property
    def prediction_mask_(self)->np.ndarray:
        """Create prediction mask

        Returns:
            np.ndarray: prediction mask, shape (n_patterns, L-wlen+1)
        """
        mask = np.zeros((self.n_patterns,self.signal_.shape[0]))
        for i,p_idx in enumerate(self.patterns_):
            mask[i,p_idx:p_idx+self.wlen] =1
            for idx in self.idxs_[p_idx]:
                mask[i,idx:idx+self.wlen] =1 

        return mask

#########################################################################################################################################################################
#########################################################################################################################################################################    

class LatentMotif(object): 
    
    def __init__(self,n_patterns:int,wlen:int,radius:float,alpha = 1.0,learning_rate =0.1,n_iterations = 100, n_starts = 1, verbose = False) -> None:
        """Initialization

        Args:
            n_patterns (int): number of patterns
            wlen (int): window length 
            radius (float): cluster radius
            alpha (float, optional): regularization parameter. Defaults to 1.0.
            learning_rate (float, optional): learning rate. Defaults to 0.1.
            n_iterations (int, optional): number of gradient iteration. Defaults to 100.
            n_strats (int, optional): number of trials. Defaults to 10.
            verbose (bool, optional): verbose. Defaults to False.
        """
        self.n_patterns = n_patterns
        self.wlen = wlen
        self.radius = radius
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_starts = n_starts
        self.verbose = verbose

    def _freq(self, patterns:np.ndarray)->float: #verified
        """Frequence score

        Args:
            patterns (np.ndarray): patterns, shape(n_patterns, wlen)

        Returns:
            float: frequence score
        """
        dist = np.sum((self.set_[:,np.newaxis,:] - patterns[np.newaxis,...])**2,axis=2)
        exp_dist = np.exp(-self.alpha/self.radius * dist)
        freq = 1/(self.n_patterns * self.set_size_) * np.sum(exp_dist)
        return freq

    def _pen(self, patterns): #verified 
        """penalty score

        Args:
            patterns (np.ndarray): patterns, shape(n_patterns, wlen)

        Returns:
            float: penalty score
        """
        if self.n_patterns>1:
            dist = np.sum((patterns[:,np.newaxis,:] - patterns[np.newaxis,...])**2,axis=2)
            pen_m = np.where(dist < 2*self.radius, (1 - dist/(2*self.radius))**2, 0)
            pen = 2/(self.n_patterns*(self.n_patterns -1))*np.sum(np.triu(pen_m,k=1))
        else: 
            pen = 0
        return pen  

    def _score(self,patterns): #verified
        """Score

        Args:
            patterns (np.ndarray): patterns, shape(n_patterns, wlen)

        Returns:
            float: Score
        """
        return self._freq(patterns) - self._pen(patterns)


    def _freq_derivative(self, patterns): 
        """Frequence deriavative

        Args:
            patterns (np.ndarray): patterns, shape(n_patterns, wlen)

        Returns:
            float: frequence derivative
        """
        diff = self.set_[:,np.newaxis,:] - patterns[np.newaxis,...]
        exp_dist = np.exp(-self.alpha/self.radius * np.sum(diff**2,axis=2))
        div_freq = -2 * self.alpha / (self.n_patterns * self.set_size_ * self.radius) * np.sum(exp_dist[...,np.newaxis]*diff, axis=0)
        return div_freq

    def _pen_derivative(self, patterns): 
        """Frequence derivative

        Args:
            patterns (np.ndarray): patterns, shape(n_patterns, wlen)

        Returns:
            float: frequence derivative
        """
        diff = patterns[:,np.newaxis,:] - patterns[np.newaxis,...]
        dist = np.sum(diff**2, axis =2)
        pen_m = np.where(dist < 2*self.radius, 2* self.radius - dist, 0)
        div_pen = -2 / (self.radius**2 * self.n_patterns * (self.n_patterns - 1)) * np.sum(pen_m[...,np.newaxis]*diff,axis=0)
        return div_pen
        

    def fit(self,signal:np.ndarray)->None:
        """Fit

        Args:
            signal (np.ndarray): signal, shape: (L,)
        """

        #initialization 
        self.signal_ = signal
        self.set_ = np.lib.stride_tricks.sliding_window_view(signal,self.wlen)
        self.set_ = (self.set_-np.mean(self.set_,axis=1).reshape(-1,1))/np.std(self.set_,axis=1).reshape(-1,1)
        self.set_size_ = self.set_.shape[0]
        self.score_ = -np.inf
        self.patterns_ = np.zeros((self.n_patterns,self.wlen))

        if self.verbose: 
            print("Start Trials")
        for i in range(self.n_starts): 
            patterns, score = self.one_fit_()
            if self.verbose:
                print(f"Trial: {i+1}/{self.n_starts}, score : {score}")
            if score > self.score_: 
                self.score_ = score
                self.patterns_ = patterns

        if self.verbose: 
            print(f"Successfully finished, best score: {self.score_}")

        return self
    
    def one_fit_(self):
        """One learning trial

        Returns:
            np.ndarray, float: patterns, score
        """

        patterns = np.random.randn(self.n_patterns, self.wlen)
        rate_adapt = np.zeros((self.n_patterns, self.wlen))

        for i in range(self.n_iterations): 
            if self.n_patterns>1:
                div = self._freq_derivative(patterns) - self._pen_derivative(patterns)
            else: 
                div = self._freq_derivative(patterns)
            rate_adapt += div**2
            patterns -= self.learning_rate/np.sqrt(rate_adapt) * div

            if self.verbose: 
                print(f"Iteration: {i+1}/{self.n_iterations}, score: {self._score(patterns)} ")

        score = self._score(patterns)
        return patterns, score

    @property
    def prediction_mask_(self)->np.ndarray: 
        """Create prediction mask

        Returns:
            np.ndarray: prediction mask, shape (n_patterns, L-wlen+1)
        """
        dist = np.sum((self.set_[:,np.newaxis,:] - self.patterns_[np.newaxis,...])**2,axis=2)
        idx_lsts = []
        for line in dist.T: 
            idxs = np.arange(self.set_size_-self.wlen+1)
            idx_lst = []
            t_distance = np.min(line)
            while t_distance < self.radius:
                try: 
                    #local next neighbor
                    t_idx = np.argmin(line)
                    idx_lst.append(idxs[t_idx])
                    t_distance = line[t_idx]

                    #remove window
                    remove_idx = np.arange(max(0,t_idx-self.wlen+1),min(len(line),t_idx+self.wlen))
                    idxs = np.delete(idxs,remove_idx)
                    line = np.delete(line,remove_idx)

                except: 
                    break
            idx_lsts.append(idx_lst)

        mask = np.zeros((self.n_patterns,self.signal_.shape[0]))
        for i,p_idx in enumerate(idx_lsts):
            for idx in p_idx:
                mask[i,idx:idx+self.wlen] =1 
        return mask

#########################################################################################################################################################################
#########################################################################################################################################################################


class MatrixProfile(object): 

    def __init__(self,n_patterns:int,wlen:int,distance_name:str,distance_params = dict(),radius_ratio = 3,n_jobs = 1) -> None:
        """Initialization

        Args:
            n_patterns (int): Number of neighbors
            wlen (int): Window length
            distance_name (str): name of the distance
            distance_params (_type_, optional): additional distance parameters. Defaults to dict().
            radius_ratio (float): radius as a ratio of min_dist. 
            n_jobs (int, optional): number of processes. Defaults to 1.
        """
        self.n_patterns = n_patterns
        self.radius_ratio = radius_ratio
        self.wlen = wlen
        self.distance_name = distance_name
        self.distance_params = distance_params
        self.n_jobs = n_jobs

    def _search_neighbors(self,idx:int,line:np.ndarray)-> tuple: 
        """Find index and distance value of the non overlapping nearest neighbors under a radius.

        Args:
            idx (int): index of the considerded line in the crossdistance matrix
            line (np.ndarray): line of the crossdistance matrix. shape: (n_sample,)

        Returns:
            tuple: neighbor index np.ndarray, neighbor distance np.ndarray
        """

        #initilization
        neighbors = []
        dists = []
        idxs = np.arange(self.mdim_)
        remove_idx = np.arange(max(0,idx-self.wlen+1),min(self.mdim_,idx+self.wlen))
        idxs = np.delete(idxs,remove_idx)
        line = np.delete(line,remove_idx)

        #search loop
        radius = np.min(line)*self.radius_ratio
        t_distance = np.min(line)
        while t_distance < radius:
            try: 
                #local next neighbor
                t_idx = np.argmin(line)
                neighbors.append(idxs[t_idx])
                dists.append(line[t_idx])

                #remove window
                remove_idx = np.arange(max(0,t_idx-self.wlen+1),min(len(line),t_idx+self.wlen))
                idxs = np.delete(idxs,remove_idx)
                line = np.delete(line,remove_idx)

                t_distance = dists[-1]
            except: 
                break
            
        return neighbors,dists

    def _elementary_profile(self,start:int,end:int)->tuple:
        """Find elementary profile of a chunk of successive lines of the crossdistance matrix

        Args:
            start (int): chunk start
            end (int): chunck end

        Returns:
            tuple: neighborhood count, neighborhood std
        """
        #initialization
        neighbors =[]
        dists = []
        line = self.distance_.first_line(start)
        mask = np.arange(max(0,start-self.wlen+1), min(self.mdim_,start+self.wlen))
        line[mask] = np.inf
        t_idx = np.argmin(line)
        t_dist = line[t_idx]
        neighbors.append(t_idx)
        dists.append(t_dist)
        
        #main loop
        for i in range(start+1,end): 
            line = self.distance_.next_line()
            mask = np.arange(max(0,i-self.wlen+1), min(self.mdim_,i+self.wlen))
            line[mask] = np.inf
            t_idx = np.argmin(line)
            t_dist = line[t_idx]
            neighbors.append(t_idx)
            dists.append(t_dist)
        return neighbors,dists

    def profile_(self)->None: 

        #divide the signal accordingly to the number of jobs
        set_idxs = np.linspace(0,self.mdim_,self.n_jobs+1,dtype=int)
        set_idxs = np.vstack((set_idxs[:-1],set_idxs[1:])).T

        # set the parrallel computation
        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._elementary_profile)(*set_idx)for set_idx in set_idxs
        )

        idxs,dists = list(zip(*results))

        self.idxs_ = np.hstack(idxs)
        self.dists_ = np.hstack(dists)
        return self

    def find_patterns_(self): 
        profile = self.dists_.copy()
        mask = []
        patterns = []

        for _ in np.arange(self.n_patterns): 
            min_idx = np.argmin(profile)
            if profile[min_idx]==np.inf: 
                break
            line = self.distance_.first_line(min_idx)
            line[mask] = np.inf
            p_idxs,dists = self._search_neighbors(min_idx,line)
            p_idxs = np.hstack((np.array([min_idx]),p_idxs))
            patterns.append(p_idxs)
            mask += np.hstack([np.arange(max(0,idx-self.wlen+1),min(self.mdim_,idx+self.wlen)) for idx in p_idxs]).astype(int).tolist()
            profile[mask] = np.inf
        
        self.patterns_ = patterns

    def fit(self,signal:np.ndarray)->None:
        """Compute the best patterns

        Args:
            signal (np.ndarray): Univariate time-series, shape: (L,)
        """
        
        #initialisation
        self.signal_ = signal
        self.mdim_ = len(signal)-self.wlen+1 
        self.distance_ = getattr(distance,self.distance_name)(self.wlen,**self.distance_params)
        self.distance_.fit(signal)

        #Compute neighborhood
        self.profile_()
        #find patterns
        self.find_patterns_()

        return self

    @property
    def prediction_mask_(self):
        mask = np.zeros((self.n_patterns,self.signal_.shape[0]))
        for i,p_idxs in enumerate(self.patterns_):
            for idx in p_idxs.astype(int):
                mask[i,idx:idx+self.wlen]=1 
        return mask

#########################################################################################################################################################################
#########################################################################################################################################################################

class PanMatrixProfile(object): 

    def __init__(self,n_patterns:int,min_wlen:int,max_wlen:int,distance_name:str,distance_params = dict(),radius_ratio = 3,normalized=False,n_jobs = 1) -> None:
        """Initialization

        Args:
            n_patterns (int): Number of neighbors
            min_wlen (int): Minimum window length
            max_wlen (int): Maximum window length
            distance_name (str): name of the distance
            distance_params (_type_, optional): additional distance parameters. Defaults to dict().
            radius_ratio (float): radius as a ratio of min_dist. 
            n_jobs (int, optional): number of processes. Defaults to 1.
        """
        self.n_patterns = n_patterns
        self.radius_ratio = radius_ratio
        self.min_wlen = min_wlen
        self.max_wlen = max_wlen
        self.distance_name = distance_name
        self.distance_params = distance_params
        self.normalized = normalized
        self.n_jobs = n_jobs

    def _search_neighbors(self,wlen_idx:int,seed_idx:int,line:np.ndarray)-> tuple: 
        """Find index and distance value of the non overlapping nearest neighbors under a radius.

        Args:
            wlen_idx (int): index of the window length and associated profile
            seed_idx (int): index of the considerded line in the crossdistance matrix
            line (np.ndarray): line of the crossdistance matrix. shape: (n_sample,)

        Returns:
            tuple: neighbor index np.ndarray, neighbor distance np.ndarray
        """

        #initilization
        neighbors = []
        dists = []
        idxs = np.arange(self.mdims_[wlen_idx])
        remove_idx = np.arange(max(0,seed_idx-self.wlens_[wlen_idx]+1),min(self.mdims_[wlen_idx],seed_idx+self.wlens_[wlen_idx]))
        idxs = np.delete(idxs,remove_idx)
        line = np.delete(line,remove_idx)

        #search loop
        radius = np.min(line)*self.radius_ratio
        t_distance = np.min(line)
        while t_distance < radius:
            try: 
                #local next neighbor
                t_idx = np.argmin(line)
                if line[t_idx] == np.inf:
                    break
                neighbors.append(idxs[t_idx])
                dists.append(line[t_idx])

                #remove window
                remove_idx = np.arange(max(0,t_idx-self.wlens_[wlen_idx]+1),min(len(line),t_idx+self.wlens_[wlen_idx]))
                idxs = np.delete(idxs,remove_idx)
                line = np.delete(line,remove_idx)

                t_distance = dists[-1]
            except: 
                break
            
        return neighbors,dists

    def _elementary_profile(self,idx:int,start:int,end:int)->tuple:
        """Find elementary profile of a chunk of successive lines of the crossdistance matrix

        Args:
            start (int): chunk start
            end (int): chunck end

        Returns:
            tuple: neighborhood count, neighborhood std
        """
        #initialization
        neighbors =[]
        dists = []
        line = self.distance_[idx].first_line(start)
        mask = np.arange(max(0,start-self.wlens_[idx]+1), min(self.mdims_[idx],start+self.wlens_[idx]))
        line[mask] = np.inf
        t_idx = np.argmin(line)
        t_dist = line[t_idx]
        neighbors.append(t_idx)
        dists.append(t_dist)
        
        #main loop
        for i in range(start+1,end): 
            line = self.distance_[idx].next_line()
            mask = np.arange(max(0,i-self.wlens_[idx]+1), min(self.mdims_[idx],i+self.wlens_[idx]))
            line[mask] = np.inf
            t_idx = np.argmin(line)
            t_dist = line[t_idx]
            neighbors.append(t_idx)
            dists.append(t_dist)
        return neighbors,dists

    def profile_(self,idx:int)->np.ndarray: 
        """Compute profile of wlen

        Args:
            idx (int): window length index

        Returns:
            np.ndarray: profile, nearest neighbor index
        """

        #divide the signal accordingly to the number of jobs
        set_idxs = np.linspace(0,self.mdims_[idx],self.n_jobs+1,dtype=int)
        set_idxs = np.vstack((set_idxs[:-1],set_idxs[1:])).T

        # set the parrallel computation
        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(partial(self._elementary_profile,idx))(*set_idx)for set_idx in set_idxs
        )

        idxs,dists = list(zip(*results))

        return np.hstack(dists),np.hstack(idxs)

    def _temporary_mask(self,wlen_idx:int,mask:list,patterns:list)->list: 
        """Create mask associated with the current research windows

        Args:
            wlen_idx (int): window length index
            mask (list): current mask
            patterns (list): list of patterns already detected

        Returns:
            list: mask for the search of neighbors
        """
        t_mask = mask.copy()
        for _, p_idxs in patterns: 
            for p_idx in p_idxs: 
                t_mask += np.arange(max(0,p_idx-self.wlens_[wlen_idx]+1),p_idx+1).astype(int).tolist()
        t_mask = np.array(t_mask)
        keep_idx = np.where(t_mask<=self.mdims_[wlen_idx])
        return t_mask[keep_idx].tolist()

    def find_patterns_(self): 
        profiles = self.profiles_.copy()
        mask = []
        patterns = []

        for iteration in np.arange(self.n_patterns): 
            if iteration == 0: 
                min_idx = np.argmin(profiles)
                wlen_idx, seed_idx = np.unravel_index(min_idx,profiles.shape)
                line = self.distance_[wlen_idx].first_line(seed_idx)
            else: 
                overlapping = True
                while overlapping and not np.all(profiles == np.inf): 
                    min_idx = np.argmin(profiles)
                    wlen_idx, seed_idx = np.unravel_index(min_idx,profiles.shape)
                    t_mask = self._temporary_mask(wlen_idx,mask,patterns)
                    if seed_idx not in t_mask: 
                        overlapping = False
                    else: 
                        profiles[:,seed_idx] = np.inf
                if not overlapping:
                    line = self.distance_[wlen_idx].first_line(seed_idx)
                    line[t_mask] = np.inf

            if np.all(profiles == np.inf):
                break
            
            p_idxs,dists = self._search_neighbors(wlen_idx,seed_idx,line)
            p_idxs = np.hstack((np.array([seed_idx]),p_idxs))
            patterns.append((self.wlens_[wlen_idx],p_idxs))
            mask += np.hstack([np.arange(max(0, idx - self.min_wlen +1),min(self.mdims_[wlen_idx],idx+self.wlens_[wlen_idx])) for idx in p_idxs]).astype(int).tolist()
            profiles[:,mask] = np.inf

        self.test_ = profiles
        
        self.patterns_ = patterns

    def fit(self,signal:np.ndarray)->None:
        """Compute the best patterns

        Args:
            signal (np.ndarray): Univariate time-series, shape: (L,)
        """
        
        #initialisation
        self.signal_ = signal
        self.profiles_ = []
        self.idxs_ = []
        self.distance_ = []
        self.wlens_ = np.arange(self.min_wlen,self.max_wlen)
        self.mdims_ = signal.shape[0] - self.wlens_ + 1
        for i,wlen in enumerate(self.wlens_):
            self.distance_.append(getattr(distance,self.distance_name)(self.wlens_[i],**self.distance_params))
            self.distance_[i].fit(signal)

            #Compute profile and idxs
            profile,idxs = self.profile_(i)
            gap = self.wlens_[i] - self.min_wlen
            if gap>0: 
                gap_profile = np.full(gap,np.inf)
                gap_idxs = np.full(gap,np.nan)
                profile = np.hstack((profile,gap_profile))
                idxs = np.hstack((idxs,gap_idxs))
            self.profiles_.append(profile)
            self.idxs_.append(idxs)
        
        self.profiles_ = np.array(self.profiles_)
        self.idxs_ = np.array(self.idxs_)

        if self.normalized: 
            self.profiles_ = self.profiles_ / np.sqrt(self.wlens_).reshape(-1,1)

        #find patterns
        self.find_patterns_()

        return self

    @property
    def prediction_mask_(self):
        mask = np.zeros((self.n_patterns,self.signal_.shape[0]))
        for i,(wlen,p_idxs) in enumerate(self.patterns_):
            for idx in p_idxs:
                mask[i,int(idx):int(idx+wlen)] =1 
        return mask

#########################################################################################################################################################################
#########################################################################################################################################################################

class Valmod(object): 

    def __init__(self,n_patterns:int,min_wlen:int,max_wlen:int,distance_name:str,distance_params = dict(),step=1,radius_ratio = 3,n_jobs = 1) -> None:
        """Initialization

        Args:
            n_patterns (int): Number of neighbors
            min_wlen (int): Minimum window length
            max_wlen (int): Maximum window length
            distance_name (str): name of the distance
            distance_params (dict, optional): additional distance parameters. Defaults to dict().
            step (dict, optional): wlen step. Defaults to 1.
            radius_ratio (float): radius as a ratio of min_dist. 
            n_jobs (int, optional): number of processes. Defaults to 1.
        """
        self.n_patterns = n_patterns
        self.radius_ratio = radius_ratio
        self.min_wlen = min_wlen
        self.max_wlen = max_wlen
        self.distance_name = distance_name
        self.step =step
        self.distance_params = distance_params
        self.n_jobs = n_jobs

    def _search_neighbors(self,wlen_idx:int,seed_idx_1:int,seed_idx_2,line:np.ndarray)-> tuple: 
        """Find index and distance value of the non overlapping nearest neighbors under a radius.

        Args:
            wlen_idx (int): index of the window length and associated profile
            seed_idx (int): index of the considerded line in the crossdistance matrix
            line (np.ndarray): line of the crossdistance matrix. shape: (n_sample,)

        Returns:
            tuple: neighbor index np.ndarray, neighbor distance np.ndarray
        """

        #initilization
        neighbors = []
        dists = []
        idxs = np.arange(self.mdims_[wlen_idx])
        remove_idx_1 = np.arange(max(0,seed_idx_1-self.wlens_[wlen_idx]+1),min(self.mdims_[wlen_idx],seed_idx_1+self.wlens_[wlen_idx]))
        remove_idx_2 = np.arange(max(0,seed_idx_2-self.wlens_[wlen_idx]+1),min(self.mdims_[wlen_idx],seed_idx_2+self.wlens_[wlen_idx]))
        remove_idx = np.hstack((remove_idx_1,remove_idx_2))
        idxs = np.delete(idxs,remove_idx)
        line = np.delete(line,remove_idx)

        #search loop
        radius = np.min(line)*self.radius_ratio
        t_distance = np.min(line)
        while t_distance < radius:
            try: 
                #local next neighbor
                t_idx = np.argmin(line)
                if line[t_idx] == np.inf:
                    break
                neighbors.append(idxs[t_idx])
                dists.append(line[t_idx])

                #remove window
                remove_idx = np.arange(max(0,t_idx-self.wlens_[wlen_idx]+1),min(len(line),t_idx+self.wlens_[wlen_idx]))
                idxs = np.delete(idxs,remove_idx)
                line = np.delete(line,remove_idx)

                t_distance = dists[-1]
            except: 
                break
            
        return neighbors,dists

    def _elementary_profile(self,idx:int,start:int,end:int)->tuple:
        """Find elementary profile of a chunk of successive lines of the crossdistance matrix

        Args:
            start (int): chunk start
            end (int): chunck end

        Returns:
            tuple: neighborhood count, neighborhood std
        """
        #initialization
        neighbors =[]
        dists = []
        line = self.distance_[idx].first_line(start)
        mask = np.arange(max(0,start-self.wlens_[idx]+1), min(self.mdims_[idx],start+self.wlens_[idx]))
        line[mask] = np.inf
        t_idx = np.argmin(line)
        t_dist = line[t_idx]
        neighbors.append(t_idx)
        dists.append(t_dist)
        
        #main loop
        for i in range(start+1,end): 
            line = self.distance_[idx].next_line()
            mask = np.arange(max(0,i-self.wlens_[idx]+1), min(self.mdims_[idx],i+self.wlens_[idx]))
            line[mask] = np.inf
            t_idx = np.argmin(line)
            t_dist = line[t_idx]
            neighbors.append(t_idx)
            dists.append(t_dist)
        return neighbors,dists

    def profile_(self,idx:int)->np.ndarray: 
        """Compute profile of wlen

        Args:
            idx (int): window length index

        Returns:
            np.ndarray: profile, nearest neighbor index
        """

        #divide the signal accordingly to the number of jobs
        set_idxs = np.linspace(0,self.mdims_[idx],self.n_jobs+1,dtype=int)
        set_idxs = np.vstack((set_idxs[:-1],set_idxs[1:])).T

        # set the parrallel computation
        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(partial(self._elementary_profile,idx))(*set_idx)for set_idx in set_idxs
        )

        idxs,dists = list(zip(*results))

        return np.hstack(dists),np.hstack(idxs)

    def _temporary_mask(self,wlen_idx:int,mask:list,patterns:list)->list: 
        """Create mask associated with the current research windows

        Args:
            wlen_idx (int): window length index
            mask (list): current mask
            patterns (list): list of patterns already detected

        Returns:
            list: mask for the search of neighbors
        """
        t_mask = mask.copy()
        for _, p_idxs in patterns: 
            for p_idx in p_idxs: 
                t_mask += np.arange(max(0,p_idx-self.wlens_[wlen_idx]+1),p_idx).astype(int).tolist()
        t_mask = np.array(t_mask)
        keep_idx = np.where(t_mask<=self.mdims_[wlen_idx])
        return t_mask[keep_idx].tolist()

    def find_patterns_(self): 
        profiles = self.profiles_.copy()
        mask = []
        patterns = []

        for iteration in np.arange(self.n_patterns): 
            if iteration == 0:
                min_idx = np.argmin(profiles)
                wlen_idx, seed_idx = np.unravel_index(min_idx,profiles.shape)
                line_1 = self.distance_[wlen_idx].first_line(seed_idx)
                line_2 = self.distance_[wlen_idx].first_line(self.idxs_[wlen_idx,seed_idx])
                line = np.min((line_1,line_2),axis=0)
            else:
                overlapping = True
                while overlapping and not np.all(profiles == np.inf): 
                    min_idx = np.argmin(profiles)
                    wlen_idx, seed_idx = np.unravel_index(min_idx,profiles.shape)
                    t_mask = self._temporary_mask(wlen_idx,mask,patterns)
                    if (seed_idx not in t_mask)*(self.idxs_[wlen_idx,seed_idx] not in t_mask): 
                        overlapping = False
                    else: 
                        profiles[:,seed_idx] = np.inf
                if not overlapping: 
                    line_1 = self.distance_[wlen_idx].first_line(seed_idx)
                    line_2 = self.distance_[wlen_idx].first_line(self.idxs_[wlen_idx,seed_idx])
                    line = np.min((line_1,line_2),axis=0)
                    line[t_mask] = np.inf

            if np.all(profiles == np.inf):
                break

            p_idxs,dists = self._search_neighbors(wlen_idx,seed_idx,self.idxs_[wlen_idx,seed_idx],line)
            p_idxs = np.hstack((np.array([seed_idx,self.idxs_[wlen_idx,seed_idx]]),p_idxs))
            patterns.append((self.wlens_[wlen_idx],p_idxs.astype(int)))
            mask += np.hstack([np.arange(max(0,idx-self.min_wlen+1),min(self.mdims_[wlen_idx],idx+self.wlens_[wlen_idx])) for idx in p_idxs]).astype(int).tolist()
            profiles[:,mask] = np.inf
        
        self.patterns_ = patterns

    def fit(self,signal:np.ndarray)->None:
        """Compute the best patterns

        Args:
            signal (np.ndarray): Univariate time-series, shape: (L,)
        """
        
        #initialisation
        self.signal_ = signal
        self.profiles_ = []
        self.idxs_ = []
        self.distance_ = []
        self.wlens_ = np.arange(self.min_wlen,self.max_wlen,self.step)
        self.mdims_ = signal.shape[0] - self.wlens_
        for i,wlen in enumerate(self.wlens_):
            self.distance_.append(getattr(distance,self.distance_name)(self.wlens_[i],**self.distance_params))
            self.distance_[i].fit(signal)

            #Compute profile and idxs
            profile,idxs = self.profile_(i)
            gap = self.wlens_[i] - self.min_wlen
            if gap>0: 
                gap_profile = np.full(gap,np.inf)
                gap_idxs = np.full(gap,np.nan)
                profile = np.hstack((profile,gap_profile))
                idxs = np.hstack((idxs,gap_idxs))
            self.profiles_.append(profile)
            self.idxs_.append(idxs)
        
        self.profiles_ = np.array(self.profiles_)
        self.profiles_ = self.profiles_ / np.sqrt(self.wlens_).reshape(-1,1)
        self.idxs_ = np.array(self.idxs_,dtype = int)

        #find patterns
        self.find_patterns_()

        return self

    @property
    def prediction_mask_(self):
        mask = np.zeros((self.n_patterns,self.signal_.shape[0]))
        for i,(wlen,p_idxs) in enumerate(self.patterns_):
            for idx in p_idxs:
                mask[i,int(idx):int(idx+wlen)] =1 
        return mask

#########################################################################################################################################################################
#########################################################################################################################################################################

class Grammarviz(object): 

    def __init__(self,n_patterns:int,alphabet_size=4,numerosity="MINDIST",window_size = 30,word_size = 6,folder_java = "./src/grammarviz",file_exchange_location="target/file_exchange") -> None:
        """_summary_

        Args:
            n_cluster (int): number of cluster
            alphabet_size (int, optional): alphabet size. Defaults to 4.
            numerosity (str, optional): numerosity reduction type. Defaults to "MINDIST".
            window_size (int, optional): window size. Defaults to 30.
            word_size (int, optional): word size. Defaults to 6.
            folder_java (str, optional): path to grammarviz java folder. Defaults to "/Users/tgermain/Documents/code/GrammarViz/grammarviz2_src".
            file_exchange_location (str, optional): path to the exchange files folder. Defaults to "target/file_exchange".
        """
        self.n_cluster = n_patterns
        self.alphabet_size = alphabet_size
        self.numerosity = numerosity
        self.window_size = window_size
        self.word_size = word_size
        self.folder_java = Path(folder_java)
        self.file_exchange_location = Path(file_exchange_location)

    def fit(self,signal:np.ndarray)->None:
        """fit siganl

        Args:
            signal (np.ndarray): signal, shape: (n_ts,)

        """
        self.signal_length = signal.shape[0]
        #prepare data
        self.data_path_= self.file_exchange_location/"data.txt"
        try: 
            os.remove(self.folder_java/self.data_path_)
        except: 
            pass
        self.output_path_ = self.file_exchange_location/"output.txt"
        try: 
            os.remove(self.folder_java/self.output_path_)
        except: 
            pass

        np.savetxt(self.folder_java/self.data_path_,signal)

        #Execute algorithm: 
        command = f"java -cp \"target/grammarviz2-1.0.1-SNAPSHOT-jar-with-dependencies.jar\" net.seninp.grammarviz.cli.TS2SequiturGrammar -d target/file_exchange/data.txt -o target/file_exchange/output.txt -a {self.alphabet_size} -p {self.word_size} -w {self.window_size} --strategy {self.numerosity}"
        subprocess.run([command],shell=True,cwd = self.folder_java/"",stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

        #read output
        self.prediction_mask_ = self._read_grammaviz_result()

        os.remove(self.folder_java/self.data_path_)
        os.remove(self.folder_java/self.output_path_)
        
        return self

    def _read_grammaviz_result(self)->np.ndarray:
        """Read output txt file to compute the mask

        Returns:
            np.ndarray: reccurent pattern prediction mask.
        """

        with open(self.folder_java/self.output_path_,"r") as f: 
            res = f.read()

        r0 = res.split("///")[1]
        r0 = r0.split("\n")[1]
        r0 = r0.split(",")[0]
        r0 = r0.split("-> ")[1]
        r0 = r0.replace(" ", ",")
        r0 = r0[1:-1]
        r0 = r0.split(",")
        rlabels, rcounts = np.unique(r0,return_counts=True)
        ridxs = np.array([i for i,r in enumerate(rlabels) if r[0]=="R"])
        if len(ridxs)>0:
            rlabels = rlabels[ridxs]
            rcounts = rcounts[ridxs]
            n_rules = ridxs.shape[0]
            counts = 0
            idxs = []

            while (counts < min(self.n_cluster,n_rules)): 
                idx = np.argmax(rcounts)
                idxs.append(idx)
                rcounts[idx] = 0
                counts +=1 

            idxs = [int(r[1:])-1 for r in rlabels[idxs]]

            res = res.split("///")[2:]
            mask = np.zeros((min(self.n_cluster,n_rules),self.signal_length))

            for i,motif in enumerate(np.array(res)[idxs]): 
                motif = motif.split("\n")
                starts = motif[2].split(":")[-1][1:]
                starts = starts.strip('][').split(', ')
                starts = np.array(starts).astype(int)
                lengths = motif[3].split(":")[-1][1:]
                lengths = lengths.strip('][').split(', ')
                lengths = np.array(lengths).astype(int)
                overlaps = np.clip(np.hstack(((starts+lengths)[:-1]-starts[1:],0)),0,np.inf).astype(int)
                for idx,wlen,ovl in zip(starts,lengths,overlaps): 
                    mask[i,idx:idx+wlen-ovl-1] = 1
        else: 
            mask = np.zeros((self.n_cluster,self.signal_length))
        
        return mask
    
#########################################################################################################################################################################
#########################################################################################################################################################################