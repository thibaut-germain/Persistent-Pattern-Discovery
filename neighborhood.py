
"""""
Inpired from paper: 

Zhu, Y., Zimmerman, Z., Senobari, N. S., Yeh, C. C. M.,Funning, G., Mueen, A., ... & Keogh, E. (2016, December). 
Matrix profile ii: Exploiting a novel algorithm and gpus to break the one hundred million barrier for time series motifs 
and joins. In 2016 IEEE 16th international conference on data mining (ICDM) (pp. 739-748). IEEE.

"""""

import numpy as np 
from joblib import Parallel,delayed
import distance as distance

class KNN(object): #verified

    def __init__(self,n_neighbors:int,wlen:int,distance_name:str,distance_params = dict(),n_jobs = 1) -> None:
        """KNN initialization

        Args:
            n_neighbors (int): Number of neighbors
            wlen (int): Window length
            distance_name (str): name of the distance
            distance_params (_type_, optional): additional distance parameters. Defaults to dict().
            n_jobs (int, optional): number of processes. Defaults to 1.
        """
        self.n_neighbors = n_neighbors
        self.wlen = wlen
        self.distance_name = distance_name
        self.distance_params = distance_params
        self.n_jobs = n_jobs

    def _search_neighbors(self,idx:int,line:np.ndarray)-> tuple: 
        """Find index and distance value of the non overlapping K nearest neighbors

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
        for i in range(self.n_neighbors): 
            try: 
                #local next neighbor
                t_idx = np.argmin(line)
                neighbors.append(idxs[t_idx])
                dists.append(line[t_idx])

                #remove window
                remove_idx = np.arange(max(0,t_idx-self.wlen+1),min(len(line),t_idx+self.wlen))
                idxs = np.delete(idxs,remove_idx)
                line = np.delete(line,remove_idx)
            except: 
                #if no neighbor append with nan
                neighbors.append(-1)
                dists.append(np.nan)
            
        return neighbors,dists

    def _elementary_knn(self,start:int,end:int)->tuple:
        """Find k nearest neighbors of a chunk of successive lines of the crossdistance matrix

        Args:
            start (int): chunk start
            end (int): chunck end

        Returns:
            tuple: neighbor index np.ndarray, neighbor distance np.ndarray
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
        

    def fit(self,signal:np.ndarray)->None: 
        """Find the index and distance of the nearest neighbors. 
        Attention, if there are less neighbors than k, the index array is filled with -1 and the distance array with np.nan

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        #initialisation
        self.signal_ = signal
        self.mdim_ = len(signal)-self.wlen+1 
        self.distance_ = getattr(distance,self.distance_name)(self.wlen,**self.distance_params)
        self.distance_.fit(signal)

        #divide the signal accordingly to the number of jobs
        set_idxs = np.linspace(0,self.mdim_,self.n_jobs+1,dtype=int)
        set_idxs = np.vstack((set_idxs[:-1],set_idxs[1:])).T

        # set the parrallel computation
        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._elementary_knn)(*set_idx)for set_idx in set_idxs
        )

        idxs,dists = list(zip(*results))
        self.idxs_ = np.concatenate(idxs,dtype =int)
        self.dists_ = np.concatenate(dists,dtype = float)
        return self

    @property
    def filtration_(self)->np.ndarray: 
        """Compute the filtration for homology persistence.

        Returns:
            np.ndarray: filtration
        """
        lst = []
        #similarity connection
        for i,(idxs,dists) in enumerate(zip(self.idxs_,self.dists_)): 
            lst.append(np.c_[np.full_like(idxs,i),idxs,dists])
        arr = np.concatenate(lst)
        #remove not existing similarity connextion and duplicates
        arr = arr[~np.isnan(arr).any(axis=1),:] 
        arr[:,:2] = np.sort(arr[:,:2],axis=1)
        arr = np.unique(arr,axis=0)

        #time connexion
        min_dist = self.dists_[:,0]
        max_dist = np.max(np.c_[min_dist[1:],min_dist[:-1]],axis =1)
        size = min_dist.shape[0]-1
        arr = np.r_[arr,np.c_[np.arange(size,dtype=int),np.arange(1,size+1,dtype=int),max_dist]]
        arr = arr[1:][np.argsort(arr[1:,-1])]
        return arr
