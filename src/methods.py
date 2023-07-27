import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from neighborhood import KNN
from persistence import BasicPersistence,ThresholdPersistenceMST
from threshold import otsu_jump
from post_processing import PostProcessing

#########################################################################################################################################
#########################################################################################################################################

class BasePersistentPattern(object): 

    def __init__(self,wlen:int,n_neighbors:int, n_patterns = None,jump=1,distance_name = "LTNormalizedEuclidean" ,min_wlen = None,alpha =0.01,beta = 0,individual_bith_cut = False,similar_length = False, similarity = 0.25,n_jobs =1): 
        self.wlen = wlen
        self.n_neighbors = n_neighbors
        self.n_patterns = n_patterns
        self.jump = jump
        self.distance_name = distance_name
        if min_wlen is not None: 
            self.min_wlen = min_wlen
        else: 
            self.min_wlen = wlen
        self.alpha = alpha
        self.beta = beta
        self.individual_bith_cut = individual_bith_cut
        self.similar_length = similar_length
        self.similarity = similarity
        self.n_jobs = n_jobs

        self.p_cut_ = None
        self.b_cut_ = None

        self.post_processing_ = PostProcessing(wlen,min_wlen,similar_length,similarity)

    def _ktanh(self,X:np.ndarray,alpha=None,beta = None)->np.ndarray:
        if alpha is None: 
            alpha = self.alpha
        if beta is None: 
            beta = self.beta
        norm_factor = np.tanh(beta**2*alpha) - np.tanh(-alpha*(4-beta**2))
        dists =  np.tanh(beta**2*alpha) - np.tanh(-alpha*(X**2-beta**2))
        return 2 * np.sqrt(dists/norm_factor)
    
    
    def _base_persistence(self,signal:np.ndarray)->None: 
        self.knn_ = KNN(self.n_neighbors,self.wlen,self.distance_name,n_jobs=self.n_jobs)
        self.knn_.fit(signal)
        self.base_persistence_ = BasicPersistence()
        self.base_persistence_.fit(self.knn_.filtration_)

    def _thresholds(self)->None: 
        pers = self.get_persistence(True)
        self.p_cut_,self.b_cut_ = otsu_jump(pers[:-1,:-1],jump=self.jump)
        if self.n_patterns is not None:
            idxs = np.where(pers[:,0]<self.b_cut_)[0]
            arr = pers[idxs]
            arr = np.sort(arr[:,1]- arr[:,0])[::-1]
            self.p_cut_ = (arr[self.n_patterns-1]+arr[self.n_patterns])/2

    def _birth_cut_dct(self)->None: 
        """Compute the dictionnary of birth cut per motif
        """
        if self.individual_bith_cut:
            pers = self.get_persistence(True)
            mask = pers[:,1] - pers[:,0] > self.p_cut_
            b_cut_dct = {}
            for line in pers[mask]: 
                if line[0]<= self.b_cut_:
                    b_cut_dct[int(line[-1])] = min(line[1],self.b_cut_)
            self.b_cut_dct_ = b_cut_dct
        else:
            self.b_cut_dct_ = None

    
    def _persistence_with_thresholds(self)->None:
        """Compute persitence based on given thresholds.
        """ 
        self.tpmst_= ThresholdPersistenceMST(persistence_threshold=self.p_cut_,birth_threshold=self.b_cut_,birth_individual_threshold=self.b_cut_dct_) 
        mst = self.base_persistence_.mst_.copy()
        mst[:,-1] = self._ktanh(mst[:,-1],self.alpha,self.beta)
        self.tpmst_.fit(mst)

    def _fit_post_processing(self): 
        #get idx_lst and birth profile
        idx_lst = []
        for seed,idxs in self.tpmst_.connected_components_.items(): 
            idx_lst.append(idxs)
        mp = self.knn_.dists_[:,0].copy()
        mp = self._ktanh(mp,self.alpha,self.beta)
        self.post_processing_.fit(idx_lst,mp)

    def fit(self,signal:np.ndarray)->None: 
        self._base_persistence(signal)
        self._thresholds()
        self._birth_cut_dct()
        self._persistence_with_thresholds()

    def get_persistence(self,with_infinite_point = True)->np.ndarray: 
        pers = self.base_persistence_.get_persistence(with_infinite_point)
        if with_infinite_point:
            pers[-1,1] =0
            pers[-1,1] = np.max(pers[:,:-1])
        pers[:,:-1] = self._ktanh(pers[:,:-1],self.alpha,self.beta)
        return pers
    
    def plot_ktanh_distance(self):
        x = np.hstack((np.linspace(2,0,101)[:-1],np.linspace(0,2,100)))
        xticks_idx = np.hstack((np.arange(200)[::25],199))
        dist = self._ktanh(x,self.alpha,self.beta)

        fig,ax = plt.subplots(1,1,figsize = (5,5))
        ax.plot(x, label = "euclidean")
        ax.plot(dist, label = r"$Ktanh_{\alpha,\beta}$")
        ax.set_xticks(xticks_idx)
        ax.set_xticklabels(np.round(x[xticks_idx],1))
        ax.set_xlabel(r"$\|x-y\|_2$")
        ax.set_ylabel("distnace")
        ax.set_title(r"$\alpha:$" + f"{np.round(self.alpha,2)}   &   " + r"$\beta:$" + f"{np.round(self.beta,2)}")
        fig.tight_layout()
        plt.legend()
        plt.show()
  
        
    def plot_persistence_diagram(self): 
        pers = self.get_persistence(True)[:,:-1]
        pers = pers[pers[:,1]-pers[:,0]!=0]
        mask1 = pers[:,0]>self.b_cut_
        mask1 += (pers[:,0]<= self.b_cut_) * ((pers[:,1] - pers[:,0])<=self.p_cut_)
        mask2 = (pers[:,0]<= self.b_cut_) * ((pers[:,1] - pers[:,0])>self.p_cut_) 
        
        fig,ax = plt.subplots(1,1,figsize = (5,5))
        p0 = Polygon([[0,0],[2,0],[2,2]],color = 'grey', alpha=0.25)
        ax.add_patch(p0)
        ax.hlines(2,0,2,color="black", lw = 0.5,zorder=1)
        ax.hlines(0,0,2,color="black", lw = 0.5,zorder=1)
        ax.vlines(2,0,2,color="black", lw = 0.5,zorder=1)
        ax.vlines(0,0,2,color="black", lw = 0.5,zorder=1)
        ax.scatter(*pers[mask1].T, color = "tab:blue",zorder=2)
        ax.scatter(*pers[mask2].T, color = "tab:orange",zorder=2)
        ax.vlines(self.b_cut_,0,2,color="red",zorder=3)
        p2 = Polygon([[0,self.p_cut_],[2-self.p_cut_,2]], color = "red",zorder = 3)
        ax.add_patch(p2)
            
        fig.tight_layout()
        plt.show()
    
    def set_distance_params(self,alpha=None,beta=None): 
        if alpha is not None: 
            self.alpha = alpha
        if beta is not None: 
            self.beta = beta
        if (self.alpha is not None) * (self.beta is not None):
            self._thresholds()
            self._birth_cut_dct()
            self._persistence_with_thresholds()
        else: 
            raise ValueError("alpha or beta is None")

    def set_cut_values(self,p_cut=None,b_cut=None): 
        if p_cut is not None: 
            self.p_cut_ = p_cut
        if b_cut is not None: 
            self.b_cut_ = b_cut
        if (self.p_cut_ is not None) * (self.b_cut_ is not None): 
            self._birth_cut_dct()
            self._persistence_with_thresholds()
        else: 
            raise ValueError("p_cut or b_cut is None")
        
    @property
    def signal_(self): 
        return self.knn_.signal_
    
    @property
    def predictions_(self): 
        self._fit_post_processing()
        return self.post_processing_.predictions_
    
    @property
    def prediction_birth_list_(self):
        self._fit_post_processing()
        return self.post_processing_.prediction_birth_list_
    
    @property
    def prediction_mask_(self)->np.ndarray: 
        self._fit_post_processing()
        return self.post_processing_.prediction_mask_
    
#########################################################################################################################################
#########################################################################################################################################

