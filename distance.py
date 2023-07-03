"""
Distance are coded to work jointly with the KNN class. They are build as an iterator class. A basic distance shoud be as follow: 
"""
import numpy as np

class Basicdistance(object): 

    def __init__(self,wlen:int,**kwargs) -> None:
        """Initialization

        Args:
            wlen (int): sliding window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray): 
        """Initialize the distance accordingly to the signal considered. 
        Compute the first line of the crossdistance matrix and the elementary elements required for reccurssivity.

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)

        """
        pass

    def first_line(self,i:int)->np.ndarray: 
        """Compute the line of the crossdistance matrix at index i 

        Args:
            i (int): line position

        Returns:
            np.ndarray: line i of the corssdistance matrix
        """
        self.first_idx_ = i
        self.idx_ = i

    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.idx_ +=1


#########################################################################################################################################
#########################################################################################################################################


class Euclidean(object): #verified

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray: 
        """Compute the squared euclidean distance of the first line of the crossdistance matrix

        Returns:
            np.ndarray: squared euclideran distance
        """
        dot_product = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')
        coeffs = np.zeros(len(self.signal_)+1)
        coeffs[1:] = np.cumsum(self.signal_**2)
        self.coeffs_ = coeffs[self.wlen:]-coeffs[:-self.wlen]
        return self.coeffs_ -2*dot_product + self.coeffs_[0]

    def first_line(self,i:int)->np.ndarray:
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """
        self.first_idx_ =i
        self.idx_ =i
        if i == 0: 
            self.basic_ = self.first_basic_.copy()
        else: 
            dot_product = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
            self.basic_ = self.coeffs_ -2*dot_product + self.coeffs_[i]
        return np.sqrt(np.clip(self.basic_,0,None))

    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.basic_[1:] = self.basic_[:-1] + \
            (self.signal_[self.idx_+self.wlen]-self.signal_[self.wlen:])**2 - \
                (self.signal_[self.idx_]-self.signal_[:-self.wlen])**2
        self.idx_ +=1 
        self.basic_[0] = self.first_basic_[self.idx_]
        return np.sqrt(np.clip(self.basic_,0,None))


#########################################################################################################################################
#########################################################################################################################################

class NormalizedEuclidean(object): #Verifed

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        dist = (self.dot_product_ - self.wlen*self.means_[i]*self.means_)/(self.wlen*self.stds_[i]*self.stds_)

        return np.sqrt(np.clip(2*self.wlen*(1-dist),0,None))

    def individual_distance(self,i,j): 
        a = (self.signal_[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b = (self.signal_[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.sqrt(np.sum((a-b)**2))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        dist = (self.dot_product_ - self.wlen*self.means_[self.idx_]*self.means_)/(self.wlen*self.stds_[self.idx_]*self.stds_)
        dist = np.sqrt(np.clip(2*self.wlen*(1-dist),0,None))
        return dist    
    
#########################################################################################################################################
#########################################################################################################################################

class UnitEuclidean(object): #Verifed

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        dist = (self.dot_product_ - self.wlen*self.means_[i]*self.means_)/(self.wlen*self.stds_[i]*self.stds_)

        return np.sqrt(np.clip(2*(1-dist),0,None))

    def individual_distance(self,i,j): 
        a = (self.signal_[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b = (self.signal_[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.sqrt(np.sum((a-b)**2))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        dist = (self.dot_product_ - self.wlen*self.means_[self.idx_]*self.means_)/(self.wlen*self.stds_[self.idx_]*self.stds_)
        dist = np.sqrt(np.clip(2*(1-dist),0,None))
        return dist    

#########################################################################################################################################
#########################################################################################################################################

class PearsonCorrelation(object): 

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        dist = 1 - (self.dot_product_/self.wlen - self.means_[i]*self.means_)/(self.stds_[i]*self.stds_)

        return dist


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        dist = 1 - (self.dot_product_/self.wlen - self.means_[self.idx_]*self.means_)/(self.stds_[self.idx_]*self.stds_)
        return dist 

#########################################################################################################################################
#########################################################################################################################################

class Cosine(object):
    
    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')


        m2 = np.zeros(len(self.signal_)+1)
        m2[1:] =np.cumsum(self.signal_**2)
        m2 = m2[self.wlen:]-m2[:-self.wlen]
        self.m2_ = np.sqrt(m2)

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        dist = 1 - (self.dot_product_)/(self.m2_[i]*self.m2_)

        return dist


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        dist = 1 - (self.dot_product_)/(self.m2_[self.idx_]*self.m2_)
        return dist 

#########################################################################################################################################
#########################################################################################################################################
class RobustNormalizedEuclidean(object): #Verifed

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        subs = np.lib.stride_tricks.sliding_window_view(self.signal_,self.wlen)
        self.sums_ = np.sum(subs,axis=1)
        self.squared_sums = np.sum(subs**2,axis=1)
        self.means_ = np.median(subs,axis=1)
        self.stds_ = np.median(np.abs(subs - self.means_.reshape(-1,1)),axis=1)

    def _compute_distance(self): 
        dist = self.squared_sums[self.idx_]*self.stds_**2 + self.sums_[self.idx_]*2*(self.stds_[self.idx_]*self.stds_*self.means_ - self.means_[self.idx_]*self.stds_**2) + \
            self.squared_sums*self.stds_[self.idx_]**2 + self.sums_*2*(self.stds_[self.idx_]*self.stds_*self.means_[self.idx_] - self.means_*self.stds_[self.idx_]**2) -\
            2*self.stds_[self.idx_]*self.stds_*self.dot_product_ +\
            self.wlen*(self.means_[self.idx_]**2*self.stds_**2 + self.means_**2*self.stds_[self.idx_]**2 - 2*self.stds_[self.idx_]*self.stds_*self.means_[self.idx_]*self.means_)

        dist /= (self.stds_[self.idx_]*self.stds_)**2
        return np.log(1+np.sqrt(np.clip(dist,0,None)))


    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        return self._compute_distance()

    def individual_distance(self,i,j): 
        a = (self.signal_[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b = (self.signal_[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.log(1+np.sqrt(np.sum((a-b)**2)))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        return self._compute_distance()  

#########################################################################################################################################
#########################################################################################################################################
class Gaussian(object): #Verifed

    def __init__(self,wlen:int,sigma =1,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen
        self.sigma = sigma

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

    def _compute_distance(self): 
        dist = (self.dot_product_ - self.wlen*self.means_[self.idx_]*self.means_)/(self.wlen*self.stds_[self.idx_]*self.stds_)
        dist = np.sqrt(np.clip(2*(1-dist),0,None))
        dist = np.exp(-dist**2/self.sigma**2)
        dist = np.sqrt(2*(1-dist))
        return dist

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        return self._compute_distance()

    def individual_distance(self,i,j): 
        a = (self.signal_[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b = (self.signal_[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.sqrt(np.sum((a-b)**2))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        return self._compute_distance()
    

#########################################################################################################################################
#########################################################################################################################################
class Tanh(object): #Verifed

    def __init__(self,wlen:int,sigma =1,offset=0,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen
        self.sigma = sigma
        self.offset = offset

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

    def _compute_distance(self): 
        dist = (self.dot_product_ - self.wlen*self.means_[self.idx_]*self.means_)/(self.wlen*self.stds_[self.idx_]*self.stds_)
        dist = np.sqrt(np.clip(2*(1-dist),0,None))
        dist  = np.tanh(-self.sigma * dist**2 + self.offset)
        dist = np.sqrt(2*(np.tanh(self.offset)-dist))
        return dist

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        return self._compute_distance()

    def individual_distance(self,i,j): 
        a = (self.signal_[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b = (self.signal_[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.sqrt(np.sum((a-b)**2))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        return self._compute_distance()

#########################################################################################################################################
#########################################################################################################################################

class LTNormalizedEuclidean(object): 

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

        self.stdt = np.sqrt((self.wlen**2 -1)/12)

        self.alphas_ = np.convolve(np.arange(self.wlen)[::-1],self.signal_,'valid')/self.wlen - self.means_*(self.wlen-1)/2
        self.alphas_ = self.alphas_/self.stdt**2
        self.etas_ = np.sqrt(self.wlen*(self.stds_**2 - self.alphas_**2 * self.stdt**2))

        

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        dist = (self.dot_product_ - self.wlen*(self.means_[i]*self.means_+self.stdt**2 * self.alphas_[i]*self.alphas_))/(self.etas_[i]*self.etas_)

        return np.sqrt(np.clip(2*(1-dist),0,None))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        dist = (self.dot_product_ - self.wlen*(self.means_[self.idx_]*self.means_+ self.stdt**2 * self.alphas_[self.idx_]*self.alphas_))/(self.etas_[self.idx_]*self.etas_)
        dist = np.sqrt(np.clip(2*(1-dist),0,None))
        return dist    
