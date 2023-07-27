### Heuristics to compute persitence and birth threshold ### 

import numpy as np
from scipy.stats import norm

def _infinite_point_treatement(persistence:np.ndarray)->np.ndarray:
    """Change death value of infinite point to the maximum death observed

    Args:
        persistence (np.ndarray): persisence attribute of a persistence class

    Returns:
        np.ndarray: transfromed persistence
    """
    mask = persistence[:,1] == np.inf
    if np.any(mask):
        max_death = np.max(persistence[np.invert(mask),1])
        t_persistence = persistence.copy()
        t_persistence[mask,1] = max_death
        return t_persistence
    else: 
        return persistence



def _jump_cut(vector : np.ndarray, offset = 1):
    """set the cut to the maximum jump

    Args:
        vector (np.ndarray): _description_
    """
    arr = np.sort(vector)
    thresholds = np.diff(arr)
    threshold_idx = np.argmax(thresholds)+offset
    return arr[threshold_idx]


def _basic_otsu(vector: np.ndarray,nbins=1024): 
    """Compute Otsu threshold

    Args:
        vector (np.ndarray): pixel array. shape: (N,)
        nbins (int, optional): number of bins for the histogram. Defaults to 1024.
    """

    #intialisation
    count, values = np.histogram(vector,nbins)
    count = count.astype(float)/vector.size
    wl = 0
    wr = 1
    ml = 0
    mr = np.mean(vector)
    lst =[]
    #loop
    for prob,value in zip(count[:-1],values[:-1]): 
        ml = (ml*wl + prob*value)/(wl + prob)
        mr = (mr*wr - prob*value)/(wr - prob)
        wl += prob
        wr -= prob
        var = wl*wr*(ml-mr)**2
        lst.append(var)

    return values[np.argmax(lst)]

def otsu(persistence:np.ndarray,kind = None,nbins=1024)-> tuple:
    """Compute Otsu threshold individualy for persistence and birth.

    Args:
        persistence (np.ndarray): persisence attribute of a persistence class
        kind (str, optional): Take skewedness into account. option: ['skewed', 'birth_skewed', 'prominence_skewed']. Defaults to None

    Returns:
        tuple: (persistence_cut, birth_cut)
    """
    t_persistence = _infinite_point_treatement(persistence)
    prominence = np.diff(t_persistence,axis =1).flatten()
    mask = prominence>0
    prominence = prominence[mask]
    birth = t_persistence[mask,0]
    prominence_cut = _basic_otsu(prominence,nbins)
    birth_cut = _basic_otsu(birth,nbins)
    if kind == 'birth_skewed':
        birth_cut = _basic_otsu(birth[birth<birth_cut],nbins)
    elif kind == 'prominence_skewed':
        prominence_cut = _basic_otsu(prominence[prominence>prominence_cut])
    elif kind == 'skewed': 
        birth_cut = _basic_otsu(birth[birth<birth_cut],nbins)
        prominence_cut = _basic_otsu(prominence[prominence>prominence_cut])
    return prominence_cut,birth_cut

def mixed_otsu(persistence:np.ndarray,nbins=1024)-> tuple:
    """Compute Otsu threshold individualy for persistence and birth.

    Args:
        persistence (np.ndarray): persisence attribute of a persistence class

    Returns:
        tuple: (persistence_cut, birth_cut)
    """
    t_persistence = _infinite_point_treatement(persistence)
    prominence = np.diff(t_persistence,axis =1).flatten()
    mask = prominence>0
    prominence = prominence[mask]
    birth = t_persistence[mask,0]
    birth_cut = _basic_otsu(birth,nbins)
    birth_cut = _basic_otsu(birth[birth<birth_cut],nbins)
    prominence = prominence[birth<birth_cut]
    prominence_cut = _basic_otsu(prominence,nbins)
    prominence_cut = _basic_otsu(prominence[prominence>prominence_cut],nbins)
    return prominence_cut,birth_cut


def _otsu_2d(x:np.ndarray,y:np.ndarray,nbins=1024)->tuple: 
    """Computes otsu threshold based on two dimension

    Args:
        x (np.ndarray): first dimension, shape: (N,)
        y (np.ndarray): second dimension, shape: (N,)
        nbins (int, optional):  number of bins for the 2d histogram. Defaults to 1024.

    Returns:
        tuple: threshold along x axis, threshold along y axis
    """
    arr,u,v= np.histogram2d(x,y,bins=nbins)
    #weight
    prob = arr/np.sum(arr)
    weight = np.cumsum(prob,axis=0)
    weight = np.cumsum(weight,axis=1)

    # birth matrix
    m_x = u[:-1].reshape(-1,1)*prob
    m_x = np.cumsum(m_x,axis=0)
    m_x = np.cumsum(m_x,axis=1)
    mean_x = m_x[-1,-1]

    # prominence matrix
    m_y = v[:-1]*prob
    m_y = np.cumsum(m_y, axis=0)
    m_y = np.cumsum(m_y, axis=1)
    mean_y = m_y[-1,-1]

    # variance computation
    a = (m_x - mean_x*weight)**2 + (m_y - mean_y*weight)**2
    b = weight*(1-weight)
    var = np.divide(a, b, out=np.zeros_like(a), where=b!=0)

    # Identify threshold cut
    idx_x,idx_y = np.unravel_index(var.argmax(), var.shape)

    return u[idx_x],v[idx_y]


def otsu_2d(persistence: np.ndarray,kind=None,nbins=1024)->tuple: 
    """computes otsu 2d thresholds based on persistence and birth dimension

    Args:
        persistence (np.ndarray): persistence array from a persistence module
        nbins (int, optional): number of bins for the histogram. Defaults to 1024.

    Returns:
        tuple: Persistence threshold, birth threshold
    """
    t_persistence = _infinite_point_treatement(persistence)
    prominence = np.diff(t_persistence,axis =1).flatten()
    mask = prominence>0
    prominence = prominence[mask]
    birth = t_persistence[mask,0]
    if kind == 'skewed':
        t1,t2 = _otsu_2d(prominence,birth,nbins)
        mask = (prominence<t1)*(prominence<t2)
        prominence = prominence[mask]
        birth = birth[mask]
    return _otsu_2d(prominence,birth,nbins)

def mad_outliers(X :np.ndarray,threshold = 3.0)->tuple:
    """Compute Median Absolute Deviation Outliers

    Args:
        X (np.ndarray): persistance array from a persistance module.
        threshold (float, optional): Threshold for cutting (3 conservative, 2.5 moderate, 2 low). Defaults to 3.

    Returns:
        tuple: outlier mask
    """
    devs =  np.abs(X-np.median(X))
    mad = np.median(devs)
    cst = norm.ppf(0.75)
    return cst*devs/mad >= threshold

def normal_outliers(X:np.ndarray,threshold=0.95)->tuple:
    """Compute Outliers based on normal distribution

    Args:
        X (np.ndarray): persistance array from a persistance module.
        threshold (float, optional): Threshold for cutting. Defaults to 0.95

    Returns:
        tuple: outlier mask
    """ 
    return (X-np.mean(X))/np.std(X) > norm.ppf(threshold)

def outlier_otsu(X:np.ndarray,threshold=0.99,nbins=1024)->tuple:
    """Compute persistance cut and birth cut.

    Args:
        X (np.ndarray): persistance array from fresistence module. 
        threshold (float, optional): threshold for outlier. Defaults to 0.95.
        nbins (int, optional): number of bins for otsu. Defaults to 1024.

    Returns:
        tuple: persistance cut, birth cut.
    """
    births = X[:,0]
    b_cut = _basic_otsu(births,nbins)
    #b_cut = _basic_otsu(births[births<b_cut],nbins)
    pers = np.diff(X[X[:,0]<b_cut],axis=1).reshape(-1)
    pers = pers[pers>0]
    pers.sort()
    idx = np.where(normal_outliers(pers,threshold))[0][0]
    p_cut = (pers[idx-1]+pers[idx])/2
    return p_cut,b_cut

def otsu_jump(X:np.ndarray,jump=1,nbins=1024)->tuple:
    """Compute persistance cut and birth cut.

    Args:
        X (np.ndarray): persistance array from fresistence module. 
        threshold (float, optional): threshold for outlier. Defaults to 0.95.
        nbins (int, optional): number of bins for otsu. Defaults to 1024.

    Returns:
        tuple: persistance cut, birth cut.
    """
    births = X[:,0]
    b_cut = _basic_otsu(births,nbins)
    #b_cut = _basic_otsu(births[births<b_cut],nbins)
    pers = np.diff(X[X[:,0]<b_cut],axis=1).reshape(-1)
    pers = pers[pers>0]
    pers = np.sort(pers)[::-1]
    diff = pers[:-1] - pers[1:]
    for _ in range(jump):
        idx = np.argmax(diff)
        diff[:idx+1] = -np.inf
    p_cut = (pers[idx]+pers[idx+1])/2
    return p_cut,b_cut

    

