import numpy as np

def transfrom_label(L:np.ndarray)->list: 
    """Transfom binary mask to a list of start and ends

    Args:
        L (np.ndarray): binary mask, shape (n_label,length_time_series)

    Returns:
        list: start and end list. 
    """
    lst = []
    for line in L: 
        line = np.hstack(([0],line,[0]))
        diff = np.diff(line)
        start = np.where(diff==1)[0]+1
        end = np.where(diff==-1)[0]
        lst.append(np.array(list(zip(start,end))))
    return np.array(lst,dtype=object)