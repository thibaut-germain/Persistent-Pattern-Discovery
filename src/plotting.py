import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import correlation_lags
from utils import transfrom_label

def pearson_correlation(s0,s1,wlen): 
    m = s0.shape[0]
    n = s1.shape[0]
    avg_coeff = np.convolve(np.ones(m),np.ones(n))
    mask = avg_coeff >= wlen
    dot_prod = np.convolve(s0,s1[::-1])/avg_coeff
    mean0 = np.convolve(s0,np.ones(n))/avg_coeff
    mean1 = np.convolve(np.ones(m),s1[::-1])/avg_coeff
    std0 = np.sqrt((np.convolve(s0**2,np.ones(n))-avg_coeff*mean0**2)/avg_coeff)
    std1 = np.sqrt((np.convolve(np.ones(m),s1[::-1]**2)-avg_coeff*mean1**2)/avg_coeff)
    pearson = (dot_prod - mean0*mean1)[mask]/(std0*std1)[mask]
    offset = correlation_lags(m,n)[mask]
    return pearson,offset

def get_optimal_lag(s0,s1,wlen): 
    corr,offsets = pearson_correlation(s0,s1,wlen)
    return offsets[np.argmax(corr)]

def get_optimal_lag_matrix(dataset,wlen=None): 
    n_ts = len(dataset)
    lags = np.zeros((n_ts,n_ts))
    if wlen is None: 
        wlen = np.mean([len(ts) for ts in dataset])//4
    for i,j in np.vstack(np.triu_indices(n_ts,1)).T: 
        lag = get_optimal_lag(dataset[i],dataset[j],wlen)
        lags[i,j] = lag
        lags[j,i] = -lag
    return lags

def get_relative_lag(dataset,wlen=None): 
    lags = get_optimal_lag_matrix(dataset,wlen)
    best_idx = np.argmin(np.sum(np.abs(lags),axis=0))
    return lags[best_idx]

def get_barycenter(dataset,lags): 
    lags = lags.copy()
    min_lag = np.min(lags)
    lags -= min_lag
    length = np.max(np.array([len(ts) for ts in dataset])+lags).astype(int)
    arr = np.zeros((len(dataset),length))
    for i,(lag,ts) in enumerate(zip(lags.astype(int),dataset)):
        arr[i, lag : lag+len(ts)] = ts
    avg_pattern = np.mean(arr,axis=0)
    x = np.arange(min_lag,min_lag + len(avg_pattern))
    return x,avg_pattern

def plot_pattern(signal,label_lst,wlen =None, align = True, scale = False, barycenter = False, sharex = True, sharey = True):
    n_motif = len(label_lst)
    fig,axs = plt.subplots(n_motif,1,sharex=sharex,sharey=sharey) 
    #cmap = plt.cm.tab10

    for i,lst in enumerate(label_lst): 
        dataset = []
        for start,end in lst: 
            dataset.append(signal[start:end])
        if scale: 
            dataset = [(ts - np.mean(ts)/np.std(ts)) for ts in dataset]
        if align: 
            lags = get_relative_lag(dataset,wlen)
            
        for j,ts in enumerate(dataset): 
            if align: 
                x = np.arange(lags[j],lags[j]+ len(ts))
            else: 
                x = np.arange(len(ts))
            if n_motif>1:
                axs[i].plot(x,ts, color = "black",alpha = 0.1)
                axs[i].title.set_text(f"Pattern {i}")
            else: 
                axs.plot(x,ts, color = "black", alpha = 0.1)
                axs.title.set_text(f"Pattern {i}")

        if align*barycenter: 
            x,avg_pattern = get_barycenter(dataset,lags)
            if n_motif > 1: 
                axs[i].plot(x,avg_pattern, color = "red")
            else: 
                axs.plot(x,avg_pattern, color = "red")     
    fig.tight_layout()
    return fig,axs

def plot_signal_pattern(signal,label_lst,birth_lst,display_start=True,birth_aware=True,min_alpha =0.2):
    fig,ax = plt.subplots(1,1,figsize =(20,5))
    ax.plot(signal, color = "black", alpha = 0.1)
    if len(label_lst) ==1: 
        cmap = lambda x : "red"
    else: 
        cmap = plt.cm.tab10
    births = np.hstack(birth_lst)
    min_birth = np.min(births)
    max_birth = np.max(births)
    if birth_aware * (max_birth != min_birth):
        alpha = lambda x : 1 - (1-min_alpha) * ((x - min_birth)/(max_birth-min_birth))
    else: 
        alpha = lambda x : 1
    for i,(labels, births) in enumerate(zip(label_lst,birth_lst)): 
        for (start,end), birth in zip(labels,births): 
            x = np.arange(start,end)
            y = signal[start:end]
            ax.plot(x,y,color = cmap(i%10), alpha = alpha(birth), label=f"Pattern {i}")

    if display_start: 
        starts = np.hstack([arr[:,0] for arr in label_lst])
        for start in starts:
            plt.axvline(x= start, color = "black", ls="--",lw=1)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    return fig,ax