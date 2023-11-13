<h1 align="center">Persistence-based Motif Discovery in Time Series</h1>

<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/thibaut-germain/Persistent-Pattern-Discovery"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/thibaut-germain/Persistent-Pattern-Discovery">
</p>
</div>

## Abstract
Motif Discovery consists in finding repeated patterns and locating their occurrences within time series without prior knowledge of their shape or location. Most state-of-the-art algorithms rely on three core parameters: the number of motifs to discover, the length of the motifs, and a similarity threshold between motif occurrences. In practice, these parameters are difficult to determine and are usually set by trial-and-error.

In this paper, we propose a new algorithm that discovers motifs of variable length without requiring a similarity threshold. At its core, the algorithm maps a time series onto a graph, summarizes it with persistent homology - a tool from topological data analysis - and identifies the most relevant motifs from the graph summary. We also present an adaptive version of the algorithm that infers the number of motifs to discover from the graph summary. Empirical evaluation on 9 labeled datasets, including 6 real-world datasets, shows that our method significantly outperforms state-of-the-art algorithms.

<p align="center">
<img src="illustrative_example.pdf" alt="drawing" width="400"/>
</p>


## Functionalities


## Prerequisites

1.  download and unzip the datasets at the root folder from the following archive:

```(bash) 
https://drive.google.com/file/d/1tfOXKbk7rhAqF4jzuMkrgklcYU3qtWzY/view?usp=sharing
```
2. All python packages needed are listed in [requirements.txt](https://github.com/thibaut-germain/Persistent-Pattern-Discovery/requirements.txt) file and can be installed simply using the pip command: 

```(bash) 
conda create --name perspa --file requirements.txt
``` 



## Reference

If you use this work, please cite:

