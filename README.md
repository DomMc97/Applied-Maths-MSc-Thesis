# Applied-Maths-MSc-Thesis
Files created for the purpose of Dominic McEwen's Applied Mathematics Msc Thesis, Imperial College London.

## Abstract

Unsupervised clustering is an important area of study which deals with the grouping of collections of similar datapoints from within a dataset. Unsupervised clustering has many applications; a recently proposed adoption by Clarke 2019 highlights how unsupervised clustering can be employed to discover hospital catchment areas in England. This study will provide an overview and refinement of Clarke's methodologies and will introduce a new heuristic to evaluate clusterings of data with an embedded geography.

The overview includes discussions on both classic and newly proposed graph construction techniques. The overview also includes a summary of the unsupervised clustering process Markov Stability Community Detection (MSCD), a multiscale method which can be seen as a 'zooming lens' from fine to coarse scales. 

The Geographical Adjacency Measure (GAM) is introduced to evaluate the clusters returned from community detection methods in a geographical setting. GAM is constructed to score the spatial adjacency of clusters. It is shown that the proposed GAM score is dependent on the geographic configuration of cluster boundaries. The dependence on boundaries will be taken leverage of, with it being shown that GAM can be used to refine the boundaries of clusters in order to improve clustering performance. GAM will also be displayed as an effective way to choose the most suitable scale, from a geographic perspective, from those found by MSCD.

Full thesis available on request.
