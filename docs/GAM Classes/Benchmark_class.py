"""
A child of GAM which contains a set of functions relating to the calculation of
benchmarks of Geographic Adjaceny measure.
"""
from warnings import warn
import numpy as np
import pandas as pd
import geopandas as gpd

from GAM_class import GAM
from Stability_class import Stability

class Benchmark(GAM):
    def __init__(self, df):
      """ Input:
            df: a Geodataframe with a column of shapely geometries.""" 
      super().__init__(df)
      self.S = None

    def get_constants(self):
        """ Gets the constants S and G for the benchmark of GAM."""
        # number of nodes
        N = self.N

        # gets neighbors
        if self.neighs is None:
            self.no_neighs()

        # get vector of 1/n for graph
        recip_n = np.array(1/self.gdf['numneigh'])

        # remove nodes with no neighbors
        recip_n[recip_n == np.inf] = 0

        # adjust N
        N = N - np.count_nonzero(recip_n == np.inf) 

        # constant S
        self.S = np.sum(recip_n)/N

        # constant G
        self.G = np.sum(np.outer(recip_n, recip_n))/N**2

    def get_mu(self, k):
        """ Calculates the expected GAM of a random clustering for k an array of
        number of communities.
        """

        # gets constants if not already found
        if self.S is None:
            self.get_constants()

        recip_k_2 = (1/k)**2

        # calculates mu
        self.mu = recip_k_2*((k - 1)*self.S + 1)

    def get_sigma(self, k):
        """ Calculates the standard deviation of GAM of a random clustering for 
        k an array of the number of communities.
        """

        # gets constants if not already found
        if self.S is None:
            self.get_constants()

        # calculates k dependent constant
        c_k = (k - 1)**2/k**4

        # creates graph based constant
        GS = self.G - self.S**2

        if GS < 0:
          warn('Possible floating point arithemetic error.')

        # calculates sigma
        self.sigma = np.sqrt(c_k*GS)
     
    # Note: Functions below are all related to testing the validatity of the
    # benchmark calculations with randomdly sampled clusterings.  

    def random_df(self,file_name, locs):
        """ Creates a cluster_df (see Stability) but where the clusters are 
        drawn uniformly at random and are of the same number of communities of 
        their correspondong Markov Time.
            Input:
                file_name: The name of the MATLAB file to be read.
                locs: An array/list of indexes relating to corresponding 
                      Markov times in t for which we'd like add cluster 
                      information for to the df. Or a string 'all' which means 
                      all labels are added (and hence times in t).
              Output:
                    ran_df: A df with columns of randomly sampled clusterings 
                    of size matching k
        """

        df = self.gdf.copy()
        
        # loads results of Markov Stability Community Detection
        stability = Stability(file_name)

        # get stability C and k
        stab_C = stability.C
        stab_k = stability.k

        # initialises ran_C
        ran_C = np.zeros_like(stab_C)

        # generates random clusters
        for i, k in enumerate(stab_k):
            ran_C[:,i] = np.random.randint(0, k, self.N)

        # sets C to ran_C
        stability.C = ran_C

        # creates df from random cluster data
        ran_df = stability.cluster_df(df, locs)

        # bug fix?
        stability.C = stab_C

        return ran_df
     
    # No longer needed can be replicated with random_df() with locs = [loc]
    def sample(self, file_name, loc):
        """ Gets  a sample random cluster for time at idx loc"""
        df = self.gdf.copy()

        stability = Stability(file_name)
        k = stability.k[loc]
        t = stability.t[loc]

        # column vector of ith cluster.
        c = np.random.randint(0, k, self.N)

        # adds a column of cluster labels.
        newDf = df.assign(label = list(c))

        return newDf


    def samples(self, file_name):
        """ Gets sample GAM scores for unique values of k found by MSCD.
            Input:
                file_name: The name of the MATLAB file to be read.
            Outputs:
                unique_k: A list of the unique values of k, the number 
                of communities found by MSCD.
                samples: A list of sample GAM scores for randoming clusters
                for each unique k.
            """

        stability = Stability(file_name)

        # get stability k
        stab_k = stability.k

        # gets unique_k locations
        unique_k, locs = np.unique(stab_k, return_index=True)

        # gets stability dataframe
        ran_df = self.random_df(file_name, locs)

        # gets sample scores
        self.gdf = ran_df
        samples = self.GAM_scores()

        return samples, unique_k
