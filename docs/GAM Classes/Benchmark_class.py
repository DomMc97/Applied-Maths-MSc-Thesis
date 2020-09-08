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
    """Calculates benchmark scores and inherites GAM where gdf is in Benchmark
    form"""

    def __init__(self, df):
      super().__init__(df)
      self.S = None

    def get_constants(self):
        """Gets the constants S and G for the benchmark of GAM"""
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
        self.G = np.sum(np.outer(recip_n,recip_n))/N**2

    def get_mu(self, k):
        """
        Calculates the expected GAM of a random clustering for k an array of
        number of communities.
        """

        # gets constants if not already found
        if self.S is None:
            self.get_constants()

        recip_k_2 = (1/k)**2

        # calculates mu
        self.mu = recip_k_2*((k - 1)*self.S + 1)

    def get_sigma(self, k):
        """
        Calculates the standard deviation of GAM of a random clustering for k
        an array of number of communities.
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


    def random_df(self,file_name, locs, style = 'index'):
        """Creates cluster df See Stability but where the clusters are
        randomised"""

        df = self.gdf.copy()

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
        ran_df = stability.cluster_df(df, locs, style)

        # bug fix?
        stability.C = stab_C

        return ran_df

    def sample(self,file_name, loc):
        """ Gets  a sample random cluster for time at idx loc"""
        df = self.gdf.copy()

        stability = Stability(file_name)
        k = stability.k[loc]
        t = stability.t[loc]

        # column vector of ith cluster.
        c = np.random.randint(0, k, self.N)

        # adds a column of cluster label.
        newDf = df.assign(label = list(c))

        return newDf


    def samples(self,file_name, style = 'index'):
        """ Gets sample GAM scores for unique values of k and returns samples
        and unique k"""

        stability = Stability(file_name)

        # get stability k
        stab_k = stability.k

        # gets unique_k locations
        unique_k, locs = np.unique(stab_k, return_index=True)

        # gets stability dataframe
        ran_df = self.random_df(file_name, locs, style = 'index')

        # gets sample scores
        self.gdf = ran_df
        samples = self.GAM_scores()

        return samples, unique_k
