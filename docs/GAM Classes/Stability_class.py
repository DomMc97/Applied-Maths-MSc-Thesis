"""
A class relating to the loading of Markov Stability Community Detection data 
from MATLAB and its merging with a DataFrame of geographical data.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat

class Stability:

    def __init__(self, file_name, remove=True):
        """ Initialises the Stability class by loading a MATLAB file of 
        Markov Stability Community Detection data and assigns attributes.
        Inputs:
            file_name: The name of the MATLAB file to be read.
            remove: Boolean asking whether the Isles of Scilly clustering data 
            should be removed.
        """
        # loads data
        data = loadmat(file_name)
        # N x T array of cluster labels.
        self.C = data['C']
        # remove Isles of Scilly from cluster information.
        if remove:
            self.C = np.delete(self.C,6639,0)
        # Array of the number of communities at each Markov time.
        self.k = data['N'][0]
        # Array of Markov times.
        self.t = data['Time'][0]
        # Array of Variation of Information at each Markov time.
        self.VI = data['VI'][0]
        # Array of the Stability score at each Markov time.
        self.S = data['S'][0]

    def cluster_df(self, df, locs):
        """ Takes an existing DataFrame which contains auxillary information on 
        each node, including its underlying Geography, and adds a column\columns 
        containing information regarding which cluster label each node has at a 
        given Markov time/times.
        Inputs:
              df: Dataframe to add clustering labels too.
              locs: An array/list of indexes relating to corresponding Markov 
              times in t for which we'd like add cluster information for to the 
              df. Or a string 'all' which means all labels are added (and hence 
              times in t).
        Outputs:
                newDf: The new dataframe.

                Returned if only 1 location i is provided.
                n: Number of communities for cluster i.
                t: Markov time of cluster i.
        """
        df = df.copy() # not needed?

        # if a single int is inputed for locs covert to a list of one element
        if type(locs) == int:
            locs = [locs]

        # if locs is a string this will be an Inavlid input or 'all'
        if type(locs) == str:
            # 'all' Markov times
            if locs == 'all':
                    # converts cluster label matrix to dataframe
                    Cdf = pd.DataFrame(self.C)
                    # sets header as time
                    Cdf.columns = ['{:e}'.format(t) for t in self.t]
                    # adds cluster labels to the existing df
                    newDf = df.join(Cdf.copy())

                    return newDf
            else:
                print('Invalid string location.')

        # case of a single time (and hence column) being added to df
        elif len(locs) == 1:
            # number of clusters at loc i
            n = self.k[locs[0]]
            # time at loc i
            t = self.t[locs[0]]
            # column vector of ith clustering.
            c = self.C[:,locs[0]]

            # adds a column of cluster label.
            newDf = df.assign(label = list(c))

            return newDf, n ,t

        # case of a subset of Markov times being added to df
        # Note this else can replicate the if when loc = [0:N]
        # and elif when loc = [i] apart from the returning of n, t
        else:
            # slices C
            myC = self.C[:, locs]
            # converts cluster label matrix to dataframe
            Cdf = pd.DataFrame(myC)
            # sets header as time
            Cdf.columns = ['{:e}'.format(t) for t in self.t[locs]] 
            newDf = df.join(Cdf.copy())

            return newDf
