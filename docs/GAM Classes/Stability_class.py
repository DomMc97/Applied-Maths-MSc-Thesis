"""
A set of functions relating to the loading and compililing of Markov Stability
data from MatLab.
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat

class Stability:

    def __init__(self, file_name,remove=True):
        """
        Initialises Stability by loading MatLab file of stability data and
        assigns attributes.
        """

        data = loadmat(file_name)
        # N x T array of cluster labels.
        self.C = data['C']
        # remove Isles of Scilly from cluster information.
        if remove:
            self.C = np.delete(self.C,6639,0)
        # Array of number of communities.
        self.k = data['N'][0]
        # Array of Markov times.
        self.t = data['Time'][0]
        # Array of Variation of Information.
        self.VI = data['VI'][0]
        # Array of Stability
        self.S = data['S'][0]

    def cluster_df(self, df, locs, style = 'index'):
        """ Adds columns to an existing dataframe with information
        Inputs:
              df: Dataframe to add clustering labels too.
              locs: Array of indexes or times to find cluster information for.
              Or a string 'all' which means all labels are added.
              style: Whether 'index' or 'time' array provided.
        Outputs:
                newDf:The new dataframe.
                If only 1 location provided return.
                n: Number of communities for cluster i.
                t: Markov time of cluster i.
        """
        df = df.copy()

        if type(locs) == int:
            locs = [locs]

        if type(locs) == str:
            if locs == 'all':
                    # converts cluster label matrix to dataframe
                    Cdf = pd.DataFrame(self.C)
                    Cdf.columns = ['{:e}'.format(t) for t in self.t] # sets header as time
                    newDf = df.join(Cdf.copy())

                    return newDf
            else:
                print('Invalid string location.')

        elif len(locs) == 1:
            #if style == 'time':
            n = self.k[locs[0]]
            t = self.t[locs[0]]

            # column vector of ith cluster.
            c = self.C[:,locs[0]]

            # adds a column of cluster label.
            newDf = df.assign(label = list(c))

            return newDf, n ,t

        else:
            # slices C
            myC = self.C[:, locs]
            # converts cluster label matrix to dataframe
            Cdf = pd.DataFrame(myC)
            Cdf.columns = ['{:e}'.format(t) for t in self.t[locs]] # sets header as time
            newDf = df.join(Cdf.copy())

            return newDf
