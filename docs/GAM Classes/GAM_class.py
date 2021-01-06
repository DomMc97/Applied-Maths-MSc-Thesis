"""
A set of functions relating to the Geographic Adjaceny measure.
Note:Class depends on geopandas and a variety of it's additional dependencies
which need to be downloaded in specific way 
https://github.com/DomMc97/Applied-Maths-MSc-Thesis/blob/master/docs/GAM%20Classes/Geopandas.ipynb.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import spatial

class GAM():
    def __init__(self, gdf):
        """
        Input:
            gdf:geodataframe with a column/columns of labels (unless using class
            just for benchmark calculation) and column of shapely polygon geometries.
        
        # NOTE: The class can easily edited to use a columns/columns of coordinate
        locations.
        """
        self.gdf = gdf
        self.N, self.M = gdf.shape
        self.neighs = None

    def get_neighs(self, method = 'poly', param = None):
        """ Finds the neighbors of each node using the geometry column of gdf.
        Inputs:
              method: How to determine whether nodes are neighbors.
              Options -'poly': If the polygon border associated to both the nodes 
                      touch (i.e they share a border).
                      -'k': If the a node is one of k-nearest neighbors based on euclidean distance
                      between centroids of the other node.
                      -'ball': epsilon ball method based on euclidean distance
                      between centroids.
              param: Additional argument for 'k' or 'ball'.
        Output: gdf a geodataframe with a column of neighbors for each node and another column of number of neighbors of that node.
        """

        # finds neighbors for k-nearest method
        if method == 'k':
            # finds centroids
            # NOTE: To be updated if only coordinate data is available
            centroids = self.gdf['geometry'].apply(
                lambda g:[g.centroid.x, g.centroid.y]).tolist()

            # spatially organising the points on a tree to reduce runtime
            kdtree = spatial.KDTree(centroids)

            # calculates the k nearest neighbors of each node
            _ , neighs = kdtree.query(centroids, k=param + 1)

            # remove self as neighbor
            neighs = neighs[:, 1:]

            # creates class attribute of neighbors
            self.neighs = neighs

            # adds number of neighbors column to gdf
            self.gdf['numneigh'] = param

        # finds neighbors ball method
        elif method == 'ball':
            # finds centroids
            # NOTE: To be updated if only coordinate data is available
            centroids = self.gdf['geometry'].apply(
                lambda g:[g.centroid.x,g.centroid.y]).tolist()

            # spatially organising the points on a tree to reduce runtime
            kdtree = spatial.KDTree(centroids)

            #calculates the nearest neighbors
            neighs = kdtree.query_ball_point(centroids, r=param)

            # remove self as neighbor
            for i, neigh in enumerate(neighs):
                neigh.remove(i)

            # creates attribute of neighbors
            self.neighs = neighs

            # adds number of neighbours column to gdf
            self.gdf['numneigh'] = [len(neigh) for neigh in neighs]

        #finds neighbors for poly method
        elif method =='poly':
            gdf = self.gdf
            # initialise self.neighs
            self.neighs = []
            
            # iterates through nodes(rows) and finds their neighbors
            for i, row in gdf.iterrows():
                # finds neighbors
                neigh = np.array(gdf[gdf.geometry.intersects(
                    row['geometry'])].index)
                
                # removes self intersections
                neigh = neigh[neigh != i]

                # adds number of neighbours column to gdf
                gdf.at[i, 'numneigh'] = len(neigh)

                # adds neigh to neighs
                self.neighs.append(neigh)

        else:
            print('Invalid Method')

        gdf = self.gdf # not needed?

        # get location of geometry column
        geom = gdf.columns.to_list().index('geometry')

        # gets list of columns
        cols = gdf.columns.to_list()

        # puts numneigh column to right of geometry
        cols.insert(geom + 1, cols.pop(cols.index('numneigh')))
        gdf = gdf.reindex(columns=cols)

        self.gdf = gdf


    def no_neighs(self):
        """Finds neighbors using get_neighs() if not already computed."""
        method = input("What method 'poly', 'ball' or 'k'?\n")

        if method == 'ball':
            param = float(input('What parameter?\n'))
        elif method == 'k':
            param = int(input('What parameter?\n'))
        else:
            param = None

        self.get_neighs(method, param)

    def GAM_df(self, remove=True):
        """Calculates the Geographic Adjacency Measure for a gdf with column
        of 'label' of cluster labels and creates a geodataframe with
        the information on neighbours.
        Input:
            remove: If true remove 0 neighbour nodes
        Output:
          score: Geographic Adjacency Measure score.
          df: An adjsuted geodataframe with an additional columns of common
          neighbours and 'gi' = common neighbors/number neighbors
        """
        # gdf to find GAM for 
        df = self.gdf
        
        # number of nodes to be summed 
        sumN = self.N

        # list of cases where common neighbors = 0
        invalid = []

        # computes neighbors if not done already
        if self.neighs is None:
            self.no_neighs()
        
        # finds the proportion of common neighbors for each node
        for i in range(sumN):
            # gets label at index i
            ilabel = df.at[i,'label']

            # gets neighbors of node i
            neighbors = self.neighs[i]

            # gets labels of neighbors of i
            nlabels = df[df.index.isin(neighbors)]['label'].tolist()

            # number of neighbors in same cluster
            com_neigh = np.count_nonzero(nlabels == ilabel)

            # number of neighbors
            num_neigh = df.at[i, 'numneigh']

            # calculates gi and adds info to df
            if num_neigh == 0:
                # adds node to list of nodes with no neighbors 
                invalid.append(df.at[i, 'msoa11nm'])
                # sets gi = 0
                gi = 0
                #
                sumN -= 1
            else:
                gi = com_neigh/num_neigh

            df.at[i, 'comneigh'] = com_neigh
            df.at[i, 'gi'] = gi

        score = sum(df["gi"]**2)/sumN

        if remove:
          for node in invalid:
              df = df[df['msoa11nm'] != node]

        self.GAMdf = df
        return df, score

    def GAM_scores(self):
        """Calculates the Geographic Adjacency Measure for a each column of
        labels of gdf.
        Output:
               scores: Array of Geographic Adjacency Measure score.
          """

        # computes neighbors if not done already.
        if self.neighs is None:
            self.no_neighs()

        df = self.gdf.copy()

        # get size
        N, M = df.shape

        # index of first column of labels
        start = df.columns.to_list().index('numneigh') + 1

        # access clusters labels
        Cdf = df.iloc[:,start:M]

        # initialises c a list of df which will make a matrix of num commneigh
        c = []
        for i in range(N):

            #gets neighbors
            neighbors = self.neighs[i]

            #get array of labels of i
            ilabels = Cdf.iloc[i, :]

            # condenses Cdf to just labels of neighbors of i
            Ndf = Cdf[df.index.isin(neighbors)]

            # common neighbors for i
            c_i = Ndf.eq(ilabels,1).sum()

            #add to matrix
            c.append(c_i)

        # convert c to a matrix in the form of a dataframe
        c = pd.concat(c, axis=1).T

        # remove data for cases with no neighbors
        c = c[df['numneigh'] != 0]
        df = df[df['numneigh'] != 0]

        # get sumN
        sumN, _ = df.shape

        # calculates g a matrix of gi's
        g = np.array(c)/(np.array(df['numneigh'])[:,None])

        #calculate gam scores by squaring and computing row sums
        scores = np.sum(g**2,axis = 0)/sumN

        return scores

    def Aggregate(self,df = None):
      """Takes a GAMdf and aggregates it based on cluster.
        Input: df of self.GAMdf type. Or None if already set.
        Output: Aggregated dataframe
      """

      #assigns df
      if df is None:
          df = self.GAMdf

      #aggreagates cluster data
      aggdf = df.copy()
      aggdf = aggdf.drop(['msoa11nm','numneigh','comneigh'],axis=1)

      # calculates gi^2 for GAM
      aggdf['gi'] = aggdf['gi']**2

      #for count on aggregation
      aggdf['num_MSOAs'] = 1

      #performs aggregation
      aggdf = aggdf.dissolve(by='label', aggfunc='sum')
      aggdf.reset_index(level=0, inplace=True)

      #adds cluster GAM column
      aggdf['GAM'] = aggdf['gi']/aggdf['num_MSOAs']

      #remove gi column
      aggdf = aggdf.drop(['gi'],axis=1)

      #renames column
      aggdf.rename(columns={'con_trust':'num_trusts'}, inplace=True)

      return aggdf
