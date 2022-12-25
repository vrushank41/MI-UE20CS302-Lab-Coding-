#Name : SUNDEEP A
#SRN : PES1UG20CS445
#Roll No : 48

import numpy as np
class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """
    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self
    
    

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        list=[]
        for i in range(len(data)):
            for j in range(len(self.centroids)):
                difference = self.centroids[j]-data[i]
                list.append(np.linalg.norm(difference)) 
        
        cluster=[]
        resize = np.reshape(list,(len(data),len(self.centroids)))

        for i in range(len(resize)):
            min = np.argmin(resize[i])
            cluster.append(min) 

        return cluster

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
        Change self.centroids
        """
        cnt = {i:0 for i in cluster_assgn}
        avg = {key:[] for key, value in cnt.items()}
        for i, j in enumerate(data.tolist()):
            c = cluster_assgn[i]
            cnt[c] += 1
            total = avg[c]
            if len(total) == 0:
                avg[c] = j
            else:
                avg[c] = [total[i] + j[i] for i in range(len(j))]
        
        sorted = list(set(cluster_assgn))
        sorted.sort()
        centroids = []

        for i in sorted:
            count = cnt[i]
            sum_cluster = avg[i]
            centroids.append([i / count for i in sum_cluster])

        self.centroids = np.array(centroids)

    def evaluate(self, data):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
        Returns:
            metric : (float.)
        """
        dist =[]
        for i in data:
            for j in self.centroids:
                dist.append(np.square(j-i))

        dist =np.sum(dist,axis=1)
        solution=0

        for i in dist:
            solution+=i
        return solution