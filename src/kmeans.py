import numpy as np
import copy
from scipy.spatial.distance import cosine
import random
import math
from data_structures import Graph, Table

#FIXME: implement elbow method for picking clusters

class kmeans:    
    def get_distance_matrix(self, vectors):
        output = np.empty(shape = (vectors.shape[0], vectors.shape[0]))
        for v_1 in range(len(vectors)):
            for v_2 in range(len(vectors)):
                output[v_1, v_2] = cosine(vectors[v_1], vectors[v_2])
        return output
    
    def get_max_num_clusters(self, vectors):
        matrix = self.get_distance_matrix(vectors)
        max = 0
        for m in matrix:
            if max < np.unique(m).shape[0]:
                max = np.unique(m).shape[0]
        return max
    
    def get_wcss(self, vectors, labels, centroids):
        within_cluster_ss = 0
        for c in range(len(centroids)):
            for v in range(len(vectors)):
                if labels[v] == c:
                    within_cluster_ss = within_cluster_ss + np.dot((vectors[v] - centroids[c]), (vectors[v] - centroids[c])) 
        return within_cluster_ss
                    
            
    def initialize_points(self, vectors, num_clusters):
        """pick the initial k random points for the cluster using kmeans"""
        # randomly select the first centroid center
        i = random.randint(0, vectors.shape[0] - 1)
        
        # append it to the centroid list
        centroids = np.array([vectors[i]], copy = True)
        
        # from 0 to self.number_of_clusters - 1
        for k in range(1, num_clusters):
        
            # for each data point, calculate cosine distance from already chosen centroids
            D = np.array([])
            for v in vectors:
                min_distance = float("inf")
                for c in centroids:
                    if min_distance > cosine(v, c):
                        min_distance = cosine(v, c)
                    
                D = np.append(D, min_distance)
            
            if np.sum(D) == 0:
                print(num_clusters)
                raise Exception("Not enough distinct vectors to form num clusters")
            
            # give each vector a probability of being chosen giving more weight to the vector with higher cosine distances
            probability = D / np.sum(D)
            
            # calculate the cummulative probability distance from probability
            cumm_prob = np.cumsum(probability)
            
            # select a random number from 0 to 1 and pick the number from cumm_prob
            r = random.random()
            i = 0
            for j, p in enumerate(cumm_prob):
                if r < p:
                    i = j
                    break
            
            # append the centroid
            centroids = np.append(centroids, [vectors[i]], axis = 0)
        
        #return initial centroids if no errors
        return centroids
    
    def cluster(self, vectors, num_clusters, max_iter = 500):
        """cluster the vectors into num_clusters clusters"""
        # initialize centroids
        centroids = self.initialize_points(vectors, num_clusters)
        pre_centroids = np.array(centroids, copy = True)
        
        # set initial labels of each cluster all to zero
        labels = np.zeros(vectors.shape[0])
        
        # repeat until max iterations reached or convergence happens
        for i in range(max_iter):
            
            # assign each vector to a cluster based off closest cosine distance
            for v in range(len(vectors)):
                min_distance = float("inf")
                for c in range(len(centroids)):
                    if min_distance > cosine(vectors[v], centroids[c]):
                        labels[v] = c
                        min_distance = cosine(vectors[v], centroids[c])
            
            # compute new average centroid averages
            for c in range(len(centroids)):
                sum = np.zeros(vectors.shape[1])
                count = 0
                for v in range(len(vectors)):
                    if labels[v] == c:
                        sum = sum + vectors[v]
                        count = count + 1
                
                # to avoid division by 0
                if count == 0:
                    return False, None, None
                    
                centroids[c] = sum / count 
            
            #if centroids don't change, break
            if np.allclose(centroids, pre_centroids):
                break
            else:
                pre_centroids = np.array(centroids, copy = True)
        
        return True, labels, centroids
    
    def train(self, vectors, num_clusters, max_iter = 500, iterations = 100):
        """perform multiple  iterations to determine the best labels and centroids"""
        best_labels = None
        best_centroids = None
        min_score = float("inf")
        
        # perform x iterations of clustering
        for i in range(iterations):
            success, labels, centroids = self.cluster(vectors, num_clusters, max_iter)
            
            # if no success
            if not success:
                continue
            # if there is only one label to pick
            else:
                wcss = self.get_wcss(vectors, labels, centroids)
                if wcss < min_score:
                    best_labels = np.array(labels, copy = True)
                    best_centroids = np.array(centroids, copy = True)
                    min_score = wcss
                
        
        return best_labels, best_centroids
    
    def pick_number_clusters(self, vectors, max_iter = 500, iterations = 10):
        """determine what is the best number of clusters using elbow method
        once the change in wcss is below a minimum threshold"""
        wcss_dict = {}
        
        max = self.get_max_num_clusters(vectors)
        # print("max", max)
        
        if max == 1:
            return max
        
        # test all values of clusters from 2 to (n - 1)
        for k in range(1, max):
        
            labels, centroids = self.train(vectors, k, max_iter, iterations)
            wcss_dict[k] = self.get_wcss(vectors, labels, centroids)
            
        # find the slope
        # best_num_clusters = 0
        # current_slope = None
        # pre_slope = None
        # for key1, key2 in zip(range(1, max - 1), range(2, max)):
            # current_slope = math.fabs(wcss_dict[key2] - wcss_dict[key1])
            # print(key1)
            # print(current_slope)
            # print(pre_slope)
            # print()
            
            # if pre_slope == None:
                # pre_slope = current_slope
                # best_num_clusters = 1
            # else:
                # if pre_slope > 1.25 * current_slope:
                    # best_slope = current_slope
                    # best_num_clusters = key1
                # pre_slope = current_slope
        
        max_slope = wcss_dict[max - 1] - wcss_dict[1]
        threshold = 0.025 * max_slope
        best_num_clusters = 1
        # print(wcss_dict)
        for key in range(1, max - 1):
            # print(str(wcss_dict[key + 1] - wcss_dict[key]))
            # print(threshold)
            # print()
            if wcss_dict[key + 1] - wcss_dict[key] > threshold:
                best_num_clusters = key + 1
                break
                
        return best_num_clusters
    
    def kmeans_merge_clustering(self, sentences, vectors, graph = None, num_clusters = None):
        if num_clusters == None:
            num_clusters = self.pick_number_clusters(vectors)
        
        if graph == None:
            graph = Graph()
            for s in sentences:
                graph.add_node(s)
                
        table = Table(sentences, vectors)
        
        parent_sentences = []
        parent_vectors = []
        
        labels, centroids = self.train(vectors, num_clusters)
        # print("labels: ", labels)
        # print("centroids: ", centroids)
        
        label_dict = {}
        for i in range(num_clusters):
            label_dict[i] = []
        
        for i in range(vectors.shape[0]):
            label_dict[labels[i]].append(vectors[i])
        
        #determine the new_sentences
        for i in range(num_clusters):
            min_vector = self.find_closest_vector(centroids[i], label_dict[i])       
            
            #found the min_vector, so go add entry to table and make parent
            graph.add_node(table.retrieve_sentence(min_vector) + "*")        
            parent_sentences.append(table.retrieve_sentence(min_vector) + "*")
            parent_vectors.append(np.array(centroids[i], copy = True))
     
            #add edges to the graph, parent to child
            for v in label_dict[i]:
                graph.add_child_edges(parent_sentences[i], table.retrieve_sentence(v))
        
        vectors = np.array(parent_vectors, copy = True)
        
        if num_clusters == 1:
            graph.set_root(parent_sentences[0])
            return graph
        else:
            # graph.print()
            return self.kmeans_merge_clustering(parent_sentences, vectors, graph)
        
    def find_closest_vector(self, centroid, vectors):
        """find v from vectors that is closest to the center
        
        ***********************************************************
        design choices:
            used cosing similiarity over euclidean distance for comparison metric
        ************************************************************************************
        
        Preconditions:
        center -- 1D numpy array -- centroid of cluster
        vectors -- 2D numpy array -- set of vectors
        Postconditions:
        returns 1D numpy array -- one that is closest to the centroid
        """
        min_distance = 100
        min_vector = None
        for v in vectors:
            if cosine(v, centroid) < min_distance:
                min_distance = cosine(v, centroid)
                min_vector = v
        
        return min_vector       

if __name__ == "__main__":
    clusterer = kmeans()
    # vecs_a = np.array([[1.0, 0.5], [2.0, 0.5], [3.0, 0.5], [4.0, 0.5], [5.0, 0.5], [6.0, 0.5], [7.0, 0.5], [8.0, 0.5], [9.0, 0.5], [10.0, 0.5]])
    # vecs_b = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [3.0, 0.5], [4.0, 0.5], [5.0, 0.5]])
    # vecs_c = np.array([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5]])
    # vecs_d = np.array([[1.0, 1.0]])
    
    #TESTING initialize_points(vecs, num_clusters)
    # for i in range(1, 2):
        # print(clusterer.initialize_points(vecs_a, i))
        # print()
        # print(clusterer.initialize_points(vecs_b, i))
        # print()
        # print(clusterer.initialize_points(vecs_c, i))
        # print()

    # print(clusterer.initialize_points(vecs_d, 1))
    
        # print(clusterer.train(vecs_d, i))
    
    # print(clusterer.pick_number_clusters(vecs_a))
    
    vecs = np.array([[0, 0], [2, 0], [14, 3], [4, 6], [3, 5], [6, 6], [7, 7], [8, 8], [9, 3], [11, 44], [44, 12], [76, 13], [14, 14], [15, 15], [16, 16], [17, 9], [18, 14], [19, 19]])
    num_rows = vecs.shape[0]
    sents = []
    for i in range(num_rows):
        sents.append(str(i))
    
    graph = clusterer.kmeans_merge_clustering(sents, vecs)
    graph.print()

    
            

                    
            
        