from sklearn.cluster import KMeans
import numpy as np
from graph import Graph, Table
from scipy.spatial import distance
from sklearn import metrics

def kmeans_merge_clustering(sentences, vectors, num_clusters):
    """
    Purpose: recursively merge the clusters together so that a graph of sentences is formed
    ***************************************************************************************
    design choices:
        used kmeans (alternative: other possible clustering algorithms
        initial parameter is num_clusters (alternative: could be branching factor, try to develop a way so that there is no parameter, use keyphrase candidate as initial num_clusters)
        pick one node as the root: 1 sentence summary
		use sentences in multiple groups in growth towards heirarchy
    """
    #construct Graph structure to return at end of method
    graph = Graph()
    
    #add all current sentences node to the graph
    for s in sentences:
        graph.add_node(s)
    
    #set up sentence/vector table
    table = Table(sentences, vectors)
    
    #these are the set of vectors to be clustered
    vectors_to_be_clustered = np.array(vectors, copy = True)

    #recursively merge
    while True:
        #these variables will be used to set up next iteration
        sents_next_level = []
        vecs_next_level = []
        
        #set up sentence_cluster_dict (index --> label, value --> list of numpy vectors)
        sentence_cluster_dict = {}
        for i in range(num_clusters):
            sentence_cluster_dict[i] = []
        
        #perform kmeans clustering, get labels for each example, and find the center of each cluster
        kmeans = KMeans(n_clusters = num_clusters).fit(vectors_to_be_clustered)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        #put each sentence and vector into the right group
        for i in range(len(vectors_to_be_clustered)):
            sentence_cluster_dict[labels[i]].append(vectors_to_be_clustered[i])
        
        #for each cluster, find out which sentence is closest to the centroid and make parent of the other sentences
        for i in range(num_clusters):
            min_vector = find_closest_vector(centroids[i], sentence_cluster_dict[i])       
            
            #found the min_vector, so go add entry to table and make parent
            graph.add_node(table.retrieve_sentence(min_vector) + "*")        
            sents_next_level.append(table.retrieve_sentence(min_vector) + "*")
            vecs_next_level.append(np.array(centroids[i], copy = True))
     
            #add edges to the graph, parent to child
            for v in sentence_cluster_dict[i]:
                graph.add_child_edges(sents_next_level[i], table.retrieve_sentence(v))
        
        #if the base case is reached, return the complete graph
        if num_clusters == 1:
            graph.set_root(sents_next_level[0])
            return graph
        #else set up the new sent/vec table, vectors_to_be_clustered, and the ideal num_clusters
        else:
            table = Table(sents_next_level, vecs_next_level)
            vectors_to_be_clustered = np.array(vecs_next_level, copy = True)
            num_clusters = find_ideal_num_clusters(vectors_to_be_clustered, num_clusters)

def find_closest_vector(center, vectors):
    """
    Purpose: find v from vectors that is closest to the center
    ***********************************************************
    design choices:
        used cosing similiarity over euclidean distance for comparison metric
    """
    min_distance = 100
    min_vector = None
    for v in vectors:
        if distance.cosine(v, center) < min_distance:
            min_distance = distance.cosine(v, center)
            min_vector = v
    
    return min_vector

def find_ideal_num_clusters(vectors, num_clusters):
    """
    Purpose: heuristic of find the optimal number of clusters based off silhouette score
    ************************************************************************************
    desgin choices:
        if num clusters is 1, 2, or 3, just use 1 cluster (ALTERNATIVE: only do this if 1 or 2)
        otherwise, test the number of clusters from 2 to n - 1 (ALTERNATIVE: make upper limit int(n / 2) + 1)
        use silhouette_score as metric to determine optimal number of clusters (ALTERNATIVE: use other sklearn metric)  
    """
    
    #if num_clusters is less than 4, then only 1 cluster will be used
    if num_clusters < 4:
        return 1
    
    #possible values of the number of clusters
    num_clusters_range = [x for x in range(2, num_clusters)]
    
    max_score = -2
    preferred_cluster = -1
    
    #pick the number of cluster with the highest silhouette score
    for c in num_clusters_range:
        kmeans = KMeans(n_clusters = c).fit(vectors)
        score = metrics.silhouette_score(vectors, kmeans.labels_, metric = 'euclidean')
        if score > max_score:
            max_score = score
            preferred_cluster = c
    
    return preferred_cluster
    
    
#main method for testing functionality
if __name__ == "__main__":
    #setup
    vecs = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19]])
    num_rows = vecs.shape[0]
    sents = []
    for i in range(num_rows):
        sents.append(str(i))
    
    #test to see if outline algorithm works
    kmeans_merge_clustering(sents, vecs, 6).print()