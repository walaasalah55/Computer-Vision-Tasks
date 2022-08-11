
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# calculate the euclidean distance bet 2 points
def euclidean_distance(point1, point2):
    
    return np.linalg.norm(np.array(point1) - np.array(point2))

# calculate the distance between 2 clusters
def clusters_distance(cluster1, cluster2):
    
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])
  
# calculate the distance between 2 centroids of 2 clusters
def clusters_distance_2(cluster1, cluster2):
   
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:
    
    def __init__(self, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k
        
    def initial_clusters(self, points):
       
        # partition pixels into k groups based on color similarity
        
        groups = {}
        d = int(256 / (self.initial_k))
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []

        for i, p in enumerate(points):
            if (i%100000 == 0):
                print('processing pixel:', i)
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))  
            groups[go].append(p)

        return [g for g in groups.values() if len(g) > 0]
        
    def fit(self, points):

        # first, we assign each point to a distinct cluster
        self.clusters_list = self.initial_clusters(points)
        print('Number of initial clusters:', len(self.clusters_list))
        while len(self.clusters_list) > self.k:
            # get the closest  pair of clusters
            cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                 key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the 2 clusters from clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the 2 clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to clusters list
            self.clusters_list.append(merged_cluster)

            print('number of clusters:', len(self.clusters_list))
        
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num
                
        # Compute cluster centers
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)
                    
    # find cluster number of point
    def predict_cluster(self, point):
        
        return self.cluster[tuple(point)]

    # find center of cluster of each point
    def predict_center(self, point):
        
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center


def get_agglomerative_output(path_to_jpg_file):
    img = mpimg.imread(path_to_jpg_file)
    print("image shape: ",img.shape)

    pixels = img.reshape(img.shape[0]*img.shape[1],3)
    print("image pixels: ",pixels.shape)
    
    agglo = AgglomerativeClustering(k=5, initial_k=25)
    agglo.fit(pixels)

    new_img = [[agglo.predict_center(list(pixel)) for pixel in row] for row in img]
    new_img = np.array(new_img, np.uint8)

    plt.imshow(new_img)
    plt.axis('off')
    plt.savefig('agglomerative.jpg',bbox_inches='tight',pad_inches = 0)