import numpy as np
import ast
from ts_cluster import TSCluster

from matplotlib import pyplot as plt


def cluster(data, num_clusters=3, num_iters=10, window=1, optimize=False, plot=False):
    obj = TSCluster(num_clusters)
    obj.k_means_clust(data, num_iters, window, optimize)
    if plot:
        obj.plot_centroids()

    return obj.get_assignments(), obj.get_centroids()


def read_data(filename):
    data = []
    with open(filename) as file:
        for line in file: 
            line = line.strip()
            x = ast.literal_eval(line)
            data.append(x)
    
    return data

def main(filename):
    # read data
    data = read_data(filename)
    cluster(data, 5, 10) # 5 clusters, 10 interations


if __name__ == '__main__':
    filename1 = 'medians_all.txt' # 2D list
    filename2 = 'diff_medians_all.txt' # 2D list
    main(filename1)

