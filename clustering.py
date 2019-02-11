import numpy as np
import pandas as pd
import datetime as d
import random
import math
from timedataframe import TimeDataFrame
from pointwiseanalysis import PointwiseAnalysis

from matplotlib import pyplot as plt


def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    distance = math.sqrt(DTW[len(s1)-1, len(s2)-1])
    #print(distance)
    return distance


def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return math.sqrt(LB_sum)






def k_means_clust(data,num_clust,num_iter,w=5):
    clusters = []
    centroids=random.sample(data,num_clust)
    # print('init centers')
    # print(centroids)
    counter=0
    for n in range(num_iter):
        counter+=1
        #print(counter)
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if True: #LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
                    else:
                        pass #print('cur_dist>min_dist')

                else:
                    pass #print('LB_Keogh(i,j,5)>min_dist')

            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]
                #assignments[closest_clust].append(ind)

        #recalculate centroids of clusters
        
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                #print(data[k])
                clust_sum=clust_sum + np.array(data[k])

            #print('centroid update...')
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]

        print(assignments)
        clusters = assignments

    return clusters, centroids


'''
tdf = TimeDataFrame('raw_data_files/MA33_2011_1_15.csv')
keys = tdf.fetch_keys()

len(keys)

medians = []
diff_medians = []

for i in range(15, len(keys)):
    print(i)
    print(keys[i])
    try:
        key_series = tdf.fetch_series(keys[i])
        pa = PointwiseAnalysis(key_series, 96)
        medians.append(pa.med_series().values.tolist())
        diff_medians.append(pa.diff_med_series().values.tolist())
    except:
        pass



with open('medians_all.txt', 'w') as f:
    for item in medians:
        f.write("%s\n" % item)


with open('diff_medians_all.txt', 'w') as f:
    for item in diff_medians:
        f.write("%s\n" % item)

'''


import ast
medians = []
with open('medians_all.txt') as file:
    for line in file: 
        line = line.strip()
        x = ast.literal_eval(line)
        medians.append(x)


diff_medians = []
with open('diff_medians_all.txt') as file:
    for line in file: 
        line = line.strip()
        x = ast.literal_eval(line)
        diff_medians.append(x)




assignments, centroids=k_means_clust(medians,4,15,1)
for i in centroids:
    plt.plot(i)

plt.show()

