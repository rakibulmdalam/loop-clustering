import numpy as np
import ast
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score, davies_bouldin_score
import matplotlib.ticker as ticker
from scipy.spatial.distance import cdist
from numpy import mean, median, var
import scores_95 as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def DTWDistance(s1,s2,w=None):
    '''
    Calculates dynamic time warping Euclidean distance between two
    sequences. Option to enforce locality constraint for window w.
    '''
    DTW={}

    if w:
        w = max(w, abs(len(s1)-len(s2)))

        for i in range(-1,len(s1)):
            for j in range(-1,len(s2)):
                DTW[(i, j)] = float('inf')

    else:
        for i in range(len(s1)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(s2)):
            DTW[(-1, i)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        if w:
            for j in range(max(0, i-w), min(len(s2), i+w)):
                dist= (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
        else:
            for j in range(len(s2)):
                dist= (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])


def prepare_features(data):
    '''
    This method prints on the console the features list, which is a 95x95 list of lists:
    For each time series, its difference from all other time series is taken as a single feature (according to the TRY #2 below).
    So, each time series has 95 features.
    Since there are 95 time series, hence the 95x95 list of lists that is being outputed.
    To save time, I just copied the output of this method to a file called "scores_95.py" (that is being inspected by the cluster method below)
    '''

    ##########
    # TRY #1: FEATURES BASED ON COMMON STATISTICAL METRICS
    ##########
    # all_scores = []
    # for i in range(95):
    #     d1 = data[i]
    #     scores = []
    #     scores.append(mean(d1))
    #     scores.append(median(d1))
    #     scores.append(var(d1))
    #     scores.append(max(d1))
    #     scores.append(min(d1))
    #     all_scores.append(scores)
    #     print "ld: " + str(i)


    ##########
    # TRY #2: FEATURES BASED ON DTW SCORES
    ##########
    all_scores = []
    for i in range(95):
        d1 = data[i]
        scores = []
        for j in range(95):
            if j == i:
                scores.append(0)
            else:
                d2 = data[j]
                score = DTWDistance(d1, d2, 1)
                scores.append(score)
        all_scores.append(scores)
        print "ld: " + str(i)

    print all_scores


def cluster():
    '''
    Its input (features for all 95 time series) comes from the file "scores_95.py"
    '''

    all_scores = sc.all_scores
    # all_scores = StandardScaler().fit_transform(all_scores)

    distortions = []
    distortions2 = []
    distortions3 = []
    silhouette_scores = []
    calinski_harabaz_scores = []
    davies_bouldin_scores = []

    K = range(1,30)
    for n_clusters in K:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)

        kmeans = clusterer.fit(all_scores)

        print kmeans.labels_
        print kmeans.inertia_
        print kmeans.n_iter_
        # print kmeans.cluster_centers_

        distortion = sum(np.min(cdist(all_scores, kmeans.cluster_centers_, 'euclidean'), axis=1)) / len(all_scores)
        distortions.append(distortion)
        distortions2.append(kmeans.inertia_)
        distortions3.append(kmeans.score(all_scores))

        if n_clusters > 1:
            silhouette_avg = silhouette_score(all_scores, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
            calinski_harabaz = calinski_harabaz_score(all_scores, kmeans.labels_)
            calinski_harabaz_scores.append(calinski_harabaz)
            print("For n_clusters =", n_clusters, "The calinski_harabaz is :", calinski_harabaz)
            davies_bouldin = davies_bouldin_score(all_scores, kmeans.labels_)
            davies_bouldin_scores.append(davies_bouldin)
            print("For n_clusters =", n_clusters, "The davies_bouldin is :", davies_bouldin)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel('k')
    plt.ylabel('Distortion as Euclidean distance')
    # plt.title('The Elbow Method showing the optimal k')
    plt.savefig("elbow1.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.plot(K, distortions2, 'bo-')
    plt.xlabel('k')
    plt.ylabel('Distortion as Inertia')
    plt.savefig("elbow2")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.plot(K, distortions3, 'bo-')
    plt.xlabel('k')
    plt.ylabel('Distortion as clustering score')
    plt.savefig("elbow3")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.plot(K[1:], silhouette_scores, 'bo')
    plt.xlabel('k')
    plt.ylabel('silhouette scores')
    plt.savefig("scores_silhouette.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.plot(K[1:], calinski_harabaz_scores, 'bo')
    plt.xlabel('k')
    plt.ylabel('calinski-harabaz scores')
    plt.savefig("scores calinski_harabaz")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.plot(K[1:], davies_bouldin_scores, 'bo')
    plt.xlabel('k')
    plt.ylabel('davies-bouldin scores')
    plt.savefig("scores davies_bouldin")

    pca = PCA().fit(all_scores)
    fig = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig("pca_variance")


    ##########
    # 2D PLOTS OF 2 PRINCIPAL COMPONENTS
    ##########
    # pca_result = PCA(n_components=2).fit_transform(all_scores)

    # # Xax=pca_result[:,0]
    # Yax=pca_result[:,1]
    # Xax=pca_result[:,2]
    # fig,ax=plt.subplots(figsize=(10,7))
    # fig.patch.set_facecolor('white')
    # # colors = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    # colors = [2,4,2,3,2,3,3,1,2,2,3,4,1,4,3,2,3,1,3,3,3,2,0,1,2,2,3,3,1,3,2,1,3,1,3,1,3,1,4,4,2,1,4,1,1,3,1,3,4,1,3,4,1,3,1,1,4,1,4,3,1,3,1,3,1,4,3,1,1,3,1,3,1,3,3,1,1,1,3,3,1,3,1,1,1,1,1,1,1,1,1,2,4,2,4]
    # ax.scatter(Xax,Yax, c=colors, cmap='viridis')
    # plt.xlabel("First Principal Component")
    # plt.ylabel("Second Principal Component")
    # plt.legend()
    # plt.savefig("pca")

    ##########
    # 3D PLOTS OF 3 PRINCIPAL COMPONENTS
    ##########
    pca_result = PCA(n_components=3).fit_transform(all_scores)

    Xax=pca_result[:,0]
    Yax=pca_result[:,1]
    Zax=pca_result[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    # colors = [2,4,2,3,2,3,3,1,2,2,3,4,1,4,3,2,3,1,3,3,3,2,0,1,2,2,3,3,1,3,2,1,3,1,3,1,3,1,4,4,2,1,4,1,1,3,1,3,4,1,3,4,1,3,1,1,4,1,4,3,1,3,1,3,1,4,3,1,1,3,1,3,1,3,3,1,1,1,3,3,1,3,1,1,1,1,1,1,1,1,1,2,4,2,4]
    ax.scatter(Xax,Yax, Zax, c=colors, cmap='viridis')
    ax.set_xlabel("1st principal component")
    ax.set_ylabel("2nd principal component")
    ax.set_zlabel("3rd principal component")
    plt.legend()
    # plt.show()
    plt.savefig("pca.pdf")


def read_data(filename):
    data = []
    with open(filename) as file:
        for line in file:
                line = line.strip()
                x = ast.literal_eval(line)
                data.append(x)
    return data


def plott(data, labels):

    format_list = ['',':']
    # print len(data)
    print len(labels)
    plt.figure()
    for l in range(2):

        ii = 0
        ind = 0

        for d in data:
            # print ii
            # print labels[ii]
            # print l
            if labels[ii]==l:

                if ind > 8:
                    plt.plot(d, format_list[l])
                    print l
                    print ii

                ind +=1

                if ind > 13:
                    break

            ii += 1

    plt.xlabel("Time slots for all days-of-the-week")
    plt.ylabel("Median of vehicle counts in 2011")
    plt.savefig("all_plot.pdf")


def plott2(data):

    i = 0
    for d in data:
        plt.figure()
        if i<2:
            plt.plot(d)
            plt.xlabel("Time slots for all days-of-the-week")
            plt.ylabel("Median of vehicle counts in 2011")
            if i==0:
                plt.title("MA33_12001_R1: Example of small cluster")
            else:
                plt.title("MA33_12001_R4: Example of large cluster")
        else:
            break
        i += 1
        plt.savefig(str(i)+"_plot")


def plot(filename, labels):

    data = read_data(filename)
    plott(data, labels)
    # plott2(data)


def main(filename):

    data = read_data(filename)

    prepare_features(data)

    cluster()


if __name__ == '__main__':

    filename1 = 'medians_all.txt' # 2D list
    # filename2 = 'diff_medians_all.txt' # 2D list

    main(filename1)

    # USED FOR PLOTTING AFTER YOU KNOW THE NUMBER OF CLUSTERS
    # labels = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    # plot(filename1, labels)





