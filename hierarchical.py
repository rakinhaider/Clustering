import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from metrics import Metrics


def get_centroids(data, clusters):
    data = data.copy(deep=True)
    data['label'] = clusters
    grouped = data.groupby(by='label')
    centroids = pd.DataFrame()
    for label, group in grouped:
        centroid = group.mean(axis=0)
        centroid.name = label
        centroids = centroids.append(centroid)

    return centroids.drop(columns=['label'])


def measure_clustering_quality(data, labels, k, Z):
    clusters = fcluster(Z, t=k, criterion='maxclust')
    centroids = get_centroids(data, clusters)
    metrics = Metrics()
    wc = metrics.wc_cluster_distances(data, clusters=clusters, centroids=centroids)
    sc = metrics.silhouette_coefficient(data, clusters=clusters)
    nmi = metrics.nmi_gain(data, labels, clusters)
    print(k)
    print('WC: {:.3f}'.format(wc))
    print('SC: {:.3f}'.format(sc))
    print('NMI: {:.3f}'.format(nmi))
    return wc, sc, nmi




if __name__ == "__main__":
    data_filename = 'digits-embedding.csv'
    target_attributes = 1
    # Preparing Dataset(i)
    data = pd.read_csv(data_filename, header=None, index_col=0)
    del data.index.name

    # Answer to the question 3.2
    grouped = data.groupby(by=[1])
    samples = pd.DataFrame()
    for index, group in grouped:
        smp = group.sample(n=10, random_state=47)
        samples = samples.append(smp)

    labels = samples[1]
    Z = {}
    data = samples.drop(columns=[1])
    Z['single'] = linkage(data, method='single')
    dendrogram(Z['single'],
               leaf_rotation=90)

    plt.savefig('outputs/' + 'single_linkge_dendogram.pdf', format='pdf')
    plt.show()


    # Answer to the question 3.3
    Z['complete'] = linkage(data, method='complete')
    dendrogram(Z['complete'],
               leaf_rotation=90)
    plt.savefig('outputs/' + 'complete_linkge_dendogram.pdf', format='pdf')
    plt.show()
    Z['average'] = linkage(data, method='average')
    dendrogram(Z['average'],
               leaf_rotation=90)
    plt.savefig('outputs/' + 'average_linkge_dendogram.pdf', format='pdf')
    plt.show()

    # Answer to the question 3.4
    K = [2, 4, 8, 16, 32]

    wc_z = {}
    sc_z = {}
    nmi_z = {}
    for linkage_name in Z.keys():
        wcs = []
        scs = []
        nmis = []
        for k in K:
            wc, sc, nmi = measure_clustering_quality(data, labels, k, Z[linkage_name])
            wcs.append(wc)
            scs.append(sc)
            nmis.append(nmi)

        wc_z[linkage_name] = wcs
        sc_z[linkage_name] = scs
        nmi_z[linkage_name] = nmis

    for linkage_name in Z.keys():
        plt.plot(K, wc_z[linkage_name], 'o-', label=linkage_name)

    plt.legend()
    plt.xticks(K)
    plt.savefig('outputs/' + 'hierarchical' + '_wc' + '.pdf', format='pdf')
    plt.cla()

    for linkage_name in Z.keys():
        plt.plot(K, sc_z[linkage_name], 'o-', label=linkage_name)

    plt.legend()
    plt.xticks(K)
    plt.savefig('outputs/' + 'hierarchical' + '_sc' + '.pdf', format='pdf')

    plt.cla()
    for linkage_name in Z.keys():
        plt.plot(K, nmi_z[linkage_name], 'o-', label=linkage_name)

    plt.legend()
    plt.xticks(K)
    plt.savefig('outputs/' + 'hierarchical' + '_nmi' + '.pdf', format='pdf')


    # Answer to the question 3.4
    best_k = 8

    # Answer to the question 3.5
    for linkage_name in Z.keys():
        _, _, nmi = measure_clustering_quality(data, labels, best_k, Z[linkage_name])
        print('NMI with {:} linkage: {:.3f}'.format(linkage_name, nmi))


