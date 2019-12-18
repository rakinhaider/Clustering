import pandas as pd
import numpy as np
import utils as util
import sys
from exploration import get_scatter_plot
from scipy.spatial.distance import pdist
import time
import matplotlib.pyplot as plt
import metrics as mtr


class KMeans:
    def __init__(self, k):
        self.k = k
        self.cluster_centroids = []
        self.labels = []

    def fit(self, data, max_iter=50, visualize=False):
        self.labels = [0] * len(data)
        select_index = np.random.randint(0, len(data), size=self.k)
        self.cluster_centroids = data.iloc[select_index]
        for i in range(max_iter):
            if not util.final:
                print(i)
            prev_labels = self.labels
            prev_centroids = self.cluster_centroids
            self.update_assignments(data)
            self.update_centroids(data)
            if visualize:
                self.visualize(data, self.labels, self.k).show()

            if not self.is_updated(prev_centroids, prev_labels):
                break

    def is_updated(self, prev_centroids, prev_labels):
        pc = prev_centroids.reset_index(drop=True)
        cc = self.cluster_centroids.reset_index(drop=True)

        pl = pd.DataFrame(prev_labels)
        cl = pd.DataFrame(self.labels)

        if pc.equals(cc) or pl.equals(cl):
            return False
        return True

    def visualize(self, data, labels, size):
        plt.cla()
        plot = get_scatter_plot(plt, data[2], data[3], labels, size=size)
        # plot = get_scatter_plot(plot, self.cluster_centroids[2], self.cluster_centroids[3], self.cluster_centroids.index, size=20, colorbar=False)
        return plot

    def predict(self, data):
        distances = self.get_centroid_distances(data)
        predictions = distances.idxmin(axis=1).values
        return predictions

    def update_assignments(self, data):
        labels = []
        distances = self.get_centroid_distances(data)
        self.labels = distances.idxmin(axis=1).values

    def get_centroid_distances(self, data):
        distances = pd.DataFrame()
        metric = mtr.Metrics()
        distances = metric.euclidean_dist(data, self.cluster_centroids)
        return distances

    def update_centroids(self, data):
        df = data.copy(deep=True)
        df['label'] = self.labels
        grouped = df.groupby(by='label')
        centroids = []
        for value, group in grouped:
            centroids.append(group.mean(axis=0))

        centroids_df = pd.DataFrame(centroids)
        self.cluster_centroids = centroids_df.drop(columns=['label'])


def train_and_measure(data, labels, k):
    model = KMeans(k)
    model.fit(data, max_iter=50, visualize=False)
    predictions = model.predict(data)
    metric = mtr.Metrics()

    wc = metric.wc_cluster_distances(data, predictions, model.cluster_centroids)
    start = time.time()
    sc = metric.silhouette_coefficient(data, predictions)
    # print("seconds %s" % (time.time() - start))
    nmi = metric.nmi_gain(data, labels, predictions)

    return model, wc, sc, nmi


if __name__ == "__main__":
    np.random.seed(0)

    data_filename = 'digits-embedding.csv'
    k = 10
    max_iter = 50
    if len(sys.argv) >= 3:
        data_filename = sys.argv[1]
        k = int(sys.argv[2])
    data = pd.read_csv(data_filename, header=None, index_col=0)
    del data.index.name

    labels = data[1]
    data = data.drop(columns=[1])
    if not util.final:
        data = data.iloc[0:20]
        k = 2
        max_iter = 5


    # Answer to the question 2.1
    model, wc, sc, nmi = train_and_measure(data, labels, k)
    print('WC: {:.3f}'.format(wc))
    print('SC: {:.3f}'.format(sc))
    print('NMI: {:.3f}'.format(nmi))
    # Plotting the final clustered version
    # plot = model.visualize(data)
    # plot.savefig('outputs/' + 'visualization.pdf', format='pdf')

    # Answer to the question 2.2 is in kmeans_analysis.py

