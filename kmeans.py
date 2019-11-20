import pandas as pd
import numpy as np
import utils as util
import sys
from exploration import get_scatter_plot
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k):
        self.k = k
        self.cluster_centroids = []
        self.labels = []

    def fit(self, data, max_iter=50, visualize=False):
        self.labels = [0] * len(data)
        np.random.seed(23)
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
                self.visualize(data).show()

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

    def visualize(self, data):
        plt.cla()
        plot = get_scatter_plot(plt, data[2], data[3], self.labels, size=10)
        plot = get_scatter_plot(plot, self.cluster_centroids[2], self.cluster_centroids[3],
                                self.cluster_centroids.index, size=20, colorbar=False)
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
        metric = Metrics()
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


class Metrics:
    def wc_cluster_distances(self, model, data):
        predictions = model.predict(data)
        data = data.copy(deep=True)
        data['label'] = predictions
        grouped = data.groupby('label')
        wc = 0
        for label, group in grouped:
            group = group[group.columns[:-1]]
            centroid_df = pd.DataFrame().append(model.cluster_centroids.loc[label])
            distances = self.euclidean_dist(group, centroid_df)
            wc = wc + np.sum((distances**2).values)
            del distances
        del grouped
        return wc

    def silhouette_coefficient(self, model, data):
        predictions = model.predict(data)
        data = data.copy(deep=True)
        data['label'] = predictions
        grouped = data.groupby('label')
        scs = pd.Series()
        for label, group in grouped:
            group = group[group.columns[:-1]]
            data_other_cluster = data.drop(group.index)[group.columns]
            a = self.euclidean_dist(group, group)
            a = a.sum(axis=1)
            if len(group) > 1:
                a = a / (len(group)-1)
            b = metric.euclidean_dist(data_other_cluster, group)
            b = b.sum(axis=0)
            if len(data_other_cluster) > 1:
                b = b / len(data_other_cluster)
            scis = (b-a).div(np.max([a.values, b.values], axis=0))
            del a
            del b
            del group
            scs = scs.append(scis)

        return scs.mean()

    def nmi_gain(self, model, data, labels):
        predictions = model.predict(data)
        data = data.copy(deep=True)
        data['cl_label'] = predictions
        data['orig_label'] = labels
        # c=y g=c
        entropy_c = self.entropy(data, 'orig_label')
        entropy_g = self.entropy(data, 'cl_label')

        entropy_c_g = self.conditional_entropy(data, conditional_attribute='cl_label', target_attribute='orig_label')
        return (entropy_c - entropy_c_g) / (entropy_c + entropy_g)

    def conditional_entropy(self, data, conditional_attribute, target_attribute):
        ecg = pd.Series()
        counts = data[conditional_attribute].value_counts()
        grouped = data.groupby(by='cl_label')
        for index, group in grouped:
            ecg.at[index] = self.entropy(group, 'orig_label')

        ecg = ecg * counts / counts.sum()
        return ecg.sum()

    def entropy(self, data, target_attribute):
        counts = data[target_attribute].value_counts()
        if len(counts) == 1:
            return 0
        counts = counts / counts.sum()
        counts = (-1) * counts * np.log10(counts)/np.log10(len(counts))
        return counts.sum()

    def euclidean_dist(self, x, y):
        if len(x)*len(y) > 10000000:
            print('splitting', len(x), len(y))
            sqrt_len = int(len(x)/2)
            distances = self.euclidean_dist(x.iloc[0:sqrt_len], y)
            dist = self.euclidean_dist(x.iloc[sqrt_len:], y)
            print(dist.shape)
            distances = distances.append(dist)
            print(distances.shape)
            del dist
            return distances
        else:
            xx = x * x
            yy = y * y
            mat_dist = pd.DataFrame(np.zeros((len(x), len(y)), dtype='float64'), columns=y.index, index=x.index)
            for index, row in y.iterrows():
                xy = x * row
                mat_dist[index] = mat_dist[index] - 2 * xy.sum(axis=1)
            xx_sum = xx.sum(axis=1)
            yy_sum = yy.sum(axis=1)
            mat_dist = mat_dist.add(xx_sum, axis=0)
            mat_dist = mat_dist + yy_sum
            del xx_sum
            del yy_sum
            del xx
            del yy
            del xy
            sqrt = np.sqrt(mat_dist)
            del mat_dist
            return sqrt


if __name__ == "__main__":
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
        data = data.iloc[0:3]
        k = 2
        max_iter = 5


    # Answer to the question 2.1
    model = KMeans(k)
    model.fit(data, max_iter=50, visualize=False)
    metric = Metrics()
    print('WC: {:.3f}'.format(metric.wc_cluster_distances(model, data)))
    print('SC: {:.3f}'.format(metric.silhouette_coefficient(model, data)))
    print('NMI: {:.3f}'.format(metric.nmi_gain(model, data, labels)))
    # Plotting the final clustered version
    plot = model.visualize(data)
    plot.savefig('outputs/' + 'visualization.pdf', format='pdf')

    # Answer to the question 2.2 is in kmeans_analysis.py

