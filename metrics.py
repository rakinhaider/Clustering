import numpy as np
import pandas as pd


class Metrics:
    def wc_cluster_distances(self, data, clusters, centroids):
        data = data.copy(deep=True)
        data['label'] = clusters
        grouped = data.groupby('label')
        wc = 0
        for label, group in grouped:
            group = group[group.columns[:-1]]
            centroid_df = pd.DataFrame().append(centroids.loc[label])
            distances = self.euclidean_dist(group, centroid_df)
            wc = wc + np.sum((distances**2).values)
            del distances
        del grouped
        return wc

    def silhouette_coefficient(self, data, clusters):
        mat_dist = self.euclidean_dist(data, data)
        data = data.copy(deep=True)
        data['label'] = clusters
        grouped = data.groupby('label')
        scs = pd.Series()
        for label, group in grouped:
            group = group[group.columns[:-1]]
            grp_indx = list(group.index)
            data_other_cluster = data.drop(grp_indx)[group.columns]
            other_grp_indx = list(data_other_cluster.index)
            a = mat_dist[grp_indx].loc[grp_indx].sum(axis=1) / (len(group)-1)
            b = mat_dist[other_grp_indx].loc[grp_indx].sum(axis=1) / len(data_other_cluster)
            scis = (b-a).div(np.max([a.values, b.values], axis=0))
            scs = scs.append(scis)

        return scs.mean()

    """
    def silhouette_coefficient(self, model, data):
        predictions = model.predict(data)
        data = data.copy(deep=True)
        data['label'] = predictions
        grouped = data.groupby('label')
        scs = pd.Series()
        for label, group in grouped:
            group = group[group.columns[:-1]]
            data_other_cluster = data.drop(group.index)[group.columns]
            l = len(group)

            batch_size = int(self.MAX_SIZE * 0.9 / len(data_other_cluster))
            a = pd.Series(np.zeros(l), index=group.index)
            for i in range(batch_size, len(group) + batch_size, batch_size):
                mat_dist = self.euclidean_dist(group.iloc[i - batch_size:i], group)
                a = a + mat_dist.sum(axis=0)
                del mat_dist

            if l > 1:
                a = a / (l-1)

            batch_size = int(self.MAX_SIZE * 0.9 / l)
            b = pd.Series(np.zeros(l), index=group.index)
            for i in range(batch_size, len(data_other_cluster)+batch_size, batch_size):
                mat_dist = self.euclidean_dist(data_other_cluster.iloc[i-batch_size:i], group)
                b = b + mat_dist.sum(axis=0)
                del mat_dist

            if len(data_other_cluster) > 1:
                b = b / len(data_other_cluster)
            scis = (b-a).div(np.max([a.values, b.values], axis=0))
            del a
            del b
            del group
            scs = scs.append(scis)

        return scs.mean()        
        """

    def nmi_gain(self, data, labels, clusters):
        data = data.copy(deep=True)
        num_clusters = len(np.unique(clusters))
        num_labels = len(np.unique(labels))
        data['cl_label'] = clusters
        data['orig_label'] = labels
        # c=y g=c
        entropy_c = self.entropy(data, 'orig_label', num_classes=num_labels)
        entropy_g = self.entropy(data, 'cl_label', num_classes=num_clusters)

        entropy_c_g = self.conditional_entropy(data, conditional_attribute='cl_label', target_attribute='orig_label')
        return (entropy_c - entropy_c_g) / (entropy_c + entropy_g)

    def conditional_entropy(self, data, conditional_attribute, target_attribute):
        ecg = pd.Series()
        num_clusters = len(data[conditional_attribute].unique())
        num_labels = len(np.unique(data[target_attribute]))
        counts = data[conditional_attribute].value_counts()
        grouped = data.groupby(by=conditional_attribute)
        for index, group in grouped:
            ecg.at[index] = self.entropy(group, target_attribute, num_labels)

        ecg = ecg * counts / counts.sum()
        return ecg.sum()

    def entropy(self, data, target_attribute, num_classes):
        counts = data[target_attribute].value_counts()
        if len(counts) == 1:
            return 0
        probab = counts / counts.sum()
        entropy = (-1) * probab * np.log10(probab) / np.log10(num_classes)
        return entropy.sum()

    """
    def euclidean_dist(self, x, y):
        mat_dist = pd.DataFrame()
        for y_index, row in y.iterrows():
            dist = (x - row) ** 2
            dist = dist.sum(axis=1)
            dist = np.sqrt(dist)
            mat_dist[y_index] = dist
        return mat_dist
    """

    def euclidean_dist(self, x, y):
        index = x.index
        x = x.values
        columns = y.index
        y = y.values

        dist = np.zeros((x.shape[0], y.shape[0]))
        for i in range(y.shape[1]):
            mat = x[:, i].reshape((len(x), 1)) - y[:, i].reshape((1, len(y)))
            mat = mat ** 2
            dist = dist + mat
            del mat

        dist = pd.DataFrame(dist, index=index, columns=columns)
        return np.sqrt(dist)
