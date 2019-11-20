import pandas as pd
import numpy as np
from kmeans import KMeans, Metrics

datasets = []

data_filename = 'digits-embedding.csv'
target_attributes = 1
# Preparing Dataset(i)
data = pd.read_csv(data_filename, header=None, index_col=0)
datasets.append(data)
# Preparing Dataset(ii)
data2 = data[(data[1] == 2) | (data[1] == 4) | (data[1] == 6) | (data[1] == 7)]
datasets.append(data2)
# Preparing Dataset(iii)
data3 = data[(data[1] == 6) | (data[1] == 7)]
datasets.append(data3)

# Answer to the question 2.2.1
K = [2, 4, 8, 16, 32]

wc_distances = []
sil_distances = []
for data in datasets:
    wc_dis = []
    sil_dist = []
    labels = data[target_attributes]
    data = data.drop(columns=[target_attributes])
    for k in K:
        print(k)
        model = KMeans(k)
        model.fit(data, max_iter=50, visualize=False)
        metric = Metrics()
        print('WC: {:.3f}'.format(metric.wc_cluster_distances(model, data)))
        print('SC: {:.3f}'.format(metric.silhouette_coefficient(model, data)))
        print('NMI: {:.3f}'.format(metric.nmi_gain(model, data, labels)))
