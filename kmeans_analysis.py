import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import train_and_measure


def plot_and_save(results, K, to_plot):
    plt.cla()
    if to_plot == 'wc':
        color = 'b'
    elif to_plot == 'sc':
        color = 'g'
    else:
        color = 'r'
    for i in range(len(results)):
        plt.plot(K, results[i][to_plot], color + 'o-', label='Dataset ' + str(i + 1) + ' ' + to_plot)
        plt.legend()
        plt.xticks(K)
        plt.savefig('outputs/' + to_plot + str((i+1)) + '.pdf', format='pdf')
        plt.cla()


    for i in range(len(results)):
        plt.plot(K, results[i][to_plot], 'o-', label=str(i + 1))

    plt.legend()
    plt.xticks(K)
    plt.savefig('outputs/' + to_plot + '.pdf', format='pdf')



if __name__ == "__main__":
    np.random.seed(0)

    datasets = []

    data_filename = 'digits-embedding.csv'
    target_attributes = 1
    # Preparing Dataset(i)
    data = pd.read_csv(data_filename, header=None, index_col=0)
    del data.index.name
    datasets.append(data)
    # Preparing Dataset(ii)
    data2 = data[(data[1] == 2) | (data[1] == 4) | (data[1] == 6) | (data[1] == 7)]
    datasets.append(data2)
    # Preparing Dataset(iii)
    data3 = data[(data[1] == 6) | (data[1] == 7)]
    datasets.append(data3)

    # Answer to the question 2.2.1
    K = [2, 4, 8, 16, 32]

    for i in range(len(datasets)):
        print('Dataset ' + str((i+1)))
        data = datasets[i]
        labels = data[target_attributes]
        data = data.drop(columns=[target_attributes])
        f = open('outputs/' + 'Dataset_' + str((i+1)) + '.csv', 'w')
        f.write('k' + '\t' + 'wc' + '\t' + 'sc'+ '\t' + 'nmi' + '\n')
        for k in K:
            print(k)
            model, wc, sc, nmi = train_and_measure(data, labels, k)
            print('WC: {:.3f}'.format(wc))
            print('SC: {:.3f}'.format(sc))
            print('NMI: {:.3f}'.format(nmi))
            f.write(str(k) + '\t' + str(wc) + '\t' + str(sc) + '\t' + str(nmi) + '\n')

        f.close()

    results = []
    for i in range(len(datasets)):
        results.append(pd.read_csv('outputs/' + 'Dataset_' + str((i+1)) + '.csv', sep='\t'))

    # print(results)

    plot_and_save(results, K, 'wc')
    plot_and_save(results, K, 'sc')
    plot_and_save(results, K, 'nmi')

    # Answer to the question 2.2.2
    best_k = [8, 4, 8]

    # Answer to the question 2.2.3
    random_seeds = np.random.randint(0, high=100, size=10)
    # random_seeds = [i for i in range(5)]
    av_wc_dataset = []
    sd_wc_dataset = []
    av_sc_dataset = []
    sd_sc_dataset = []

    for i in range(len(datasets)):
        print('Dataset', str(i+1))
        av_wc = []
        sd_wc = []
        av_sc = []
        sd_sc = []
        for k in K:
            print(k)
            wcs = []
            scs = []
            for random_seed in random_seeds:
                np.random.seed(random_seed)
                data = datasets[i]
                labels = data[target_attributes]
                data = data.drop(columns=[target_attributes])
                model, wc, sc, nmi = train_and_measure(data, labels, k)
                wcs.append(wc)
                scs.append(sc)

            av_wc.append(np.mean(wcs))
            sd_wc.append(np.std(wcs))
            av_sc.append(np.mean(scs))
            sd_sc.append(np.std(scs))

            print('Average within cluster SSD: {:.3f}'.format(av_wc[-1]))
            print('Standard Deviation of within cluster SSD: {:.3f}'.format(sd_wc[-1]))
            print('Average Silhouette Coefficient: {:.3f}'.format(av_sc[-1]))
            print('Standard Deviation of Silhouette Coefficients: {:.3f}'.format(sd_sc[-1]))


        av_wc_dataset.append(av_wc)
        sd_wc_dataset.append(sd_wc)
        av_sc_dataset.append(av_sc)
        sd_sc_dataset.append(sd_sc)

    plt.clf()
    for i in range(len(datasets)):
        plt.errorbar(K, av_wc_dataset[i], yerr=sd_wc_dataset[i], label= 'Dataset ' + str(i+1), fmt='o-')

    plt.legend(loc='upper right')
    plt.xticks(K)
    plt.tight_layout(pad=3)
    plt.xlabel('K.')
    plt.ylabel('WC SSD')
    plt.savefig('outputs/' + '2_2_3_WCSSD.pdf', format='pdf')
    plt.clf()

    for i in range(len(datasets)):
        plt.errorbar(K, av_sc_dataset[i], yerr=sd_sc_dataset[i], label= 'Dataset ' + str(i+1), fmt='o-')

    plt.legend(loc='upper right')
    plt.xticks(K)
    plt.tight_layout(pad=3)
    plt.xlabel('K.')
    plt.ylabel('Silhouette Coefficient')
    plt.savefig('outputs/' + '2_2_3_SC.pdf', format='pdf')
    plt.clf()

    # Answer to the question 2.2.4
    best_k = [8, 4, 8]
    for i in range(len(datasets)):
        print('Dataset', str(i+1))
        data = datasets[i]
        labels = data[target_attributes]
        data = data.drop(columns=[target_attributes])
        np.random.seed(0)
        model, wc, sc, nmi = train_and_measure(data, labels, best_k[i])
        print('NMI: {:.3f}'.format(nmi))
        data['label'] = labels
        data = data.sample(n=1000, random_state=47)
        plt.clf()
        model.visualize(data[data.columns[:-1]], model.predict(data[data.columns[:-1]]), size=best_k[i])
        plt.tight_layout()
        plt.savefig('outputs/' + '2_2_4_Dataset_' + 'i'*(i+1) + '.pdf', format='pdf')
