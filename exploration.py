import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmp


def get_image_matrix(row):
    image = np.array(row[2:786].values).reshape((28, 28))
    return image


def get_scatter_plot(plt, x, y, c, size=None, colorbar=True):
    colormap = cmp.get_cmap('Spectral')
    if size is None:
        plt.scatter(x, y, c=c, cmap=colormap)
    else:
        plt.scatter(x, y, c=c, s=size, cmap=colormap)
    if colorbar:
        plt.colorbar()

    return plt


if __name__ == "__main__":
    # Answer to the question 1.1
    data = pd.read_csv('digits-raw.csv', header=None)

    for i in range(10):
        row = data[data[1] == i].iloc[0]
        image = get_image_matrix(row)
        plt.imshow(image, cmap='gray')
        plt.xlabel('Digit ' + str(i))
        plt.savefig('outputs/' + 'Digit ' + str(i) + '.pdf', format='pdf')
        plt.cla()

    # Answer to the question 1.2
    data = pd.read_csv('digits-embedding.csv', header=None)
    selected_row_indices = np.random.randint(0, len(data), size=1000)
    selected_row = data.iloc[selected_row_indices]
    plt = get_scatter_plot(plt, selected_row[2], selected_row[3], selected_row[1])
    plt.savefig('outputs/' + 'Clusters.pdf', format='pdf')
    plt.show()
