from metrics import Metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame()

K = [2, 4, 8, 16, 32]

df1 = pd.read_csv('outputs/2.2.3.Dataset_1.csv', sep='\t', header=None)
df2 = pd.read_csv('outputs/2.2.3.Dataset_2.csv', sep='\t', header=None)
df3 = pd.read_csv('outputs/2.2.3.Dataset_3.csv', sep='\t', header=None)

print(df1)
print(df2)
print(df3)

plt.errorbar(K, df1[1], yerr=df1[2], fmt='o-', label='Dataset 1')
plt.errorbar(K, df2[1], yerr=df2[2], fmt='o-', label='Dataset 2')
plt.errorbar(K, df3[1], yerr=df3[2], fmt='o-', label='Dataset 3')
plt.legend(loc='upper right')
plt.xticks(K)
plt.tight_layout(pad=3)
plt.xlabel('K.')
plt.ylabel('WC SSD')
plt.savefig('outputs/' + '2_2_3_WCSSD.pdf', format='pdf')
plt.show()

plt.cla()

plt.errorbar(K, df1[3], yerr=df1[4], fmt='o-', label='Dataset 1')
plt.errorbar(K, df2[3], yerr=df2[4], fmt='o-', label='Dataset 2')
plt.errorbar(K, df3[3], yerr=df3[4], fmt='o-', label='Dataset 3')
plt.legend(loc='upper right')
plt.xticks(K)
plt.tight_layout(pad=3)
plt.xlabel('K.')
plt.ylabel('Silhouette Coefficient')
plt.savefig('outputs/' + '2_2_3_SC.pdf', format='pdf')
plt.show()

