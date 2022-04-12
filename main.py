import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

nb_clusters = 2
returns = pd.read_excel("./DATA_PROJECT.xlsx", index_col=0)  # monthly returns of stocks

distance = pdist(returns.T, metric='correlation')  # clustering distance is the correlation between returns
output = linkage(distance, method='ward')

# GRAPH
# plt.figure(figsize=(10, 10))
# dendrogram(output, color_threshold=1.5, truncate_mode='level', orientation='right', leaf_font_size=10,
#            labels=returns.columns)
# plt.show()

clusters = fcluster(output, nb_clusters, criterion='maxclust')
