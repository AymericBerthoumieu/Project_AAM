from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import financial_measures
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np

nb_clusters = 2
returns = pd.read_excel("./DATA_PROJECT.xlsx", index_col=0)  # monthly returns of stocks
market = returns.mean(axis=1)  # TODO: what is the market?
rfr = 0
depth_regression = 36
depth_momentum = 12
init_level = 100

# weights of the portfolio
weights = pd.DataFrame(columns=returns.columns, index=returns.index)
# momentum scores based on returns for the full sample
returns_momentum_scores = financial_measures.momentum(depth_momentum, returns, False)
# volatilities
volatilities = financial_measures.volatility(12, returns, np.sqrt(12))  # TODO: volatility depth?

for i in tqdm(range(max(depth_momentum, depth_regression), len(returns.index))):
    # MOMENTUM SCORE
    # momentum based on return
    returns_momentum = returns_momentum_scores.iloc[i]
    # momentum based on alphas
    # Computations of the betas in the CAPM model for the second momentum metric using the last 36 month of data
    betas = financial_measures.capm(returns.iloc[i - depth_regression + 1:i + 1, :].values,
                                    market.iloc[i - depth_regression + 1:i + 1].values, rfr)[1, :]
    alphas = pd.DataFrame(index=returns.iloc[i - depth_momentum + 1:i + 1].index)
    for a, asset in enumerate(returns.columns):
        alphas[asset] = (market.iloc[i - depth_momentum + 1:i + 1] - rfr) * betas[a] + rfr
    specific_momentum = alphas.mean()
    global_momentum_score = ((specific_momentum + returns_momentum) / 2).to_frame("score")

    # CLUSTERING
    distance = pdist(returns.iloc[:i + 1, :].T,
                     metric='correlation')  # clustering distance is the correlation between returns
    output = linkage(distance, method='ward')
    clusters = fcluster(output, nb_clusters, criterion='maxclust')
    global_momentum_score["cluster"] = clusters

    # PORTFOLIO CONSTRUCTION
    for c in range(nb_clusters):
        scores_cluster = global_momentum_score[global_momentum_score.cluster == c + 1]["score"].sort_values()
        median_score = scores_cluster[int(scores_cluster.shape[0] / 2) - 1]
        # weights are inverse volatility
        cluster_weights = (
                ((scores_cluster >= median_score) * 2 - 1) / volatilities[scores_cluster.index].iloc[i]).to_frame(
            "w").fillna(0)
        # scaling intra cluster
        cluster_weights[cluster_weights.w < 0] = cluster_weights[cluster_weights.w < 0] / abs(
            cluster_weights[cluster_weights.w <= 0]["w"].sum())
        cluster_weights[cluster_weights.w > 0] = cluster_weights[cluster_weights.w > 0] / cluster_weights[
            cluster_weights.w > 0]["w"].sum()
        # global scaling
        cluster_weights = cluster_weights * len(cluster_weights) / returns.shape[1]
        # saving the weights
        weights.iloc[i][cluster_weights.index] = cluster_weights["w"]

# RETURNS AND LEVELS OF THE STRAT
returns_strat = (weights * returns).dropna().sum()
level_strat = returns.add(1).cumprod()
level_strat = init_level * level_strat / level_strat[0]
