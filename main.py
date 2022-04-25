from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils  # if your IDE does not manage to import utils,
# you will need to import sys and give it the working path using sys.path.append

nb_clusters = 2
returns = pd.read_excel("./DATA_PROJECT.xlsx", index_col=0)  # monthly returns of stocks
market = returns["EUROSTOXX 50 TR index"]
returns = returns.drop("EUROSTOXX 50 TR index", axis=1)
rfr = 0
depth_regression = 36
depth_momentum = 12
init_level = 100
test_luck = True
nb_rd_portfolio = 100  # portfolios to test the null hypothesis of luck
cvar = 95
q = 95

# weights of the portfolio
weights = pd.DataFrame(columns=returns.columns, index=returns.index)
weights_luck = [pd.DataFrame(columns=returns.columns, index=returns.index) for _ in range(nb_rd_portfolio)]
# momentum scores based on returns for the full sample
returns_momentum_scores = utils.momentum(depth_momentum, returns, False)
# volatilities
volatilities = utils.volatility(6, returns, np.sqrt(12))

for i in tqdm(range(max(depth_momentum, depth_regression), len(returns.index))):
    # MOMENTUM SCORE
    # momentum based on return
    returns_momentum = returns_momentum_scores.iloc[i]
    # momentum based on alphas
    # Computations of the betas in the CAPM model for the second momentum metric using the last 36 month of data
    betas = utils.capm(returns.iloc[i - depth_regression + 1:i + 1, :].values,
                                    market.iloc[i - depth_regression + 1:i + 1].values, rfr)[1, :]
    alphas = pd.DataFrame(index=returns.iloc[i - depth_momentum + 1:i + 1].index)
    for a, asset in enumerate(returns.columns):
        alphas[asset] = (market.iloc[i - depth_momentum + 1:i + 1] - rfr) * betas[a] + rfr
    specific_momentum = alphas.mean()
    global_momentum_score = ((specific_momentum + returns_momentum) / 2).to_frame("score")

    # CLUSTERING
    # clustering distance is the correlation between returns
    distance = pdist(returns.iloc[:i + 1, :].T, metric='correlation')
    output = linkage(distance, method='ward')
    clusters = fcluster(output, nb_clusters, criterion='maxclust')
    global_momentum_score["cluster"] = clusters

    # PORTFOLIO CONSTRUCTION
    for c in range(nb_clusters):
        # long or short position given the score of the asset and the median score
        scores_cluster = global_momentum_score[global_momentum_score.cluster == c + 1]["score"]
        median_score = np.median(scores_cluster)
        long_short_positions = ((scores_cluster >= median_score) * 2 - 1)  # puts 1 if score >= median_score else -1
        # weighting by the inverse of the volatility
        cluster_weights = utils.inverse_vol(long_short_positions, volatilities.iloc[i])
        # global scaling
        cluster_weights = cluster_weights * len(cluster_weights) / returns.shape[1]
        # saving the weights
        weights.iloc[i][cluster_weights.index] = cluster_weights["w"]

    # RANDOM PORTFOLIO
    if test_luck:
        for p in range(nb_rd_portfolio):
            # random long short position
            long_short_positions = pd.Series(1, index=returns.columns)
            short = np.random.choice(returns.columns, int(returns.shape[1]/2))
            long_short_positions[short] = -1
            # weighting by the inverse of the volatility
            cluster_weights = utils.inverse_vol(long_short_positions, volatilities.iloc[i])
            # saving the weights
            weights_luck[p].iloc[i][cluster_weights.index] = cluster_weights["w"]

# RETURNS AND LEVELS OF THE STRAT
returns_strat = (weights * returns).dropna().sum(axis=1)
level_strat = returns_strat.add(1).cumprod()
level_strat = init_level * level_strat / level_strat[0]
market_level = market[depth_regression:].add(1).cumprod()
market_level = init_level * market_level / market_level[0]

stats_market = utils.Statistics(market, freq=12, rfr=rfr, CVaR_conf=cvar/100).run()
stats = utils.Statistics(returns_strat, freq=12, rfr=rfr, CVaR_conf=cvar/100).run()

# RETURNS AND LEVELS OF RANDOM PORTFOLIOS
if test_luck:
    returns_luck = list(map(lambda x: (x * returns).dropna().sum(axis=1), weights_luck))
    level_luck = list(map(lambda x: x.add(1).cumprod(), returns_luck))
    level_luck = list(map(lambda x: init_level * x / x[0], level_luck))

    perf = list()
    ann_return = list()
    ann_vol = list()
    sharpe = list()
    max_drawdown = list()
    c_var = list()
    for p in range(nb_rd_portfolio):
        plt.plot(level_luck[p], c="grey", alpha=0.3)
        stats_luck = utils.Statistics(returns_luck[p], freq=12, rfr=rfr, CVaR_conf=cvar/100).run()
        perf.append(stats_luck["performance"])
        ann_return.append(stats_luck["annualized return"])
        ann_vol.append(stats_luck["annualized volatility"])
        sharpe.append(stats_luck["Sharpe ratio"])
        max_drawdown.append(stats_luck["maximum drawdown"]["full period"])
        c_var.append(stats_luck[f"CVaR ({cvar}%)"])

    # Statistics
    p = np.percentile(perf, q)
    r = np.percentile(ann_return, q)
    vol = np.percentile(ann_vol, q)
    sr = np.percentile(sharpe, q)
    md = np.percentile(max_drawdown, q)
    cv = np.percentile(c_var, q)

    print(f"Performance strategy: {stats['performance']}, market: {stats_market['performance']}, luck: {p}")
    print(f"Annualized return strategy: {stats['annualized return']}, market: {stats_market['annualized return']}, luck: {r}")
    print(f"Annualized volatility strategy: {stats['annualized volatility']}, market: {stats_market['annualized volatility']}, luck: {vol}")
    print(f"Sharpe strategy: {stats['Sharpe ratio']}, market: {stats_market['Sharpe ratio']}, luck: {sr}")
    print(f"Maximum drawdown strategy: {stats['maximum drawdown']['full period']}, market: {stats_market['maximum drawdown']['full period']}, luck: {md}")
    print(f"CVaR strategy: {stats[f'CVaR ({cvar}%)']}, market: {stats_market[f'CVaR ({cvar}%)']}, luck: {cv}")

plt.plot(market_level, label="Eurostoxx 50")
plt.plot(level_strat, label="Strategy")
plt.title("Strategy track record")
plt.xlabel("Dates")
plt.ylabel("Level")
plt.legend()


