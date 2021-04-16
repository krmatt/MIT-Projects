import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
seed = 4
K = 4
title = 'K = ' + str(K) + '; Seed = ' + str(seed)

mixture, post = common.init(X, K, seed)
mixture, post, cost = kmeans.run(X, mixture, post)

common.plot(X, mixture, post, title)

print('K =', K, 'Seed =', seed, 'cost:', cost)
