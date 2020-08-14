import bisect
import random
import itertools
import numpy as np

def weighted_sample(population, weights, k=1):
    """Like random.sample, but add weights.
    """
    n = population.shape[0]
    if n<=k: return np.arange(0,n)
    population = [population[i] for i in range(n)]
    weights = weights.tolist()

    if n == 0:
        return np.zeros(0)


    cum_weights = list(itertools.accumulate(weights))
    total = cum_weights[-1]
    if total <= 0:
        return random.sample(population, k=k)
    hi = len(cum_weights) - 1

    selected = set()
    _bisect = bisect.bisect
    _random = random.random
    selected_add = selected.add
    result = [None] * k
    for i in range(k):
        j = _bisect(cum_weights, _random()*total, 0, hi)
        while j in selected:
            j = _bisect(cum_weights, _random()*total, 0, hi)
        selected_add(j)
        result[i] = j
    return np.array(result)