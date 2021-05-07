# K-nearest neighbors é um conceito de determinar a qual grupo um determinado ponto se encaixa
# Fazemos isso analisando os k pontos mais próximos a ele
# K ímpar é conveniente pois podemos determinar o grupo
# Exemplo K=3, 2 pontos mais próximos no grupo 1 e o terceiro ponto no grupo 2
# Neste caso, é mais provável que o ponto seja do grupo 1

from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

def k_nearest_neighbors(data, predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common)
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset,new_features, k=3)
print(result)

[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1], color=result)
plt.show()