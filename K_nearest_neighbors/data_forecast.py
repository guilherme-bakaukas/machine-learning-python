import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k

    return vote_result, confidence

accuracies = []

for i in range(25):
    df = pd.read_csv("breast-cancer-wisconsin-data.txt")
    df.replace('?', -9999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)  # shuffle no data

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]  # primeiros 0.8 treinados
    test_data = full_data[-int(test_size * len(full_data)):]  # 0.2 ultimos para teste

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
        # i[-1] indica o ultimo elemento "class" que irá indicar se é 2 ou 4
        # damos append do vetor sem a "class" em 2 ou 4

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    #aqui vamos utilizar os dados ja obtidos (train_set) e iremos submeter os dados de teste
    #através dos dados de treino, vamos determinar o grupo de cada elemento de teste em test_set

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            # usar esse código para analisar a confiança do resultado que não foi correto
            #else:
                #print("Wrong prediction, with confidence: ",confidence)
            total += 1

    accuracies.append(correct/total)

print("Average accuracy: ", sum(accuracies)/len(accuracies))