import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

accuracies = []

for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin-data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    # aqui nós retiramos a coluna dos id's pois não será necessária
    # substituimos ? por -99999 para não ter valores inválidos
    # objetivo é prever se o caso é maligno ou benigno (class)

    X = np.array(df.drop(['class'], 1))#vetores de teste
    y = np.array(df['class'])#vetores dos resultados

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    accuracies.append(accuracy)

print("Average accuracy: ", sum(accuracies)/len(accuracies))