# K-Nearest-Neighbors
Esse modelo trata de uma previsão que leva em conta a proximadade do ponto previsto
A ideia é termos grupos diferentes para um determinado vetor de informações
### Exemplo:
Vetor1 = [1,2,3,4,5,6] pertence ao grupo 1
Vetor2 = [2,3,9,8,2,3] pertence ao grupo 2
Utilizando um vetor teste de [3,1,3,6,7,8] queremos determinar seu grupo...

A técnica resume-se em determinar um numero K de pontos a serem analisados de acordo com sua proximidade
e a partir deles determinar o grupo que o ponto teste pertence. Por exemplo:

Se determinamos um k=3, encontramos os 3 pontos mais próximos e a partir deles indicamos o grupo ao qual o ponto pertence.
Caso 2 pontos mais próximos sejam grupo 1 e o ponto restante seja do grupo 2, sabemos que a probabilidade de que o ponto seja do grupo 1 é maior (aproximadamente 66,67%).

### Exemplo analisado
Nos arquivos da pasta, visamos analisar dados de cancer de mama e determinar se são malignos ou benignos (2 grupos) atraves dos dados
dos pacientes.
Para isso fazemos de duas maneiras, train test split do sklearn e neighbors.KNeighborsClassifier()
e através da distância euclideana entre os pontos feita na mão.

No arquivo menu é possível ter mais informações sobre os dados utilizados para a previsão.

Site que contém os dados: https://archive.ics.uci.edu/ml/datasets.php
