import math
import quandl
import numpy as np
from sklearn import preprocessing,model_selection
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df=quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]#aqui definimos as colunas que queremos ver

df['HL_PCT']=(df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0#definimos novas colunas

df['PCT_change']=(df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]#elencamos as colunas que queremos ver

forecast_col='Adj. Close'
df.fillna(-99999, inplace=True)
#preenche valores NaN(Not a Number, valor numérico indefinido) com o valor -99999

forecast_out=int(math.ceil(0.01*len(df)))
#iremos separar 1% do dataframe para prever seus resultados

df['label']=df[forecast_col].shift(-forecast_out)
# a função shift desloca os valores do Adj. Close para cima em 1% do tamanho do df
# assim é possível ter uma previsão do valor em um período de 1% do tamanho total, a coluna label indica o Adj. Close futuro.

x=np.array(df.drop(['label'],1))
#x é o array com os elementos das demais colunas sem o label que será o previsto

x=preprocessing.scale(x)#padroniza os valores para uma mesma escala
x=x[:-forecast_out]#apenas será testado e treinado os dados antes do 1% final
x_lately=x[-forecast_out:]#indica os últimos 1% dos dados, utilizados para gerar a previsão

df.dropna(inplace=True)
#exclui os valores vazios(valores do label vazios no final devido ao shift)
y=np.array(df['label'])#criamos um array com os valores da coluna label (previsão)

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)
#divide o array em teste e treino, 0,2 em teste e portanto 0,8 em treino
#os pontos são embaralhados
#regressão linear, pontos nas coordenadas x e y, assim, separamos pontos de teste e de treino

#alguns links sobre esse passo:
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#https://www.youtube.com/watch?v=b0L47BeklTE

#CÓDIGO QUE FOI RODADO NA PRIMEIRA VEZ (salvamos com pickle para não treinar o modelo todas as vezes):

#clf=LinearRegression()#regressão linear com os dados obtidos
#clf.fit(x_train,y_train)
#with open('linearregression.pickle','wb') as f:
#   pickle.dump(clf, f)

#salvamos o clf para n ser necessário treiná-lo a todo uso (podemos achar na pasta)
#após rodar uma vez o código acima, não precimos mais dele, pois está salvo

pickle_in = open('linearregression.pickle', 'rb')
clf=pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)
forecast_set=clf.predict(x_lately)
#aqui realizamos a previsão com os últimos dados em x, assim teremos os valores para esses próximos dias

print("Dados previstos: ",forecast_set)
print("Dias de revisão: ", forecast_out)

print("Acurácia: ", accuracy)#precisão do modelo

# Agora vamos plotar um gráfico para visualizar a previsão relizada
df['Forecast']=np.nan

last_date=df.iloc[-1].name #último dia
last_unix=last_date.timestamp()
one_day=86400 #segundos em um dia
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
    # define nan para as demais colunas que não sejam forecast

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
