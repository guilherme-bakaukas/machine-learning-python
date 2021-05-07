from statistics import mean
import numpy as np

# Y = mX + b
#aqui temos um exemplo de cálculo do coeficiente angular 'm' utilizando a regressão linear

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7],dtype=np.float64)

def define_m(xs,ys):
    m=(mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs))
    return m

m = define_m(xs,ys)
print("Coeficiente angular: ", m)