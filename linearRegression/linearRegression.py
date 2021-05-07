from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# Y = mX + b
# Aqui temos um exemplo de cálculo do coeficiente angular 'm' e coeficiente linear 'b' utilizando a regressão linear

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7],dtype=np.float64)

def define_coeficients(xs,ys):
    m = (mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b = define_coeficients(xs,ys)
print("Coeficiente angular: ", m)
print("Coeficiente linear: ", b)

regression_Y = m*xs + b

plt.plot(xs, regression_Y, color="blue")
plt.scatter(xs, ys, color='red')
plt.show()
