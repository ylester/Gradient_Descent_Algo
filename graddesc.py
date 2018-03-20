from numpy import zeros, ones
import numpy as np
import pandas as pd
from pylab import scatter, show, title
import pylab
import math

##Import the data here
file = "A4dataset.csv"
df = pd.read_csv(file)

x = np.array(df['0'])
y = np.array(df['0.92893'])

#number of training samples
m = len(df)

#Adding a column of ones to x
xi = ones(shape=(m, 2))
xi[:, 1] = x

#Initialize theta parameters
theta = zeros(shape=(2, 1))
t0 = theta[0]
t1 = theta[1]

#Some gradient descent settings
iterations = 1500
alpha = 0.01


def gradientDesc(alpha, x, y, t0, t1, ep=0.0001, max_iter=10000):

    # Total Cost Error Function
    j = (sum([((t0 * math.exp((t1 * x[i]) + 1)) - y[i]) ** 2 / (2 * m) for i in range(m)]))

    for i in range(m):
        #for each training sample
        hyp = (t0 * math.exp((t1 * x[i]) + 1))
        grad0 = (1.0 / m) * sum([(hyp - y[i]) * hyp for i in range(m)])
        grad1 = (1.0 / m) * sum([((hyp - y[i]) * (hyp)*x[i]) for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        # update theta
        t0 = temp0
        t1 = temp1

    return (t0, t1, j)


gd = gradientDesc(alpha, x, y, t0=0.49, t1=1.3, max_iter=1000)

yp = []
for i in range(m):
    # y_predict = gd[0] * math.exp((gd[1] * x[i]) +1)
    y_predict = math.exp(x[i]+ 0.3)
    yp.append(y_predict)


ypred = np.array(yp)
print(ypred)

pylab.plot(x, y, 'o')
pylab.plot(x, ypred, 'k-')
pylab.title("Yesha Lester A4 Gradient Descent")
pylab.ylim([-1,25])
pylab.show()
print("Done!")



#Data Linarization
new_y = []

for i in range(m):
    ylog = math.log10(y[i])
    new_y.append(ylog)


y2 = np.array(new_y)
yp2 = []
gd2 = gradientDesc(alpha, x, y2, t0=1, t1=5, max_iter=1000)

for i in range(m):
    y_predict = math.log10(gd[0]) + (gd[1]*x[i]) + 1
    yp2.append(y_predict)

yp2array = np.array(yp2)
pylab.plot(x, y2, 'o')
pylab.plot(x, yp2array, 'k-')
pylab.ylim([-1,2])
pylab.title("Yesha Lester Gradient Descent")
pylab.show()
