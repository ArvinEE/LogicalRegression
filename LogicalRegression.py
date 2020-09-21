import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = 'C:/Users/Arvin/Desktop/ex2data1.txt'
data = pd.read_csv(path, names=['Exam 1', 'Exam 2', 'Accepted'])
data.head()
fig,ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1]['Exam 2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='exam1',
          ylabel='exam2')
plt.show()

def get_Xy(data):
    data.insert(0, 'ones', 1)
    X_ = data.iloc[:, 0:-1]
    X = X_.values

    y_ = data.iloc[:, -1]
    y = y_.values.reshape(len(y_), 1)

    return X, y
X,y = get_Xy(data)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunction(X, y, theta):
    A = sigmoid(X @ theta)

    first = y * np.log(A)
    second = (1 - y) * np.log(1 - A)

    return -np.sum(first + second) / len(X)
theta = np.zeros((3,1))
theta.shape
cost_init = costFunction(X,y,theta)
print(cost_init)


def gradientDescent(X, y, theta, iters, alpha):
    m = len(X)
    costs = []

    for i in range(iters):
        A = sigmoid(X @ theta)
        theta = theta - (alpha / m) * X.T @ (A - y)
        cost = costFunction(X, y, theta)
        costs.append(cost)
        if i % 1000 == 0:
            print(cost)
    return costs, theta

alpha = 0.0004
iters=2000000
costs,theta_final =  gradientDescent(X,y,theta,iters,alpha)

theta_final
def predict(X, theta):
    prob = sigmoid(X @ theta)

    return [1 if x >= 0.5 else 0 for x in prob]
y_ = np.array(predict(X,theta_final))
y_pre = y_.reshape(len(y_),1)

acc  = np.mean(y_pre == y)

print(acc)
coef1 = - theta_final[0,0] / theta_final[2,0]
coef2 = - theta_final[1,0] / theta_final[2,0]
x = np.linspace(20,100,100)
f = coef1 + coef2 * x

fig,ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1]['Exam 2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='exam1',
          ylabel='exam2')

ax.plot(x,y,c='g')
plt.show()