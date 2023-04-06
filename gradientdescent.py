import numpy as np
import pandas as pd

npArray = np.loadtxt("train.csv", delimiter = ",")

X_test = np.loadtxt("test.csv", delimiter = ",")


X_train = npArray[: , : -1]
Y_train = npArray[: ,-1]
print(X_train.shape  )
print(Y_train.shape)

import pandas as pd 
df = pd.DataFrame(X_train)
df

for i in df.columns:
    name = str(i)+"_"+str(i)
    df[name] = df[i]**2
df.describe()
X_train = df.values
X_train.shape

df1 = pd.DataFrame(X_test)
for i in df1.columns:
    name = str(i)+"_"+str(i)
    df1[name] = df1[i]**2
df1.shape
X_test = df1.values
X_test.shape

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train , Y_train)

def gradient_descent(x,y,alpha,n):
    M = len(x[:,0])
    N = len(x[0])
    m = np.zeros(N)
    costs = []
    itr = []
    for i in range(n):
        m = step_gradient(x,y,alpha,m)
        cos = cost(x,y,m)
        costs.append(cos)
        itr.append(i)
        print(i,"Cost: ",cos)
    return m,costs,itr
    
def step_gradient(x,y,alpha,m):
    m_slope = np.zeros(len(m))
    M = len(x[:,0])
    N = len(x[0])
    
    for i in range(M):
        for j in range(N):
            m_slope[j] += (-2/M)*(y[i]-(m*x[i]).sum())*x[i][j]
    m = m - alpha*m_slope
    return m
    
def cost(x,y,m):
    cost = 0
    M = len(x[:,0])
    
    for i in range(M):
        cost += (1/M)*((y[i]-(m*x[i]).sum())**2)
    return cost
    
def predict(x,m):
    M = len(x[:,0])
    y_pred = np.zeros(M)
    
    for i in range(M):
        y_pred[i] = (m*x[i]).sum()
        
    return y_pred
    
learning_rate = 0.018
num_of_iterations =5000
m,costs,itr = gradient_descent(X_train,Y_train, learning_rate, num_of_iterations)

print(*m)

y_test_pred = predict(X_test,m)
np.savetxt("Gryk.csv",y_test_pred)
