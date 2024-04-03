import numpy as np
import pandas as pd 


df_tbtichluy = pd.read_csv("tbtichluy.csv")

#df_tbtichluy = df_tbtichluy.sample(frac=1)
print (df_tbtichluy.shape)
print (df_tbtichluy.head(10))

list = ["GRE Score","TOEFL Score","University Rating","Chance of Admits"]
X = df_tbtichluy[list]
print (X.head(10))

y = df_tbtichluy["CGPA"]
print (y.head(10))

from sklearn.decomposition import PCA
X = PCA(1).fit_transform(X) 
print (X[:10])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=35)

from sklearn import linear_model
regr = linear_model.LinearRegression().fit(X_train, y_train)

y_pred = regr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
# Các hệ số
print('Hệ số: \n', regr.coef_)
print('Bias: \n', regr.intercept_)#giá trị lệch đề bù đắp những sai số
# Sai số bình phương trung bình
print('Sai số bình phương trung bình: %.2f'% mean_squared_error(y_test, y_pred))
# Hệ số xác định:1 là dự đoán hoàn hảo
print('Hệ số xác định: %.2f'% r2_score(y_test, y_pred))

import matplotlib.pyplot as plt
plt.scatter(X_train, y_train,  color='green')
plt.scatter(X_train, regr.predict(X_train),  color='red')
plt.scatter(X_test[:10,:], y_test[:10],  color='black')
plt.title('Linear regression for trung bình tích lũy')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


plt.plot([min(y_test), max(y_test)],[min(y_pred),max(y_pred)])
plt.scatter(y_test, y_pred,  color='red')
plt.title('Compare')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()
