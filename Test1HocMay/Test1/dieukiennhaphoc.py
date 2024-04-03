# Thư viện xử lý ma trận, mảng,
import numpy as np;
import pandas as pd
#Load dữ liệu
df_nhaphoc = pd.read_csv("nhaphoc1.csv")
print ("Ma Trận: ")
print (df_nhaphoc.shape)
print ("Hiển Thị Dữ Liệu" )
print (df_nhaphoc.head(10))

list = [ 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Chance of Admit ']
X = df_nhaphoc[list]#đặc trưng biến độc lập
print (X.head(10))
y = df_nhaphoc['GRE Score']# biến phụ thuộc
print (y.head(10))
#Giảm kích thước tuyến tính bằng cách sử dụng Phân rã giá trị đơn lẻ của dữ liệu để chiếu dữ liệu đó vào không gian có chiều thấp hơn.
# Dữ liệu đầu vào được căn giữa nhưng không được chia tỷ lệ cho từng tính năng trước khi áp dụng SVD.
from sklearn.decomposition import PCA
X = PCA(1).fit_transform(X)#Chỉnh mô hình với X và áp dụng giảm kích thước trên X.
print (X[:10])
X.shape
# Dùng model_selection helper để chia dữ liệu
from sklearn.model_selection import train_test_split
#Phân loại dữ liệu train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)# Tách tập dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn import linear_model# mô-đun được sử dụng để thực hiện hồi quy tuyến tính

regr = linear_model.LinearRegression().fit(X_train, y_train)#Bình phương nhỏ nhất thông thường Hồi quy tuyến tính
y_pred = regr.predict(X_test)# thực hiện dự đoán trên tập dữ liệu thử nghiệm

from sklearn.metrics import mean_squared_error, r2_score#hàm điểm hồi quy.
# Tính toán các dự đoán
# Các hệ số

print('Hệ số: \n', regr.coef_)
print('Bias: \n', regr.intercept_)#Bias: nghĩa là độ lệch, biểu thị sự chênh lệch giữa giá trị trung bình mà mô hình dự đoán và giá trị thực tế của dữ liệu.
# Sai số bình phương trung bình
print('Sai số bình phương trung bình: %.2f'% mean_squared_error(y_test, y_pred))
# Hệ số xác định:1 là dự đoán hoàn hảo
print('Hệ số xác định: %.2f'% r2_score(y_test, y_pred))# tỷ lệ của phương sai trong biến phụ thuộc có thể dự đoán được từ (các) biến độc lập

import matplotlib.pyplot as plt #pyplot giúp mô tả dữ liệu thông qua các biểu đồ trực quan
# Vẽ đồ thị để biểu diễn dữ liệu
plt.scatter(X_train, y_train,  color='green')
plt.scatter(X_train, regr.predict(X_train),  color='red')
plt.scatter(X_test[:10,:], y_test[:10],  color='black')
plt.title('Hồi quy tuyến tính cho Điểm GRE')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
plt.plot([min(y_test), max(y_test)],[min(y_pred),max(y_pred)])
# Kiểm tra mức độ lỗi của model (Mean Squared Error)
plt.scatter(y_test, y_pred,  color='red')
plt.title('Compare')#Đối chiếu
# In giá trị y test thực tế
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()
