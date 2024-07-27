from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#read csv
df  =  pd.read_csv('D:/projects/Trainning/ML/Assignment 2/Mall_Customers.csv')
cluster = 5 # This value is observed using elbo method
set_ = df[['Annual Income (k$)','Spending Score (1-100)']]
test,train= train_test_split(set_,test_size=0.5,random_state=42)

#Cluster 
kmeans = KMeans(n_clusters=cluster)
kmeans.fit_predict(train)

train[f'KMeans_{cluster}']=kmeans.labels_

# plt.scatter(x=train['Annual Income (k$)'],y=train['Spending Score (1-100)'],c=train[f'KMeans_{cluster}'],label=train[f'KMeans_{cluster}'])
# plt.title("Train Result")
# plt.show()

predict = kmeans.predict(test)
test[f'KMeans_{cluster}']=predict


plt.scatter(x=test['Annual Income (k$)'],y=test['Spending Score (1-100)'],c=test[f'KMeans_{cluster}'],label=test[f'KMeans_{cluster}'])
plt.title("Test Result")
plt.show()






