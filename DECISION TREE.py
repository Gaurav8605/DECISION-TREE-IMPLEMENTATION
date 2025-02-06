
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('iris.csv')

df.info()

df

L=['sepal.length','sepal.width','petal.length','petal.width']
x=df[L]
y=df['variety']

x,y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)

from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(criterion='entropy')
DT.fit(x_train,y_train)

pre=DT.predict(x_test)

print(pre)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pre))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pre))

from sklearn.metrics import classification_report
print(classification_report(y_test,pre))

from sklearn import tree

tree.plot_tree(DT,filled=True)

