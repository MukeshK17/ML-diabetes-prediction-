import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# %matplotlib inline


# importing data
data = pd.read_csv('diabetes.csv')
print(data.isnull().sum())

data["Glucose"] = data["Glucose"].replace(0,data["Glucose"].mean())
data["BloodPressure"] = data["BloodPressure"].replace(0,data["BloodPressure"].mean())
data["SkinThickness"] = data["SkinThickness"].replace(0,data["SkinThickness"].mean())
data["Insulin"] = data["Insulin"].replace(0,data["Insulin"].mean())
data["BMI"] = data["BMI"].replace(0,data["BMI"].mean())


##pregnanicies vs the target output
a= sns.kdeplot(data.Pregnancies[data.Outcome==0],color='red',fill=True)
b= sns.kdeplot(data.Pregnancies[data.Outcome==1],color='blue',fill=True)
plt.title('Pregnancies vs Outcome')
plt.show()


X = data.drop('Outcome',axis=1)
y = data['Outcome']

## correlatuion matrix
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm',fmt='.1f',linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()



from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,y,test_size =0.33,random_state=42)

##KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


training_accuracy =[]
test_accuracy =[]

for K in range(1,40):
  Knn = KNeighborsClassifier(K)
  Knn.fit(X_train, y_train)

  training_accuracy.append(Knn.score(X_train, y_train))
  test_accuracy.append(Knn.score(X_test, y_test))
print("training_accuracy:", training_accuracy[-2])
print("test_accuracy:", test_accuracy[-2])


new_data = [[6, 250, 5, 500, 0, 26, 0.627, 50]]
plt.plot(range(1,40),training_accuracy)
plt.plot(range(1,40),test_accuracy)
plt.grid()
prediction = Knn.predict(new_data)
print("Prediction_value:", *prediction)



## Decision Tree Classifier
new_data_1 = [[6, 20, 5, 5, 0, 26, 0.627, 50]]
dtree = DecisionTreeClassifier(random_state=42,max_depth=3)
dtree.fit(X_train,y_train)
print("training_accuracy:",dtree.score(X_train,y_train))
print("testing_accuracy:",dtree.score(X_test,y_test))
print("prediction_value:", *dtree.predict(new_data_1))


##MLP classifier
new_data_2 = [[6, 20, 5, 5, 0, 26, 0.627, 0]]
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state = 42)
mlp.fit(X_train, y_train)
print("training_accuracy:",mlp.score(X_train,y_train))
print("testing_accuracy:",mlp.score(X_test, y_test))
print("prediction_value:",*mlp.predict(new_data_2))


## Standardized MLP
new_data_3 = [[6, 0, 5, 5, 0, 26, 0.627, 10]]
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

mlp_ = MLPClassifier(random_state=42)
mlp_.fit(X_train_scaled, y_train)

print("training_accuracy:", mlp_.score(X_train_scaled, y_train))
print("testing_accuracy:", mlp_.score(X_test_scaled, y_test))
print("prediction_value:", *mlp_.predict(new_data_3))