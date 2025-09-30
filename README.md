# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program & Output:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: INDHUMATHI L
RegisterNumber:  212224220037
*/
```

```
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("drive/MyDrive/ML/Salary.csv")
data.head()
```
<img width="865" height="320" alt="image" src="https://github.com/user-attachments/assets/60f2e69e-84c5-48d4-abbf-484c5382f2f1" />

```
data.info()
```
<img width="425" height="242" alt="image" src="https://github.com/user-attachments/assets/9e770700-6442-4d1c-a0a6-0f6fcde207d1" />

```
data.isnull().sum()
```
<img width="209" height="239" alt="image" src="https://github.com/user-attachments/assets/214209b9-d364-4a75-8166-d6fdecbcaa5c" />

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
<img width="283" height="256" alt="image" src="https://github.com/user-attachments/assets/b5e7cf6a-346a-4096-b393-df6a5221f3d6" />

```
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
```
<img width="228" height="290" alt="image" src="https://github.com/user-attachments/assets/66e04f4c-6270-47a0-a5bc-6d8ae77c2ccc" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
```
<img width="413" height="42" alt="image" src="https://github.com/user-attachments/assets/d1b032ab-69c0-4df6-b308-2456d0737e99" />

```
from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
r2
```
<img width="457" height="33" alt="image" src="https://github.com/user-attachments/assets/0f9924fa-7364-43a0-af4c-7cc42aaaaeff" />

```
dt.predict([[5,6]])
```
<img width="1736" height="78" alt="image" src="https://github.com/user-attachments/assets/fe92a54f-8b4f-4a9a-82d6-da0c62373ec7" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
