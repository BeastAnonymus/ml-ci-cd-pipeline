import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/iris.csv')
x= data.drop('species', axis=1)
y= data['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = joblib.load('model/model.pkl')

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:2f}')