# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Churn.csv')

X = dataset.iloc[:,:5]

y = dataset.iloc[:,5]

#Splitting Training and Test Set

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model2.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
print(model.predict(X.iloc[0,:].values.reshape(1,-1)))