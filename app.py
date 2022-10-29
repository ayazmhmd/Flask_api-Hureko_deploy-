#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_percentage_error, mean_squared_error, roc_auc_score, log_loss, precision_recall_fscore_support, mean_absolute_error, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np


# In[ ]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[ ]:


@app.route('/')
def home():
    return render_template('index.html')


# In[ ]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee status is {}'.format(output))


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




