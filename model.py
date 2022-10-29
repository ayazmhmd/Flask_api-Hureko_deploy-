#!/usr/bin/env python
# coding: utf-8

# In[101]:


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

# In[65]:


df = pd.read_csv('Bondora_raw.csv')


# In[66]:


df.head(2)


# In[67]:


df.isnull().sum()


# In[68]:


# To show all the rows of pandas dataframe
percent_missing = df.isnull().sum() * 100 / len(df)
round_percent_missing=round(percent_missing,0)
print(round_percent_missing.to_string())


# In[69]:


show_percentage_greater_than_40=(round_percent_missing>40)
print(show_percentage_greater_than_40.to_string())


# In[70]:


show_percentage_greater_than_40


# In[71]:


df.columns


# In[72]:


col_null_greater_than_40_percent=['StageActiveSince','ContractEndDate','NrOfDependants','EmploymentPosition','WorkExperience','WorkExperience','PlannedPrincipalTillDate','CurrentDebtDaysPrimary',
                                  'DebtOccuredOn','CurrentDebtDaysSecondary','DebtOccuredOnForSecondary','DefaultDate','PlannedPrincipalPostDefault','PlannedInterestPostDefault',
                                  'EAD1','EAD2','PrincipalRecovery','InterestRecovery','RecoveryStage','EL_V0','Rating_V0','EL_V1','Rating_V1',
                                  'Rating_V2','ActiveLateCategory','CreditScoreEsEquifaxRisk','CreditScoreFiAsiakasTietoRiskGrade','CreditScoreEeMini',
                                  'PrincipalWriteOffs','InterestAndPenaltyWriteOffs','PreviousEarlyRepaymentsBefoleLoan','GracePeriodStart',
                                  'GracePeriodEnd','NextPaymentDate','ReScheduledOn','PrincipalDebtServicingCost','InterestAndPenaltyDebtServicingCost','ActiveLateLastPaymentCategory']


# In[73]:


df = df.drop(col_null_greater_than_40_percent,axis=1)


# In[74]:


# To show all the rows of pandas dataframe
percent_missing = df.isnull().sum() * 100 / len(df)
round_percent_missing=round(percent_missing,0)
print(round_percent_missing.to_string())


# In[75]:


df.isnull().sum()


# In[76]:


df.shape


# In[77]:


df.dropna(axis=0,inplace=True)


# In[78]:


df.shape


# In[79]:


df.duplicated().sum()


# In[80]:


df['County'].drop


# In[82]:


df.columns


# In[83]:


features=['VerificationType','Amount','Interest','DebtToIncome','PrincipalPaymentsMade','AmountOfPreviousLoansBeforeLoan',
          'PreviousEarlyRepaymentsCountBeforeLoan','LoanDuration','NrOfScheduledPayments','Status']


# In[84]:


df_new=df[features]


# In[85]:


#Checking distribution of categorical variables
categorical_df = df_new.select_dtypes('object')
categorical_df.info()


# In[86]:


df_new['Status'].value_counts()


# In[87]:


order_label={"Late":0,"Current":1,"Repaid":2}
df_new['Status']=df_new['Status'].map(order_label)


# In[88]:


df_new['Status'].value_counts()


# In[89]:


y = df_new['Status']
y


# In[90]:


independant_features = ['VerificationType','Amount','Interest','DebtToIncome','PrincipalPaymentsMade','AmountOfPreviousLoansBeforeLoan',
                        'PreviousEarlyRepaymentsCountBeforeLoan','LoanDuration','NrOfScheduledPayments']
X = df_new[independant_features]
X


# In[91]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[92]:


pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)),
                     ('lr_classifier',LogisticRegression(random_state=0))])


# In[93]:


pipeline_randomforest=Pipeline([('scalar3',StandardScaler()),
                     ('pca3',PCA(n_components=2)),
                     ('rf_classifier',RandomForestClassifier())])


# In[94]:


## LEts make the list of pipelines
pipelines = [pipeline_lr, pipeline_randomforest]


# In[95]:


best_accuracy=0.0
best_classifier=0
best_pipeline=""


# In[96]:


# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'RandomForest'}

# Fit the pipelines
for pipe in pipelines:
	pipe=pipe.fit(X_train, y_train)


# In[97]:


for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))


# In[98]:


for i,model in enumerate(pipelines):
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))


# In[99]:


pred = pipe.predict(X_test)
print('test accuracy = ', round(accuracy_score(y_test, pred)*100, 2), '%')


# In[100]:


print(classification_report(y_test, pred, digits=3))


# In[ ]:

# Saving model to disk
pickle.dump(pipe, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6,2,3,5,6,2,5]]))



# %%
