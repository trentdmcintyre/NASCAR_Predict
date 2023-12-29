#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


NASCAR_Data = pd.read_csv(r"C:\Users\trent\Desktop\Career.Work.DataProj\Hobby Analytics Files\NASCAR With Dummy Data.csv")
print(NASCAR_Data[:][0:5])


# In[3]:


# LogReg without test_train_split Top 10

# Fitting the logistic regression model
X = NASCAR_Data[['Start','Hendrick Motorsports','JTG Daugherty Racing','Kaulig Racing','Petty GMS Motorsports','Richard Childress Racing','Spire Motorsports','Trackhouse Racing','Front Row Motorsports','Live Fast Motorsports','RFK Racing','Rick Ware Racing','Stewart-Haas Racing','Team Penske','Wood Brothers Racing','23XI Racing','Joe Gibbs Racing','Beard Motorsports','NY Racing Team','The Money Team Racing','MBM Motorsports','Team Hezeberg powered by','Reaume Brothers Racing','Intermediate','Superspeedway','Short Track','Road Course','Speedway']]
y10 = np.ravel(NASCAR_Data[['Top 10']])
NASCAR_Logreg = linear_model.LogisticRegression(max_iter = 100000)
NASCAR_Logreg.fit(X, y10)

# New Data Frame for only the predictors
predictors = NASCAR_Data[['Start','Hendrick Motorsports','JTG Daugherty Racing','Kaulig Racing','Petty GMS Motorsports','Richard Childress Racing','Spire Motorsports','Trackhouse Racing','Front Row Motorsports','Live Fast Motorsports','RFK Racing','Rick Ware Racing','Stewart-Haas Racing','Team Penske','Wood Brothers Racing','23XI Racing','Joe Gibbs Racing','Beard Motorsports','NY Racing Team','The Money Team Racing','MBM Motorsports','Team Hezeberg powered by','Reaume Brothers Racing','Intermediate','Superspeedway','Short Track','Road Course','Speedway']]

# Making Predictions
NASCAR_Predict_10P = NASCAR_Logreg.predict_proba(predictors)
NASCAR_Predict_10O = NASCAR_Logreg.predict(predictors)

print(NASCAR_Predict_10P)
print(NASCAR_Predict_10O)


# In[4]:


# Adding Top 10 probability as a column
NASCAR_Data[['Not Top 10 Probability','Top 10 Probability']] = NASCAR_Predict_10P
NASCAR_Data['Top 10 Predicted'] = NASCAR_Predict_10O
print(NASCAR_Data)


# In[5]:


# Running again to for Win Probability

# Fitting the logistic regression model
X = NASCAR_Data[['Start','Hendrick Motorsports','JTG Daugherty Racing','Kaulig Racing','Petty GMS Motorsports','Richard Childress Racing','Spire Motorsports','Trackhouse Racing','Front Row Motorsports','Live Fast Motorsports','RFK Racing','Rick Ware Racing','Stewart-Haas Racing','Team Penske','Wood Brothers Racing','23XI Racing','Joe Gibbs Racing','Beard Motorsports','NY Racing Team','The Money Team Racing','MBM Motorsports','Team Hezeberg powered by','Reaume Brothers Racing','Intermediate','Superspeedway','Short Track','Road Course','Speedway']]
yW = np.ravel(NASCAR_Data[['Win']])
NASCAR_Logreg = linear_model.LogisticRegression(max_iter = 100000)
NASCAR_Logreg.fit(X, yW)

# New Data Frame for only the predictors
predictors = NASCAR_Data[['Start','Hendrick Motorsports','JTG Daugherty Racing','Kaulig Racing','Petty GMS Motorsports','Richard Childress Racing','Spire Motorsports','Trackhouse Racing','Front Row Motorsports','Live Fast Motorsports','RFK Racing','Rick Ware Racing','Stewart-Haas Racing','Team Penske','Wood Brothers Racing','23XI Racing','Joe Gibbs Racing','Beard Motorsports','NY Racing Team','The Money Team Racing','MBM Motorsports','Team Hezeberg powered by','Reaume Brothers Racing','Intermediate','Superspeedway','Short Track','Road Course','Speedway']]

# Making Predictions
NASCAR_Predict_WP = NASCAR_Logreg.predict_proba(predictors)
NASCAR_Predict_WO = NASCAR_Logreg.predict(predictors)

print(NASCAR_Predict_WP)
print(NASCAR_Predict_WO)


# In[23]:


# Adding Win probability as a column
NASCAR_Data[['Not Win Probability','Win Probability']] = NASCAR_Predict_WP
NASCAR_Data['Win Predicted'] = NASCAR_Predict_WO

print(NASCAR_Data)


# In[24]:


# Exporting 2022-2023 Season Probabilities

NASCAR_Data.to_csv(r"C:\Users\trent\Desktop\Career.Work.DataProj\Hobby Analytics Files\NASCAR_NextGen_With_Proba.csv")


# In[11]:


# Training the models using test_train_split

X_train10, X_test10, y_train10, y_test10 = train_test_split(X,y10,test_size=.2)
X_trainW, X_testW, y_trainW, y_testW = train_test_split(X,yW,test_size=.2)


# In[15]:


# Fitting the models

# Top 10
NASCAR_Logreg10 = linear_model.LogisticRegression(max_iter = 100000)
NASCAR_Logreg10.fit(X_train10, y_train10)

# Win
NASCAR_LogregW = linear_model.LogisticRegression(max_iter = 100000)
NASCAR_LogregW.fit(X_trainW, y_trainW)


# In[18]:


# Testing the Top 10 model
NASCAR_Logreg10.predict(X_test10)


# In[21]:


# Testing the Top 10 model probability
NASCAR_Logreg10.predict_proba(X_test10)


# In[19]:


# Testing the Win Model
NASCAR_LogregW.predict(X_testW)


# In[22]:


# Testing the Win Model for Probability
NASCAR_LogregW.predict_proba(X_testW)


# In[64]:


# Testing a hypothetical 2024 Daytona 500

# Import Data
FakeDaytona = pd.read_csv(r"C:\Users\trent\Desktop\Career.Work.DataProj\Hobby Analytics Files\Hypothetical 2024 Daytona 500 3.csv")
print(FakeDaytona[:][0:5])


# In[69]:


XD10 = FakeDaytona[['Start','Hendrick Motorsports','JTG Daugherty Racing','Kaulig Racing','Petty GMS Motorsports','Richard Childress Racing','Spire Motorsports','Trackhouse Racing','Front Row Motorsports','Live Fast Motorsports','RFK Racing','Rick Ware Racing','Stewart-Haas Racing','Team Penske','Wood Brothers Racing','23XI Racing','Joe Gibbs Racing','Beard Motorsports','NY Racing Team','The Money Team Racing','MBM Motorsports','Team Hezeberg powered by','Reaume Brothers Racing','Intermediate','Superspeedway','Short Track','Road Course','Speedway']]

# Making Predictions
FakeDaytona_Predict_10P = NASCAR_Logreg10.predict_proba(XD10)

print(FakeDaytona_Predict_10P)


# In[71]:


# Add back into Workbook

FakeDaytona[['Not Top 10 Prob','Top 10 Prob']] = FakeDaytona_Predict_10P

print(FakeDaytona)


# In[72]:


# Export back to Excel

FakeDaytona.to_csv(r"C:\Users\trent\Desktop\Career.Work.DataProj\Hobby Analytics Files\Hypothetical 2024 Daytona 500 with proba.csv")

