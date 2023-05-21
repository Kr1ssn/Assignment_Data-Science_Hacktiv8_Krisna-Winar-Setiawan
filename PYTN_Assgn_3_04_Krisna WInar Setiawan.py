#!/usr/bin/env python
# coding: utf-8

# ## Project Overview
# 
# Project Overview
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
# 
# There are four datasets:
# 
# bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014] <br>
# bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.<br>
# bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).<br>
# bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
# The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).<br>
# 
# The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

# ## Import Libraries

# In[102]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Dataset

# In[70]:


df = pd.read_csv("bank-additional-full.csv", sep=';')
df.tail(10)


# This dataset contains:
# 
# Input variables:
# 
# # bank client data:
# 1 - age (numeric)<br>
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br>
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br>
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br>
# 5 - default: has credit in default? (categorical: 'no', 'yes', 'unknown')<br>
# 6 - housing: has housing loan? (categorical: 'no', 'yes', 'unknown')<br>
# 7 - loan: has personal loan? (categorical: 'no', 'yes', 'unknown')<br>
# # related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: 'cellular', 'telephone')<br>
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br>
# 10 - day_of_week: last contact day of the week (categorical: 'mon', 'tue', 'wed', 'thu', 'fri')<br>
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y 
# is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.<br>
# 
# # other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br>
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br>
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)<br>
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure', 'nonexistent', 'success')<br>
# # social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)<br>
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)<br>
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)<br>
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)<br>
# 20 - nr.employed: number of employees - quarterly indicator (numeric)
# 
# ## Output variable (desired target):
# 21 - y - has the client subscribed a term deposit? (binary: 'yes', 'no')<br>

# In[71]:


df.columns


# In[72]:


df.info()


# In[73]:


df.isnull().sum()


# There are no missing values

# In[74]:


df.rename(columns={"default":"credit", "y":"subscribed"}, inplace=True)


# In[75]:


# mengecek keseimbangan jumlah label/output dataset
plt.rcParams['figure.figsize']=(15,6)
plt.subplot(121)
plt.title("Checking Balance of Dataset Label")
ax = sns.countplot(x='subscribed', data=df)
for i in ax.patches:
    ax.annotate(format(i.get_height(),'0.1f'), (i.get_x() + i.get_width()/2.,i.get_height()),
               ha='center', va='center', xytext=(0,7), textcoords='offset points')
    
plt.subplot(122)
plt.title("Percentage of Subcribed Status")
subscribed_values_count = df['subscribed'].value_counts()
subscribed_size = subscribed_values_count.values.tolist()
subscribed_labels = 'No', 'Yes'
colors=['red', 'lightgreen']
pcs, texts, autotexts = plt.pie(subscribed_size, labels=subscribed_labels, colors=colors,
                             autopct='%2.2f%%', shadow=True, startangle=150)

for text, autotext in zip(texts, autotexts):
    text.set_fontsize(13)
    autotext.set_fontsize(13)

plt.axis('equal')
plt.show()
    



# Dari visualisasi data diatas dapat kita lihat bahwa jumlah label dataset yang digunakan adalah imbalanced, hal ini secara implisit akan mempengaruhi kinerja model yang akan dihasilkan. oleh karena itu harus dilakukan proses balancing data.

# In[76]:


# Categorical columns exploration

categorical_cols = ['job','marital','education','credit','housing',
                   'loan','contact','month','poutcome']
fig, ax = plt.subplots(3,3, sharex=False, sharey=False, figsize=(25,20))
count = 0
for cat_col in categorical_cols:
    value_count = df[cat_col].value_counts()
    ax_x = count//3
    ax_y = count%3
    x_range = np.arange(0, len(value_count))
    ax[ax_x, ax_y].bar(x_range, value_count.values, tick_label=value_count.index)
    ax[ax_x, ax_y].set_title(f"Bar plot of {cat_col}")
    
    for i in ax[ax_x, ax_y].get_xticklabels():
        i.set_rotation(90)
    
    count+=1
plt.show()
    


# In[77]:


# Numerical columns exploration
num_cols = ['duration','campaign','pdays','previous','emp.var.rate',
            'cons.conf.idx','euribor3m','nr.employed']

fig, ax = plt.subplots(3,3, sharex=False, sharey=False, figsize=(20,15))
count = 0
for num_col in num_cols:
    ax_x = count//3
    ax_y = count%3
    
    ax[ax_x, ax_y].hist(df[num_col])
    ax[ax_x, ax_y].set_title(f"histogram of {num_col}")
    count+=1
plt.show()


# In[78]:


df['subscribed'].replace({'yes':1, 'no':0}, inplace=True)


# In[79]:


corr = df.corr()
print(corr['subscribed'].sort_values(axis=0, ascending=True))


# In[80]:


# drop features yang memiliki korelasi lebih sedikit dan tidak relevan dengan proses klasifikasi
df.drop(columns=['nr.employed','pdays','euribor3m','emp.var.rate',
    'cons.price.idx','day_of_week','cons.conf.idx','contact','month'],axis=0, inplace=True)


# In[81]:


df


# In[82]:


encoder = LabelEncoder()
col = ['marital','credit','housing','loan']

for i in col:
    df[i] = encoder.fit_transform(df[i])


# In[83]:


# Encoding
cat_features = ['job','marital','education','credit','housing','loan','poutcome']
df = pd.get_dummies(df, columns=cat_features, drop_first=True)
df


# In[84]:


#Assign variable features dan label
X = df.drop(columns='subscribed', axis=1).values
y = df['subscribed'].values


# In[85]:


#splitting data scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=103)


# In[86]:


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


# In[87]:


# Logistik Regression
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
ypred = model_lr.predict(X_test)
print("Accuracy: ",metrics.accuracy_score(y_test, ypred))
print(metrics.confusion_matrix(y_test, ypred))
print(metrics.classification_report(y_test, ypred))


# In[88]:


# SUPPORT VECTOR MACHINES (SVM)
from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
ypred2=svm_model.predict(X_test)
print("Acc: ", metrics.accuracy_score(y_test, ypred2))
print(metrics.confusion_matrix(y_test, ypred2))
print(metrics.classification_report(y_test, ypred2))


# In[89]:


# K-NEAREST NEIGHBOR ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
pred_cv5= KNN.predict(X_test)
print("Accuracy score K-Nearest: ", metrics.accuracy_score(y_test, pred_cv5))
matrix5 = metrics.confusion_matrix(y_test, pred_cv5)
print(matrix5)
print(metrics.classification_report(y_test, pred_cv5))


# In[90]:


# DECISION TREE
dt = DecisionTreeClassifier()

# fit model into training data
dt.fit(X_train,y_train)

# get y prediction
y_pred_dt = dt.predict(X_test)

# print score
print("Accuracy score Decision Tree = ", metrics.accuracy_score(y_test,y_pred_dt))
print(metrics.classification_report(y_test,y_pred_dt))

# confusion matrix
print(metrics.confusion_matrix(y_test,y_pred_dt))


# #### Akurasi setelah melakukan handling Imbalanced Dataset
# 

# In[91]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[92]:


X_train_smote, y_train_smote= smote.fit_resample(X_train.astype('float'),y_train)


# In[93]:


from collections import Counter
print("Before oversampling: ",Counter(y_train))
print("After oversampling: ",Counter(y_train_smote))


# In[94]:


# Logistik Regression
from sklearn.linear_model import LogisticRegression
model_lr2 = LogisticRegression()
model_lr2.fit(X_train_smote, y_train_smote)
ypred_smote = model_lr2.predict(X_test)
print("Accuracy Score Logistik Regression : ",metrics.accuracy_score(y_test, ypred_smote))
print(metrics.confusion_matrix(y_test, ypred_smote))
print(metrics.classification_report(y_test, ypred_smote))


# In[95]:


# SUPPORT VECTOR MACHINES (SVM)
from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(X_train_smote, y_train_smote)
ypred2=svm_model.predict(X_test)
print("Accuracy Score SVM: ", metrics.accuracy_score(y_test, ypred2))
print(metrics.confusion_matrix(y_test, ypred2))
print(metrics.classification_report(y_test, ypred2))


# In[96]:


# K-NEAREST NEIGHBOR ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train_smote, y_train_smote)
pred_cv5= KNN.predict(X_test)
print("Accuracy score K-Nearest: ", metrics.accuracy_score(y_test, pred_cv5))
matrix5 = metrics.confusion_matrix(y_test, pred_cv5)
print(matrix5)
print(metrics.classification_report(y_test, pred_cv5))


# In[101]:


# DECISION TREE
dt = DecisionTreeClassifier()

# fit model into training data
dt.fit(X_train_smote, y_train_smote)

# get y prediction
y_pred_dt = dt.predict(X_test)

# print score
print("Accuracy score Decision Tree = ", metrics.accuracy_score(y_test,y_pred_dt))
print(metrics.classification_report(y_test,y_pred_dt))

# confusion matrix
print(metrics.confusion_matrix(y_test,y_pred_dt))


# In[98]:


df.head()


# In[99]:


print("Accuracy Logistik Regression pada imbalanced dataset : ",
      metrics.accuracy_score(y_test, ypred))
print(metrics.confusion_matrix(y_test, ypred))
recall1 = (10731)/(10731+238)
precision1= 10731/(10731+903)
print("precision : ", precision1)
print("recall: ",recall1)
print(metrics.classification_report(y_test, ypred))
print("Accuracy Logistik Regression setelah melakukan balancing dataset: ",
     metrics.accuracy_score(y_test, ypred_smote))
print(metrics.confusion_matrix(y_test, ypred_smote))
recall2= 9282/(9282+1687)
precision2 = 9282/(9282+308)
print("precision: ",precision2)
print("recall: ", recall2)
print(metrics.classification_report(y_test, ypred_smote))


# Untuk Hasil akhiir Classification algoritma Logistic Regression memiliki performa yang lebih baik dari yang algoritma yang lain seperti SVM, KNN, dan DT. Hasil classification diatas dapat dilihat bahwa proses klasifikasi dengan dataset imbalanced menghasilkan accuracy sebesar 0.91. Serta hasil klasifikasi setelah dilakukan balancing dataset menggunakan metode oversampling SMOTE menghasilkan accuracy sebesar 0.84.

# ## Projects Rubric
# Code Review
# 
# Criteria
# Meet Expectations
# 
# 1 Logistic Regression
# Mengimplementasikan Logistic Regression Dengan Scikit-Learn
# 
# 2 K-Nearest Neighbors
# Mengimplementasikan K-Nearest Neighbors Dengan Scikit-Learn
# 
# 3 Support Vector Machine
# Mengimplementasikan Support Vector Machine Dengan Scikit-Learn
# 
# 4 Decision Tree
# Mengimplementasikan Decision Tree Dengan Scikit-Learn
# 
# 5 Random Forest
# Mengimplementasikan Random Forest Dengan Scikit-Learn
# 
# 6 Naive Bayes
# Mengimplementasikan Naive Bayes Dengan Scikit-Learn
# 
# 7 Confusion Matrix
# Mengimpelentasikan Confusion Matrix Regression Dengan Scikit-Learn
# 
# 8 Visualization
# Menganalisa Data Menggunakan Setidaknya 2 Tipe Grafik/Plot.
# 
# 9 Preprocessing
# Melakukan Preproses Dataset Sebelum Melakukan Penelitian Lebih Dalam.
# 
# 10 Apakah Kode Berjalan Tanpa Ada Eror?
# Kode Berjalan Tanpa Ada Eror. Seluruh Kode Berfungsi Dan Dibuat Dengan Benar.
# Readibility
# 
# Criteria
# Meet Expectations
# 
# 11 Tertata Dengan Baik
# Semua Cell Di Notebook Terdokumentasi Dengan Baik Dengan Markdown Pada Tiap Cell Untuk Penjelasan Kode.
# Analysis
# 
# Criteria
# Meet Expectations
# 
# 12 Algorithm Analysis
# Student Menjelaskan Alasan Mengapa Memilih Menggunakan Algoritma Tersebut Untuk Membuat Model.

# In[ ]:




