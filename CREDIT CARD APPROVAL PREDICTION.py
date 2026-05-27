#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


application_record=pd.read_csv("application_record.csv")
credit_record=pd.read_csv("credit_record.csv")


# In[4]:


application_record.head()


# ## Creating The Target variable

# In[3]:


# Replacing X,C values with 0 as they are Good clients
credit_record.replace(['X','C'], 0,inplace=True)


# In[4]:


credit_record.STATUS = pd.to_numeric(credit_record.STATUS)


# In[5]:


credit_record.head()


# In[6]:


# Searching for customers who have at least one late month
drop_ls = []
for i in range(len(credit_record)):
    if credit_record.STATUS[i] != 0:
        drop_ls.append(credit_record.ID[i])


# In[7]:


len(drop_ls)


# In[8]:


# Changing the STATUS of any client with one late month to 1
for i in range(len(credit_record)): 
        if credit_record.ID[i] in drop_ls:
            credit_record.STATUS[i] = 1


# In[9]:


credit_record


# In[10]:


credit_record.STATUS.value_counts()


# In[11]:


credit_record.drop_duplicates(inplace=True)
credit_record 


# ## Merging two datasets
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[12]:


print(f'No. of IDs in1 application_record = {len(application_record.ID)} No. of IDs in credit_record = {len(credit_record.ID)}')


# In[13]:


m_dataset = application_record.merge(credit_record, on=['ID'], how='inner')


# In[14]:


m_dataset.drop(['ID'],inplace=True,axis=1)


# In[15]:


m_dataset.duplicated().sum()


# In[16]:


m_dataset.drop_duplicates(inplace=True)


# In[17]:


m_dataset


# In[18]:


m_dataset.count()


# ## Getting merged dataset's Information and Description

# In[19]:


m_dataset.info()


# In[20]:


m_dataset.describe()


# In[21]:


m_dataset.isna().sum()


# In[22]:


m_dataset.OCCUPATION_TYPE


# In[23]:


m_dataset.OCCUPATION_TYPE.value_counts()


# ## Replacing null values with "other"

# In[24]:


m_dataset.OCCUPATION_TYPE.replace(np.nan, 'Other', inplace = True)


# In[25]:


m_dataset.OCCUPATION_TYPE.value_counts()


# In[26]:


m_dataset.isnull().sum()


# In[27]:


import seaborn as sns


# In[28]:


sns.pairplot(credit_record.sample(200))


# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt

heatmap_data = m_dataset[['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'FLAG_PHONE']]

correlation_matrix = heatmap_data.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

plt.show()


# In[30]:


plt.figure(figsize=(5,5))
sns.countplot(x='FLAG_OWN_CAR', data=m_dataset,hue='CODE_GENDER')
plt.title('Car Owner or Not')
plt.show()


# In[31]:


plt.figure(figsize=(5,5))
sns.countplot(x='FLAG_WORK_PHONE', data=m_dataset,hue='CODE_GENDER')
plt.title('having Phone or Not')
plt.show()


# In[32]:


plt.figure(figsize=(30,10))
sns.countplot(x='OCCUPATION_TYPE', data=m_dataset,hue='CODE_GENDER')
plt.title('Type of Occupation')
plt.show()


# In[33]:


m_dataset.head()


# In[34]:


plt.figure(figsize=(30,10))
sns.countplot(x='NAME_FAMILY_STATUS', data=m_dataset,hue='CODE_GENDER')
plt.title('Family Status')
plt.show()


# In[35]:


plt.figure(figsize=(30,10))
sns.countplot(x='NAME_HOUSING_TYPE', data=m_dataset,hue='CODE_GENDER')
plt.title('House Type')
plt.show()


# In[36]:


plt.figure(figsize=(30,10))
sns.countplot(x='CNT_FAM_MEMBERS', data=m_dataset,hue='CODE_GENDER')
plt.title('Count of family members')
plt.show()


# In[37]:


plt.figure(figsize=(30,10))
sns.countplot(x='NAME_EDUCATION_TYPE', data=m_dataset,hue='CODE_GENDER')
plt.title('Education type')
plt.show()


# In[38]:


# Create a pivot table for better visualization
pivot_table = m_dataset.pivot_table(index='CODE_GENDER', columns='STATUS', aggfunc='count', fill_value=0)

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='g')
plt.title('Credit Card Bill Payment Status by ID and Month')
plt.xlabel('Month')
plt.ylabel('ID')
plt.show()


# In[39]:


plt.figure(figsize=(8, 5))
sns.countplot(x='CODE_GENDER', data=m_dataset, palette='viridis')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[40]:


plt.figure(figsize=(10, 6))
sns.countplot(x='CODE_GENDER', hue='STATUS', data=m_dataset, palette='coolwarm')
plt.title('Credit Card Bill Payment Status by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Payment Status')
plt.show()


# In[41]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='FLAG_OWN_CAR', data=m_dataset, palette='Set1')
plt.title('Car Ownership Distribution')

plt.subplot(1, 2, 2)
sns.countplot(x='FLAG_OWN_REALTY', data=m_dataset, palette='Set2')
plt.title('Realty Ownership Distribution')

plt.show()


# In[42]:


plt.figure(figsize=(10, 6))
sns.countplot(x='NAME_EDUCATION_TYPE', data=m_dataset, palette='coolwarm')
plt.title('Education Level Distribution')
plt.xlabel('Education Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# In[44]:


X = m_dataset.drop(columns=['STATUS'])
y = m_dataset['STATUS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[45]:


# Preprocessing and Modeling Pipeline
numeric_features = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']
categorical_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[46]:


from sklearn.ensemble import RandomForestClassifier

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred))


# In[47]:


from sklearn.tree import DecisionTreeClassifier

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', DecisionTreeClassifier(random_state=42))])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("DT Classification Report:\n", classification_report(y_test, y_pred))


# In[48]:


from sklearn.ensemble import AdaBoostClassifier

model_adaboost = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', AdaBoostClassifier(random_state=42))])

model_adaboost.fit(X_train, y_train)

y_pred_adaboost = model_adaboost.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_adaboost))
print("AdaBoost Classification Report:\n", classification_report(y_test, y_pred_adaboost))


# In[49]:


from sklearn.neighbors import KNeighborsClassifier

model_knn = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', KNeighborsClassifier())])

model_knn.fit(X_train, y_train)

y_pred_knn = model_knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))


# In[50]:


from sklearn.linear_model import LogisticRegression

model_logreg = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(random_state=42))])

model_logreg.fit(X_train, y_train)

y_pred_logreg = model_logreg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logreg))


# In[51]:


from lightgbm import LGBMClassifier

model_lgbm = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LGBMClassifier(random_state=42))])

model_lgbm.fit(X_train, y_train)

y_pred_lgbm = model_lgbm.predict(X_test)

print("LightGBM Classifier Accuracy:", accuracy_score(y_test, y_pred_lgbm))
print("LightGBM Classifier Classification Report:\n", classification_report(y_test, y_pred_lgbm))


# In[52]:


from catboost import CatBoostClassifier

model_catboost = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', CatBoostClassifier(random_state=42, verbose=0))])

model_catboost.fit(X_train, y_train)

y_pred_catboost = model_catboost.predict(X_test)

print("CatBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_catboost))
print("CatBoost Classifier Classification Report:\n", classification_report(y_test, y_pred_catboost))


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

model_extra_trees = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', ExtraTreesClassifier(random_state=42))])

model_extra_trees.fit(X_train, y_train)

y_pred_extra_trees = model_extra_trees.predict(X_test)

print("Extra Trees Classifier Accuracy:", accuracy_score(y_test, y_pred_extra_trees))
print("Extra Trees Classifier Classification Report:\n", classification_report(y_test, y_pred_extra_trees))


# In[54]:


from sklearn.naive_bayes import GaussianNB


model_nb = Pipeline(steps=[('preprocessor', preprocessor),
                            ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
                            ('classifier', GaussianNB())])

model_nb.fit(X_train, y_train)

y_pred_nb = model_nb.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))


# In[55]:


from sklearn.ensemble import GradientBoostingClassifier



# Define the Gradient Boosting pipeline
model_gb = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', GradientBoostingClassifier(random_state=42))])

# Train the model
model_gb.fit(X_train, y_train)

# Make predictions
y_pred_gb = model_gb.predict(X_test)

# Evaluate the model
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Gradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gb))


# In[ ]:




