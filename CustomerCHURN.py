#!/usr/bin/env python
# coding: utf-8

# # Stats and ML Project
# 
# ## Customer Churning 
# 
# ### Presented By-
# 
# Aditya Asrani
# 
# Aditya Bhatnagar
# 
# Aishwarya Kamalakannan
# 
# Kaustubh Chalke
# 
# Rohit Bhat
# 
# 
# ## Introduction
# 
# Churning is one of the telecoms industry's main challenges. Research has shown that the average monthly churn rate among America's top 4 wireless carriers is 1.9%-2%.

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format 
import matplotlib.pyplot as plt

sns.set(style = 'white')


# Let us read the data file below:

# In[8]:


telecom_cust= pd.read_csv('/Users/adityaasrani/Downloads/C.csv')

print (telecom_cust.shape)


# ## EDA

# In[9]:


telecom_cust.head(3)


# In[10]:


telecom_cust.columns.values


# Let us check the data for missing values:

# In[11]:


j


# 

# In[12]:


telecom_cust.TotalCharges = pd.to_numeric(telecom_cust.TotalCharges, errors='coerce')
telecom_cust.isnull().sum()


# In[14]:


#Removing missing values 
telecom_cust.dropna(inplace = True)
#Remove customer IDs from the data set
df2 = telecom_cust.iloc[:,1:]
#Convertin the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Let's convert all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
df_dummies.head()


# In[15]:


#Get Correlation of "Churn" with other variables:
plt.figure(figsize=(15,8))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# Contracts from month to month, shortage of cyber protection and tech assistance, appear to align favorably with churning. Although, tenure and two year contracts are in negative correlation with churn. 
# Also services like Online Surveillance, live TV, online storage, tech help, etc. without internet access appear to be churn-related negatively too.

# ## DATA EXPLORATION:
# 
# Let's start by exploring our data set first, in order to better understand the data dynamics and potentially shape any hypothesis. First we'll look at how different variables are spread and then slice and dice our results for some important patterns. 
# ### A.) Demographics - 
# 
# Let's first consider the customers' gender, age span, partner and dependency status.
# 
# 1) Gender Distribution - Our data set is divided into two halves ; male and female.

# In[16]:


colors = ['#4D3425','#E4512B']
ax = (telecom_cust['gender'].value_counts()*100.0 /len(telecom_cust)).plot(kind='bar',
                                                                           stacked = True,
                                                                          rot = 0,
                                                                          color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers')
ax.set_xlabel('Gender')
ax.set_ylabel('% Customers')
ax.set_title('Gender Distribution')

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-3.5,             str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            color='white',
           weight = 'bold')


# Senior Citizen % - Just 16 per cent of senior citizens are clients. And the bulk of our data consumers are younger men.

# In[17]:


ax = (telecom_cust['SeniorCitizen'].value_counts()*100.0 /len(telecom_cust)).plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 )                                                                           
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('Senior Citizens',fontsize = 12)
ax.set_title('% of Senior Citizens', fontsize = 12)


# Partner and dependent status - Almost 48% of the customers have a partner, while only 30% of the total customers have dependents.

# In[18]:


df2 = pd.melt(telecom_cust, id_vars=['customerID'], value_vars=['Dependents','Partner'])
df3 = df2.groupby(['variable','value']).count().unstack()
df3 = df3*100/len(telecom_cust)
colors = ['#4D3425','#E4512B']
ax = df3.loc[:,'customerID'].plot.bar(stacked=True, color=colors,
                                      figsize=(8,6),rot = 0,
                                     width = 0.2)

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers',size = 14)
ax.set_xlabel('')
ax.set_title('% Customers with dependents and partners',size = 14)
ax.legend(loc = 'center',prop={'size':14})

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',
               size = 14)


# What would be interesting is to look at the % of customers, who have partners, also have dependents. We will explore this next:
# 
# Among the customers who have a partner, only about half of them have a dependent, while the other half do not have any independents. Also, among the customers who do not have any partner, a majority (80%) of them do not have any dependents .

# In[19]:


colors = ['#4D3425','#E4512B']
partner_dependents = telecom_cust.groupby(['Partner','Dependents']).size().unstack()

ax = (partner_dependents.T*100.0 / partner_dependents.T.sum()).T.plot(kind='bar',
                                                                width = 0.2,
                                                                stacked = True,
                                                                rot = 0, 
                                                                figsize = (8,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Dependents',fontsize =14)
ax.set_ylabel('% Customers',size = 14)
ax.set_title('% Customers with/without dependents based on whether they have a partner',size = 14)
ax.xaxis.label.set_size(14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',
               size = 14)


# If we looked at any differences between the % of customers with/without dependents and partners by gender,there is no difference in their distribution by gender. Additionally, there is no difference in senior citizen status by gender.
# 
# 

# ### B.) Customer Account Information:
# Let us look at the tenure and contract;
# 
# 1)Tenure: By looking at the histogram below, we can see that many subscribers have been with the telecom company for just a month, although only a few have been around for around 72 months. Potentially, that may be because various companies have different contracts. And it may be more / less simpler for the consumers to stay / leave the telecom firm depending on the deal they are in.

# In[20]:


ax = sns.distplot(telecom_cust['tenure'], hist=True, kde=False, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
ax.set_ylabel('# of Customers')
ax.set_xlabel('Tenure (months)')
ax.set_title('# of Customers by their tenure')


# 2)Contracts: To understand the above graph, lets first look at the # of customers by different contracts.

# In[21]:


ax = telecom_cust['Contract'].value_counts().plot(kind = 'bar',rot = 0, width = 0.3)
ax.set_ylabel('# of Customers')
ax.set_title('# of Customers by Contract Type')


# As we can see from this graph most of the customers are in the month to month contract. While there are equal number of customers in the 1 year and 2 year contracts.
# 
# Below we will understand the tenure of customers based on their contract type.

# In[22]:


fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, sharey = True, figsize = (20,6))

ax = sns.distplot(telecom_cust[telecom_cust['Contract']=='Month-to-month']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'turquoise',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax1)
ax.set_ylabel('# of Customers')
ax.set_xlabel('Tenure (months)')
ax.set_title('Month to Month Contract')

ax = sns.distplot(telecom_cust[telecom_cust['Contract']=='One year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'steelblue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax2)
ax.set_xlabel('Tenure (months)',size = 14)
ax.set_title('One Year Contract',size = 14)

ax = sns.distplot(telecom_cust[telecom_cust['Contract']=='Two year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'darkblue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax3)

ax.set_xlabel('Tenure (months)')
ax.set_title('Two Year Contract')


# Interestingly, most of the monthly contracts last 1-2 months, while the 2 year contracts appear to run over 70 months. This shows that clients who sign a longer term are more committed to the company and prefer to stick with it for a longer time. 
# 
# That is also what we see on correlation with churn rate in the previous chart.

# ### C. Let us now look at the distribution of various services used by customers

# In[23]:


telecom_cust.columns.values


# In[24]:


services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))
for i, item in enumerate(services):
    if i < 3:
        ax = telecom_cust[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0)
        
    elif i >=3 and i < 6:
        ax = telecom_cust[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0)
        
    elif i < 9:
        ax = telecom_cust[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0)
    ax.set_title(item)


# ### D.) Now let's take a quick look at the relation between monthly and total charges
# 
# We will observe that the total charges increases as the monthly bill for a customer increases.

# In[25]:


telecom_cust[['MonthlyCharges', 'TotalCharges']].plot.scatter(x = 'MonthlyCharges',
                                                              y='TotalCharges')


# ### E.) Finally, let's take a look at out predictor variable (Churn) and understand its interaction with other important variables as was found out in the correlation plot.
# 
# Lets first look at the churn rate in our data

# In[26]:


colors = ['#4D3425','#E4512B']
ax = (telecom_cust['Churn'].value_counts()*100.0 /len(telecom_cust)).plot(kind='bar',
                                                                           stacked = True,
                                                                          rot = 0,
                                                                          color = colors,
                                                                         figsize = (8,6))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers',size = 14)
ax.set_xlabel('Churn',size = 14)
ax.set_title('Churn Rate', size = 14)

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-4.0,             str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            color='white',
           weight = 'bold',
           size = 14)


# 74 per cent of customers are not churning in our results. The evidence were obviously biased, because we can assume a vast number of the consumers not to churn. It is crucial to bear in mind as skewedness could contribute to a lot of false negatives for our modeling. We will see how to avoid skewing in the data in the modelling section.

# Lets now explore the churn rate by tenure, seniority, contract type, monthly charges and total charges to see how it varies by these variables.
# #### i.) Churn vs Tenure:
# As we can see form the below plot, the customers who do not churn, they tend to stay for a longer tenure with the telecom company.

# In[27]:


sns.boxplot(x = telecom_cust.Churn, y = telecom_cust.tenure)


# #### ii.) Churn by Contract Type: 
# 
# Similar to what we saw in the correlation plot, the customers who have a month to month contract have a very high churn rate.

# In[28]:


colors = ['#4D3425','#E4512B']
contract_churn = telecom_cust.groupby(['Contract','Churn']).size().unstack()

ax = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0, 
                                                                figsize = (10,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Contract Type',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',
               size = 14)


# #### iii.) Churn by Seniority: 
# 
# Senior Citizens have almost double the churn rate than younger population.

# In[29]:


colors = ['#4D3425','#E4512B']
seniority_churn = telecom_cust.groupby(['SeniorCitizen','Churn']).size().unstack()

ax = (seniority_churn.T*100.0 / seniority_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.2,
                                                                stacked = True,
                                                                rot = 0, 
                                                                figsize = (8,6),
                                                                color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers')
ax.set_title('Churn by Seniority Level',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',size =14)


# #### iv.) Churn by Monthly Charges: 
# 
# Higher % of customers churn when the monthly charges are high.
# 
# 

# In[30]:


ax = sns.kdeplot(telecom_cust.MonthlyCharges[(telecom_cust["Churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(telecom_cust.MonthlyCharges[(telecom_cust["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of monthly charges by churn')


# #### v.) Churn by Total Charges: 
# 
# It seems that there is higer churn when the total charges are lower.

# In[31]:


ax = sns.kdeplot(telecom_cust.TotalCharges[(telecom_cust["Churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(telecom_cust.TotalCharges[(telecom_cust["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Total Charges')
ax.set_title('Distribution of total charges by churn')


# After going through the above EDA we will develop some predictive models and compare them.
# We will develop Logistic Regression, Random Forest, SVM, ADA Boost and XG Boost
# 
# ## 1)Logistic Regression

# In[32]:


# We will use the data frame where we had created dummy variables
y = df_dummies['Churn'].values
X = df_dummies.drop(columns = ['Churn'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features


# In logistic regression, it is necessary to scale the variables so that they are all within a range of 0 to 1 This allowed me to increase the precision from 79.7% to 80.7%. Furthermore, you can find below that the meaning of variables is often consistent with what we see in the Random Forest algorithm and the EDA we have done above.

# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[35]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# In[36]:


from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))


# In[38]:


#To get the weights of all the variables
weights = pd.Series(model.coef_[0],
                index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# In[39]:


print(weights.sort_values(ascending = False)[-10:].plot(kind='bar'))


# Observations:
# 
# Certain variables have a negative relation to our expected variable (Churn), while others have a positive relation. Negative relation means churn likeliness of the churn decreases. Let's sum up a couple of the important features below:
# 
# -Getting a 2 month term, as we found in our EDA, lowers churn chances. Two-month term with tenure has the most detrimental association with Churn as predicted by logistical regressions 
# 
# -Providing DSL internet access often decreases Churn's capacity.
# 
# -Finally, overall payments, annual contracts, fiber optic internet services and seniority growing lead in higher churn levels. This is important as consumers are likely to cancel because of it, while fiber optic networks are cheaper. I guess to fully grasp why this is happening, we need to discuss further.

# ## 2)Random Forest Classifier:

# In[40]:


from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[41]:


importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')


# Observations: 
# 
# -The most important predictor variables to forecast churn come from the random forest method, monthly deal, tenure and overall costs. 
# 
# -Random forest findings are quite close to logistical regression and in line with what we had predicted from our EDA.

# ## 3)Support Vecor Machine (SVM):

# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


# In[43]:


from sklearn.svm import SVC

model.svm = SVC(kernel='linear') 
model.svm.fit(X_train,y_train)
preds = model.svm.predict(X_test)
metrics.accuracy_score(y_test, preds)


# In[44]:


# Create the Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,preds))  


# With SVM I was able to improve to 82 percent the precision. However, for a successful forecast, we need to take a deeper look at the true positive and true negative concentrations within the Under the Curve Field (AUC).

# In[45]:


ax1 = sns.catplot(x="gender", kind="count", hue="Churn", data=telecom_cust,
                  estimator=lambda x: sum(x==0)*100.0/len(x))
#ax1.yaxis.set_major_formatter(mtick.PercentFormatter())


# ## 4)ADA Boost :
# 

# In[46]:


# AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
# n_estimators = 50 (default value) 
# base_estimator = DecisionTreeClassifier (default value)
model.fit(X_train,y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)


# In[49]:


conda install -c anaconda py-xgboost


# ## 5)XG Boost:

# In[51]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)


# ## Conclusion;
# We were able to increase the precision of test results to approximately 83 percent with XG Upgrade. Like all the other methods, XG Boost is the best amongst all the other methods, which gives the best output for our churning .
# 
# 
# 
