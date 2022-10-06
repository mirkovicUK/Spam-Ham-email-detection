#%% 
#1.1 Import libraries
import pandas as pd
import numpy as np
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from   sklearn.metrics import plot_confusion_matrix
from   sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from   sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys
from   sklearn.model_selection import GridSearchCV


#%%
#1.2 Import and sort data

df = pd.read_csv("spam.csv")
df.drop(["v3","v4", "v5"], axis=1, inplace=True)
# check ham column
df["spam"].unique()
# check for missing val 
len(df.loc[df["spam"] == " "]) 
len(df.loc[df["email"] == " "])

df["spam"].replace("spam",1 , inplace=True)
df["spam"].replace("ham",0 , inplace=True)

y = df["spam"].copy()
X = df["email"].copy()
df.head()
X.head()
y.head()


# %%
#1.3 Encoding
#count words and frequency
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
column_names = vectorizer.get_feature_names_out()

#convert X back to panda object for visualization
X = pd.DataFrame(X.toarray())
X.columns = column_names
X.head()
X.shape

#Data pictorial
sns.set()
sns.countplot(data=df, x=y).set_title("Ham --------- Spam ", fontweight = "bold")
#plt.show()

# %%
#1.4Building Model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25, stratify=y)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

Score = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Model test dataset1: ",end="")
print(Score)
#confusion matrix new data
plot_confusion_matrix(model,
                    X_test,
                    y_test,
                    values_format="d",
                    display_labels=["Ham", "Spam"])


# %%
#1.7 Measuring performance on new data set and printing confusin matrix

#read in new data
df_new = pd.read_csv("test_data/spam_ham1.csv")
df_new.head()
#Check for missing values
len(df_new.loc[df_new["spam"] == " "]) 
len(df_new.loc[df_new["email"] == " "])

y_new = df_new["spam"].copy()
y_new.unique()
X_new = df_new["email"].copy()
y_new.head()
X_new.head()

# Check percentage of spam ham in data set
sum(y_new) / len(y_new) # new data set contains 24% of spam emails 

# count words and frequency
X_new = vectorizer.transform(X_new)
X_new = pd.DataFrame(X_new.toarray())
#Confirm New data set b4 feeding to clf_xgb
X_new.shape
y_new.shape

#prediction
y_predict = model.predict(X_new)
score=accuracy_score(y_new, y_predict)
print("Acuracy score on same model on new data set is: ",end="")
print(score)

# confusion matrix
plot_confusion_matrix(model,
                    X_new,
                    y_new,
                    values_format="d",
                    display_labels=["Ham", "Spam"])


# %%
