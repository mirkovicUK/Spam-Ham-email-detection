#%% 
#1.1 Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from   sklearn.model_selection import GridSearchCV # use forcross validation
from   sklearn.metrics import plot_confusion_matrix
from   sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from   sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from   sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import sys


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


#%%
#1.4 Looking at data -> high disbalans 
# Check procentage of ham in email
sum(y) / len(y) # ham 87% spam 13%
sum(y)
len(y) - sum(y)
#procentage vizualization
sns.set()
sns.countplot(data=df, x=y).set_title("Ham --------- Spam ", fontweight = "bold")
#plt.show()


# %%
#1.5Building preliminary GXBoost model 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25, stratify=y)

clf_xgb = xgb.XGBClassifier(objective="binary:logistic", 
                                early_stopping_rounds=10,
                                eval_metric="auc", 
                                seed=42
                            )
clf_xgb.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)])
print("Best Iteration no: {}".format(clf_xgb.get_booster().best_iteration))
print("Best Iteration score: {}".format(clf_xgb.get_booster().best_score))

##############################################
#double check / different metrics
##############################################
#Predict = clf_xgb.predict(X_test)
#Score = accuracy_score(y_test, Predict)
#print("THIS IS DOUBLE CHECK OF FIT: ",end="")
#print(Score)
##############################################

#%%
#1.6 Print Confusion Matrix
plot_confusion_matrix(clf_xgb,
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
y_predict = clf_xgb.predict(X_new)
score=accuracy_score(y_new, y_predict)
print("Acuracy score on same model on new data set is: ",end="")
print(score)

# confusion matrix
plot_confusion_matrix(clf_xgb,
                    X_new,
                    y_new,
                    values_format="d",
                    display_labels=["Ham", "Spam"])


# %%
#1.8 Optimize xgBoost parameters using Cross Validation and GridSearch()

param_grid = {"max_depth" : [3], 
            "learning_rate" : [0.75],
            "gamma" : [0.01, 0.05, 0.01, 0.10, 0.15, 0.25 ], 
            "reg_lambda" : [3.0], 
            "scale_pos_weight" : [0.5] 
            }


######################################################################################################
# CROSs VALIDATION  GridSearchCV optimization comented out for optimal program flow                 ##
# Optimization si done on initial data set                                                          ##
# This optimization is for   CountVectorizer() no parameters                                        ##                                     
################################################################################################## ###
#optimal_params = GridSearchCV(estimator=xgb.XGBClassifier(objective="binary:logistic", 
#                                                        seed=42, 
#                                                        subsample=0.9, 
#                                                        colsample_bytree=0.5, 
#                                                        early_stopping_rounds=10, 
#                                                        eval_metric="auc"),
#                            param_grid=param_grid,
#                            scoring="roc_auc",
#                            verbose=2, 
#                            n_jobs=10,
#                            cv = 3)                     
#optimal_params.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])
#print(optimal_params.best_params_)
#sys.exit()
######################################################################################################################################################################
#rounds of optimization and output 
######################################################################################################################################################################
#Round1:param_grid = {"max_depth" : [3, 4, 5], "learning_rate" : [0.05, 0.1, 0.25],"gamma" : [0, 0.25, 1.0],"reg_lambda" : [1, 5, 10], "scale_pos_weight" : [2, 3, 4]}
# output cv = {'gamma': 1.0, 'learning_rate': 0.25, 'max_depth': 4, 'reg_lambda': 1, 'scale_pos_weight': 2}
#Round2:{"max_depth" : [3, 4, 5], "learning_rate" : [0.05, 0.1, 0.25],"gamma" : [1, 2, 3],"reg_lambda" : [1, 5, 10], scale_pos_weight" : [2, 3, 4]}
# output cv={'gamma': 2, 'learning_rate': 0.5, 'max_depth': 4, 'reg_lambda': 1.0, 'scale_pos_weight': 3}
#Round3: param_grid = {"max_depth" : [4], "learning_rate" : [0.25, 0.50, 0.75],"gamma" : [2,5,10], "reg_lambda" : [1.0, 3.0, 5.0], "scale_pos_weight" : [3,5] }
# output: {'gamma': 2, 'learning_rate': 0.5, 'max_depth': 4, 'reg_lambda': 1.0, 'scale_pos_weight': 3}
#etc...
#final paramiters for new model :{'gamma': 1.5, 'learning_rate': 0.5, 'max_depth': 4, 'reg_lambda': 1.0, }
#####################################################################################################################################################################
#####################################################################################################################################################################


#%%
#1.9 Building optimized GXBoost Model
clf_xgb = xgb.XGBClassifier(seed=42, 
                            objective="binary:logistic",
                            gamma=1.5,
                            learning_rate=0.5,
                            max_depth=4,
                            reg_lambda=1, 
                            scale_pos_weight=0.1, 
                            early_stopping_rounds=10, 
                            eval_metric="auc")
clf_xgb.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)])

#prediction for original data set
y_pred = clf_xgb.predict(X_test)
Score_one = accuracy_score(y_true=y_test, y_pred=y_pred)
print("OPTIMIZED MODEL original data set accuracy: ",end="")
print(Score_one)
#confusion matrix Original data
plot_confusion_matrix(clf_xgb,
                    X_test,
                    y_test,
                    values_format="d",
                    display_labels=["Ham", "Spam"])

#Prediction for new dataset
y_pred_new = clf_xgb.predict(X_new)
Score_two = accuracy_score(y_true=y_new, y_pred=y_pred_new)
print("OPTIMIZED MODEL NEW dataset accuracy: ",end="")
print(Score_two)
#confusion matrix new data
plot_confusion_matrix(clf_xgb,
                    X_new,
                    y_new,
                    values_format="d",
                    display_labels=["Ham", "Spam"])