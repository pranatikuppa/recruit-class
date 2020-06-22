#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import statements 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#getting the data from a live url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)


# In[3]:


#reading the first 5 rows 
dataset.head()


# In[4]:


#splitting the dataset into attributes / labels 
#label be whether or not they were accepted
#how to match names/student id back to their prediction?

#x represents the attributes, y represents the labels
x = dataset. iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# In[5]:


#splitting into testing/training data 
#80% training data, 20% testing data 
from sklearn.model_selection import train_test_split
def split_data():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    return x_train, x_test, y_train, y_test


# In[6]:


#FEATURE SCALING --> this is likely what we've been referring to as 'weighing'
#look into the standard scaling function and see how it can be adapted for our purposes
from sklearn.preprocessing import StandardScaler
#ogX_train, ogX_test, y_train, y_test = split_data() #INCLUDE

def feature_scale(x_train, x_test): 
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

#x_train, x_test = feature_scale(ogX_train, ogX_test) #INCLUDE


# In[7]:


#setting up KNN 
#5 is the most common used K initially --> we can begin to change this either lower/higher
#depending on how well/not well the predictions are matching the actual 
from sklearn.neighbors import KNeighborsClassifier

#initially, the k-value is passed in as a standard value 5 (right now, used 3)
k_val = 3

def k_setup(k, x_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    return classifier

#making predictions
def make_pred(k, x_train, y_train, x_test):
    y_pred = k_setup(k, x_train, y_train).predict(x_test)
    return y_pred


#y_pred= make_pred(k_val)


# In[8]:


#printing the confusion matrix, classification report and accuracy score if needed
def print_stats(y_test, y_pred): 
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    #return 
    



#ideally, the precision for everything would be 1.00 (which means it completely matches)
#though this is highly unlikely, so we're targeting around 90% (perhaps?)

#when k = 5, it'll get to 100%


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#function that performs knn given a k value, and optionally prints a report on the outcome
def k_trials(k_val, toPrint=False):
    acc_scores = []
    for _ in range(1): #changed as this just to force some error from this 'perfect ' dataset
        ogX_train, ogX_test, y_train, y_test = split_data()
        x_train, x_test = feature_scale(ogX_train, ogX_test)
        y_pred= make_pred(k_val, x_train, y_train, x_test)
        #print(accuracy_score(y_test, y_pred))
        acc_scores.append(accuracy_score(y_test, y_pred))
    if toPrint: 
        print_stats(y_test, y_pred)
    #print(acc_scores)
    return max(acc_scores)
    #return acc_scores
    
vals = {k_val: k_trials(k_val, True)}
#values(k_val)
#print(vals)


# In[10]:


#max_trials = 40
#for now, making max_trials 3 
max_trials = 3
def changek():
    curr_k = k_val #update this
    prev_acc = vals[k_val]
    new_acc = 0
    op = True #True, since initially addition
    adjusted = False #before entering the loop, K is assumed to not be adjusted
    for _ in range(max_trials):
        if 1.0 in vals.values(): #already reached perfect accuracy
            #reality: vals.values() includes an accuracy > 90%
            #print("reaches here")
            break
        else: 
            if adjusted == False: 
                #initial block only used to kickstart adjustment fluctuation
                curr_k +=1 #increase k value by 1 initially since hypothetically, this will increase accuracy
                vals[curr_k] = k_trials(curr_k)
                new_acc = vals[curr_k]
                adjusted = True 
            else: 
                if new_acc <= prev_acc: 
                    #keep doing the same operation as before 
                    prev_acc = vals[curr_k]
                    if op: 
                        curr_k += 1
                    else: 
                        curr_k -=1
                    vals[curr_k] = k_trials(curr_k)
                    new_acc = vals[curr_k]
                        
                else: #prev_acc > new_acc 
                    prev_acc = vals[curr_k]
                    op = not op
                    if op: 
                        curr_k += 2 # so as to not test same value again
                    else: 
                        curr_k -=2  
                    vals[curr_k] = k_trials(curr_k)
                    new_acc = vals[curr_k]
                    
                    
                #get the val associated w the curr_k
                #if the accuracy is < prev_a
                #then change the operation of updating the k, and perform the next update iteration

changek()

        


# In[11]:


print(vals)


# In[13]:


def get_highest_k(dict):
    accuracies = dict.values() # list of all the values in the dictionary
    highest_accuracy = max(dict.values()) #gets the highest value
    highest_accuracy_dict = {} #new dictionary that will take in the key/val pairs w/ the highest accuracy
    for i in dict:
        if dict[i] == highest_accuracy:
            highest_accuracy_dict[i] = dict[i] #accounts for if there are more than one keys with same accuracy
    return highest_accuracy_dict #returns a dictionary with the highest accuracy k-values

print(get_highest_k(vals))


# In[ ]:




