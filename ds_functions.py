import os
import zipfile
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import itertools
from scipy import stats


def zip_to_df(filepath, sep=',', encoding='auto', header='infer'):
	"""Extracts a CSV file from a zip and then loads into a pandas DataFrame"""
	zfile = zipfile.ZipFile(filepath)
	for finfo in zfile.infolist():
		ifile = zfile.open(finfo)
	df = pd.read_csv(ifile, sep=sep)
	return df

def cm_plot(cm, classes=None, title=None, cmap=plt.cm.Reds, figsize=(10,8), normalize=False, cbar=True):
	"""reads in a confusion matrix from a scikit_learn CM, returns a matplotlib plot"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	
	p = plt.figure(figsize=figsize)
	p = plt.imshow(cm, interpolation='nearest', cmap=cmap)
	
	if title != None:
		p = plt.title(title)
		
	if cbar:
		p = plt.colorbar()
	
	if classes != None:
		tick_marks = np.arange(len(classes))
		p = plt.xticks(tick_marks, classes, rotation=45)
		p = plt.yticks(tick_marks, classes)
	
	fmt = '.2f' if normalize else 'd'
	
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		p = plt.text(j, i, format(cm[i, j], fmt),
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")
	
	p = plt.tight_layout()
	p = plt.ylabel('True label')
	p = plt.xlabel('Predicted label')
	return p

def cm_to_df(cm, classes=None):
	"""Converts a sklearn confusion matrix into a pandas DF"""
	df = pandas.DataFrame(cm, index=classes, columns=classes)
	return df
	
def create_test_train(df):
    """Creates a train-test split from a dataframe. First one-hot encodes, then splits. Returns 
	X_trian, X-test, y_train, y_test"""
    # create X and y
    y = df.country_group
    X = df.drop(['id', 'country_group', 'country_destination', 'date_account_created', 'timestamp_first_active', 
            'date_first_booking'], axis=1)
    
    # specify type for signup flow
    X['signup_flow'] = X['signup_flow'].astype('category')
    # One hot encoding of categorical variables 
    # using Pandas get_dummies method
    X_onehot = pd.get_dummies(X) 
    X_onehot.age = X_onehot.age.fillna(0)
    X_onehot.days_to_book = X_onehot.days_to_book.fillna(100000)
    
    # create test train split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_onehot, y, test_size=0.1, random_state=1)
    # return test and train groups
    return X_train, X_test, y_train, y_test