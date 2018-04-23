import os
import sys
import zipfile
import pandas as pd
import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import re
import seaborn as sns
import pickle
import gzip
import bz2

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import *
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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


def images_to_df(filepath, img_size=(200, 200), pics_per_class=None):
    """Extracts images from data file and converts each into a pandas dataframe. 
    Outputs a pandas dataframe"""
    
    # set column names
    cols = ['px' + str(i) for i in range(1,1 + 3 * img_size[0] * img_size[1])]
    cols = ['class', 'id'] + cols   
    
    ret_df = pd.DataFrame()
    # Exception: Path ./C:/Users/Nick/kaggle/plant-seedlings/colorized not found
    # extrac the images through each folder. 
    out_folder = './' + re.search('^.*/(.*?)\..*$', filepath).group(1)
    print('Out Folder:', out_folder)
    
    if not os.path.isdir(out_folder):
        try:
            zfile = zipfile.ZipFile(filepath)
            zfile.extractall()
        except Exception as e:
            print(e)
               
    if not os.path.isdir(out_folder):
        raise Exception('Path {0} not found.'.format(out_folder))
        
    for dname in os.listdir(out_folder):
        if os.path.isdir(out_folder + '/' + dname):
            for fname in os.listdir(out_folder + '/' + dname)[0:pics_per_class]:
                cl = dname
                ix = fname[:-4]
                im = cv2.imread(out_folder + '/' + dname + '/' + fname)
                
                # to put into an DataFrame for classification, 
                # all the images need to be the same dimension. 
                im = cv2.resize(im, (img_size[0],img_size[1])).reshape(-1)
                im = im.reshape(1, len(im))
                df = pd.DataFrame({
                    'class':[cl],
                    'id':[ix]
                })
                df = df.join(pd.DataFrame(im))
                ret_df = pd.concat([ret_df, df])
    ret_df.columns = cols
    return ret_df.reset_index(drop=True)

def get_data(filepath, read_pickle=False, write_pickle=False):
    if read_pickle:
        big_data = pd.read_pickle(filepath)
    else:
        data = {}
        for file in filepath: #, alpha_file, colorized_file]
            name = re.search('^.*/(.*?)\.zip$', file).group(1)
            data[name]= images_to_df(file, 
                                    img_size=img_size, 
                                     pics_per_class=pics_per_class)
        if len(data) > 1:
            for i in range(0, len(data) - 1):
                big_data = data[i].merge(data[i+1], on=['id', 'class'])
        else:
            big_data = data[0]
    
        if write_pickle:
            big_data.to_pickle(pickle_file)
    return big_data


def save_pickle(obj, out_file, zip_class=None):
    if zip_class != None:
        f = zip_class.open(out_file, 'wb')
    else:
        f = open(out_file, 'wb')
    pickle.dump(obj, f)
    f.close()
        
def load_pickle(in_file, zip_class=None):
    if zip_class != None:
        f = zip_class.open(in_file, 'rb')
    else:
        f = open(in_file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret


# Create a class that will take in the data and return a trained model with 
# results
class sk_cls():
    """sk_cls takes an sk-learn classifier, pandas dataset, and list of features and
    target variables, and builds a trained model with results statistics"""
    
    def __init__(self, cls, data, X, y, ratio=0.1, cv=None):
        # check data
#         if data.isnull().sum().any():
#             raise Exception('Data contains NA values. Correct before attempting to model')
        
        # set initial variables
        self.model = cls
        self.score = 0
        self.conf_matrix = None
        self.cls_report = None
        self.y_pred = None
        self.__tgt = y
        self.__ftr = list(X)
        self.__sorted_classes = sorted(data[self.__tgt].unique())
        self.__cv = cv
        
        # create train test split
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            data[self.__ftr],
            data[self.__tgt], test_size=ratio)
    
    def train_cls(self, cls, X_train, X_test, y_train, y_test):
        """Fir the model and """
        if self.__cv == None:
            # until I can figure out why I'm getting NAs I'll use fillna
            cls.fit(X_train.fillna(0), y_train)
            y_pred = cls.predict(X_test)
            sc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
        else:
            X = pd.concat([X_train, X_test])
            y = pd.concat([y_train, y_test])
            y_pred = cross_val_predict(cls, X.fillna(0), y, cv=self.__cv, n_jobs=2)
            sc = accuracy_score(y, y_pred)
            cr = classification_report(y_train, y_pred)
            cm = confusion_matrix(y_train, y_pred)
            
        results = {
            'cls':cls,
            'sc':sc,
            'cm':cm,
            'cr':cr,
            'y_pred':y_pred
        }
        return results
    
    def run(self):
        """Run the model"""
        results = self.train_cls(self.model,
                                 self.__X_train, self.__X_test, 
                                 self.__y_train, self.__y_test)
        self.model = results['cls']
        self.score = results['sc']
        self.conf_matrix = results['cm']
        self.cls_report = results['cr']
        self.y_pred = results['y_pred']
        
    def show(self, plot=True, title=None, cbar=False, normed=True):
        """outputs all results including a cm plot if enabled"""
        if self.score == 0:
            raise Exception('Model has not been run. Use sk_cls.run() first')
            
        print('Score:\n', self.score)
        print('Classification Report:\n', self.cls_report)
        if plot:
            title = 'Confusion Matrix' if title == None else title
            
            cm_plot(self.conf_matrix, 
                    classes=self.__sorted_classes, 
                    title=title, 
                    normalize=normed,
                    cbar=cbar,
                    cmap=plt.cm.Greens)
            plt.show()
        else:
            print(self.conf_matrix)
            
