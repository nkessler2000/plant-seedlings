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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
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


# Create a class that will take in the data and return a trained model with 
# results
class sk_cls():
    """sk_cls takes an sk-learn classifier, pandas dataset, and list of features and
    target variables, and builds a trained model with results statistics"""
    
    def __init__(self, cls, data, X, y, ratio=0.1):
        # check data
        if data.isnull().sum().any():
            raise Exception('Data contains NA values. Correct before attempting to model')
        
        # set initial variables
        self.model = cls
        self.score = 0
        self.conf_matrix = None
        self.cls_report = None
        self.__tgt = y
        self.__ftr = X
        self.__sorted_classes = sorted(y.unique())
        
        # create train test split
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            data.loc[:, self.__ftr],
            data.loc[:, self.__tgt], test_size=ratio)
    
    def train_cls(self, cls, X_train, X_test, y_train, y_test):
        """Fir the model and """
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        sc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        results = {
            'cls':cls,
            'sc':sc,
            'cm':cm,
            'cr':cr
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
        
    def show(self, plot=True, title=None, cbar=False, normed=True):
        """outputs all results including a cm plot"""
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
