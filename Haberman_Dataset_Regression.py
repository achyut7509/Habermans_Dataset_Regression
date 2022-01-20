# IMPORT NECESSARY LIBRARIES

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# IMPORT DATASET

haberman_dataset = pd.read_csv('H:/DATA SCIENCE/Real-Time Datasets/Logistic_Regression/haberman.data')

# DATA UNDERSTANDING 

  ## ABOUT DATA
  ## Attribute Information:
      ## 1. Age of patient at time of operation (numerical)
      ## 2. Patient's year of operation (year - 1900, numerical)
      ## 3. Number of positive axillary nodes detected (numerical)
      ## 4. Survival status (class attribute)
         ## 1 = the patient survived 5 years or longer
         ## 2 = the patient died within 5 year

  ## Initial Analysis

data_shape          = haberman_dataset.shape
data_null_check     = haberman_dataset.isna().sum()
data_dtypes         = haberman_dataset.dtypes
data_description    = haberman_dataset.describe(include = 'all')

# MODEL BUILDING

X = haberman_dataset.drop(labels = 'Survival_status', axis = 1)
y = haberman_dataset[['Survival_status']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12,)

# MODEL TRAINING

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)

logistic_model.intercept_

coefficients = np.array(logistic_model.coef_).T

coefficients_df = pd.DataFrame(coefficients,columns=['coefficients'],index=['Age','Year_of_operation(1900s)','Positive_axillary_nodes_detected'])

# MODEL TESTING

    ## TRAINING MODEL

y_predict_train = logistic_model.predict(X_train)
y_predict_train

    ## TEST MODEL

y_predict_test = logistic_model.predict(X_test)
y_predict_test

# MODEL EVALUATION

from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix

    ## TRAINING DATA

accuracy_score(y_train,y_predict_train)  # Accuracy = 0.7379912663755459
precision_score(y_train,y_predict_train) # Precision = 0.7379912663755459
recall_score(y_train,y_predict_train)    # Recall = 0.9447852760736196
classification_report(y_train, y_predict_train) 
confusion_matrix(y_true = y_train , y_pred = y_predict_train) 

print('TRAINING DATA ANALYSIS')
print('\n----------------------------------------------------')
print('Accuracy Score   :', round(accuracy_score(y_train,y_predict_train),5))
print('Precision Score  :',round(precision_score(y_train,y_predict_train),5))
print('Recall Score     :',round(recall_score(y_train,y_predict_train),5))
print('Confusion Matrix :\n',confusion_matrix(y_train,y_predict_train))
print('Classification Report :\n',classification_report(y_train,y_predict_train))

## TEST DATA

accuracy_score(y_test,y_predict_test)  # Accuracy = 0.7922077922077922
precision_score(y_test,y_predict_test) # Precision = 0.8382352941176471
recall_score(y_test,y_predict_test)    # Recall = 0.9193548387096774
classification_report(y_test, y_predict_test) 
confusion_matrix(y_true = y_test , y_pred = y_predict_test) 

print('TEST DATA ANALYSIS')
print('\n----------------------------------------------------')
print('Accuracy Score   :', round(accuracy_score(y_test,y_predict_test),5))
print('Precision Score  :',round(precision_score(y_test,y_predict_test),5))
print('Recall Score     :',round(recall_score(y_test,y_predict_test),5))
print('Confusion Matrix :\n',confusion_matrix(y_test,y_predict_test))
print('Classification Report :\n',classification_report(y_test,y_predict_test))
