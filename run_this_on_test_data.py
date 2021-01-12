import csv
import datetime
import dateutil.relativedelta
import pandas as pd
import numpy as np
from data_prep_and_training import data_prep
from joblib import load
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def write_csv(df,regressor_prediction, classifier_prediction):
  #get prediction date
  last_date = df.iloc[-1].date
  d = datetime.datetime.strptime(last_date, "%Y-%m-%d")
  prediction_date = d + dateutil.relativedelta.relativedelta(months=1)
  year_pred = prediction_date.year
  month_pred = prediction_date.month

  #we only need predictions for the most recent date
  comp_id_list = []
  for i in range(len(regressor_prediction)):
      if df.iloc[i].date == last_date:
          comp_id_list.append(df.iloc[i].comp_id)
  num_comps = len(comp_id_list)
  regressor_predictions_to_keep = regressor_prediction[-(num_comps+1):-1]
  classifier_predictions_to_keep = classifier_prediction[-(num_comps+1):-1] 

  #arrange the data to be written to csv
  ret_rows = np.zeros((len(regressor_predictions_to_keep),4))
  gof_rows = np.zeros((len(classifier_predictions_to_keep),4))
  ret_rows[:,0] = year_pred
  gof_rows[:,0] = year_pred
  ret_rows[:,1] = month_pred
  gof_rows[:,1] = month_pred
  ret_rows[:,2] = comp_id_list
  gof_rows[:,2] = comp_id_list
  ret_rows[:,3] = regressor_predictions_to_keep
  gof_rows[:,3] = classifier_predictions_to_keep

  #write csvs
  ret_fields = ['year_pred','month_pred','comp_id','m_ret']
  gof_fields = ['year_pred','month_pred','comp_id','m_gof']
  ret_filename = 'returns.csv'
  gof_filename = 'grow-fall.csv'
  with open(ret_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(ret_fields)
    csvwriter.writerows(ret_rows)
  with open(gof_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(gof_fields)
    csvwriter.writerows(gof_rows)

#read test csv and prep it
df = pd.read_csv("finalproject_training.csv") ###############CHANGE THIS TO TEST BEFORE WE PASS IT IN!#################
X_r,X_c,Y_r,Y_c = data_prep(df)

#load serialized predictors
rf_regressor = load("rf_regressor.serialized")
rf_classifier = load("rf_classifier.serialized")

#do the predictions
regressor_prediction = rf_regressor.predict(X_r)
classifier_prediction = rf_classifier.predict(X_c)

#write the output csvs
write_csv(df,regressor_prediction, classifier_prediction)
