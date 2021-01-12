import csv
import datetime
import dateutil.relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump,load
from matplotlib.colors import ListedColormap
from scipy import interp
from sklearn import datasets, svm, metrics
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score, GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
#import feature_selector

# data prep

def data_prep(df):

    #removing columns with dates or strings...and the monthly return which will be our target
    features = list(df.columns)
    str_features = ['date','sent_date','last_report','curcdq', 'datacqtr', 'datafqtr', 'costat']
    for i in str_features:
        features.remove(i)

    #implementing new features
    df["PtoE_f12"] = df['close']/df['epsf12']
    df["PtoE_fxq"] = df['close']/df['epsfxq']
    df["PtoE_pxq"] =df['close']/df['epspxq']
    df["PtoE_x12"] =df['close']/df['epsx12']
    df["PtoB"] = df["close"]/df["atq"]
    df['lt_debt_r'] = df['dd1q']/df['dlttq'] #% of long term debt due in 1 year

    #need to delete features used in compound ones, otherwise multicollinearity
    feat_to_replace = ['epsf12','epsfxq','epspxq','atq','dd1q','dlttq']

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(by = ['comp_id','date'])
    for i in feat_to_replace:
        if i in features:
            features.remove(i)

    dates = pd.Series(df['date'].values).unique()
    companies = pd.Series(df['comp_id'].values).unique()
    num_comp = len(companies)
    df['mom1m']=0

    mom = [np.log(df['close'].iloc[i] / df['close'].iloc[i-1]) if df['comp_id'].iloc[i] == df['comp_id'].iloc[i-1] else 0 for i in range(len(df))]

    next_ret = [df['m_ret'].iloc[i] if df['comp_id'].iloc[i] == df['comp_id'].iloc[i-1] else 0 for i in range(1,len(df))]
    next_ret +=[0]

    df['mom1m'] = mom
    df['class'] = df['mom1m'] >= 0
    df['target_ret'] = next_ret

    new_features = ["PtoE_f12","PtoE_fxq","PtoE_pxq","PtoE_x12","lt_debt_r","PtoB"]
    features = features + new_features

    #re-sorting by date for time series splitting later
    df = df.sort_values(by = ['date'])

    ###CODE FROM THE FEATURE SELECTOR PACKAGE####
    #Output a list of 25 features to delete, based on 3 metrics
    #1)High Collinearity
    #2)Zero Importance Features
    #3)Low Importance Features
    ###See the attached feature_selector.py file for generating this list (lightgbm required)

    fs_feat = ['cshfd12','m_high','gsector','gsubind','cshfdq','lseq','SP500WeeklyClose','gind','cshprq','ibmiiq','esopctq','m_low_adj','epsx12','tfvceq','ibq','txdbclq','rdipq','xoprq','ibcomq','PtoE_pxq','m_low','epspxq','fyearq','niq','SP500WeeklyLow']
    for f in fs_feat:
        if f in features:
            features.remove(f)

    #standardize the data
    X = df.loc[:,features].values
    X = StandardScaler().fit_transform(X)

    #monthly returns/class untouched
    Y_regressor = df['target_ret'].values
    Y_classifier = df['class'].values

    #PCA using simple imputer
    imp_r = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_1r = imp_r.fit_transform(X)
    imp_c = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_1c = imp_c.fit_transform(X)

    #using a fixed number of 53 components, as they explain almost all of the variance
    #UPDATE: Removing this because it isn't working on the test data file
    '''
    pca_r = PCA(n_components=53)
    pca_c = PCA(n_components=53)
    pca_r.fit(X_1r,Y_regressor)
    pca_c.fit(X_1c,Y_classifier)
    most_important_r = [np.abs(pca_r.components_[i]).argmax() for i in range(len(pca_r.components_))]
    most_important_c = [np.abs(pca_c.components_[i]).argmax() for i in range(len(pca_c.components_))]
    most_important_names_r = [features[most_important_r[i]] for i in range(len(pca_r.components_))]
    most_important_names_c = [features[most_important_c[i]] for i in range(len(pca_c.components_))]
    unique_r = np.unique(np.array(most_important_names_r))
    unique_c = np.unique(np.array(most_important_names_c))
    X_r = df.loc[:,unique_r].values
    X_r = StandardScaler().fit_transform(X_r)
    X_r = imp_r.fit_transform(X_r)
    X_c = df.loc[:,unique_c].values
    X_c = StandardScaler().fit_transform(X_c)
    X_c = imp_c.fit_transform(X_c)
    return X_r,X_c,Y_regressor,Y_classifier
    '''

    return X_1r,X_1c,Y_regressor,Y_classifier

# Regressor training

def rf_regressor_train(X,Y,splits):
    tscv = TimeSeriesSplit(n_splits=splits)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        random_forest = RandomForestRegressor(max_depth=15,n_estimators=150,n_jobs=-1,max_features='sqrt',random_state=55, min_samples_leaf=2)
        random_forest.fit(X_train, Y_train)

        #print("Training score:",random_forest.score(X_train, Y_train))
        #print("Test score:",random_forest.score(X_test, Y_test))
    return random_forest


def run_model_comparison(X_train,X_test,Y_train,Y_test):
    """
    Performs comparative analysis of classification models
    Plots calibration of multiple different classifiers, as well as an 
    accuracy, precision, and recall score
    
        Inputs:
            X_train,X_test,Y_train,Y_test: pre split matrices of test and training data
            
    """
    #setup the classifiers to test
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier(max_depth=15,n_estimators=95)
    knn = KNeighborsClassifier()
    #create a plot figure to store results in
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    #plot a 45 degree line that represents perfect calibration
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    #for each classifier we created
    for clf, name in [
                  (lr, 'Logistic Regression'),
                  (gnb, 'Naive Bayes'),
                  (svc,  'Support Vector Classification'),
                  (knn, 'K Nearest Neighbors'),
                  (rfc, 'Random Forest'),
                  ]:
        #perform the fit
        clf.fit(X_train, Y_train)

        #if the classifier has a predict_proba, use that as the prob of positives
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        #otherwise we use the deficision function
        else:
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(Y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))
        Y_prediction = clf.predict(X_test)
        
        print(name)
        print(clf.score(X_test,Y_test))
        print(metrics.precision_score(Y_test,Y_prediction))
        print(metrics.recall_score(Y_test,Y_prediction))
    
    #finish setting up the plot
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    #show
    plt.tight_layout()
    plt.show()
    #returns nothing, outputs all results
    return

def rf_regr_hyperparameters(X,Y):
    rf = RandomForestRegressor()
    tscv = TimeSeriesSplit(n_splits=7)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        random_grid = {'max_depth': [10,15,20],
                  'n_estimators': [80,100,150,200],
                  'min_samples_leaf': [2,4],
                  'max_features': ['auto','sqrt']}
        rsc = RandomizedSearchCV(estimator = rf, param_distributions=random_grid,
                           cv=3,scoring='neg_mean_squared_error')

        rsc_result = rsc.fit(X_train,Y_train)
        best_params = rsc_result.best_params_
        cv_results = rsc.cv_results_
        for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
            print(np.sqrt(-mean_score), params)
            print("Best params:",rsc.best_params_)
            print("Best score:",rsc.best_score_)


def rf_class_hyperparameters(X,Y,depths,estimators):
    rf = RandomForestClassifier()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
    gsc = GridSearchCV(estimator = rf,
                   param_grid ={
                           'max_depth':depths,
                           'n_estimators':estimators,
                           },
                           cv=3,verbose=0,
                           n_jobs=-1)

    grid_result = gsc.fit(X_train,Y_train)
    best_params = grid_result.best_params_
    #print(best_params)
    rfc = RandomForestClassifier(max_depth = best_params['max_depth'], n_estimators = best_params['n_estimators'],n_jobs=-1)
    return rfc

def train_rf_class(X,Y,rfc,n_splits):
    """Uses timeseries split to train the random forest classifier
        Inputs: X, Y: matrices of feature and label data
                rfc: random forest classifier (use output of rf_class_hyperparameters)
                n_splits: number of time series splits to run
        Returns: a fitted random forest classifier
    """
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        random_forest = RandomForestClassifier(max_depth=15,n_estimators=95,n_jobs=-1)
        random_forest.fit(X_train, Y_train)

        #print("Training score:",random_forest.score(X_train, Y_train))
        #print("Test score:",random_forest.score(X_test, Y_test))
    
    return random_forest

def main():
    df = pd.read_csv("finalproject_training.csv")
    X_regressor,X_classifier,Y_regressor,Y_classifier = data_prep(df)
    n_splits = 7 # optimal number of regressor splits found from tuning
    rf_regressor = rf_regressor_train(X_regressor,Y_regressor,n_splits)
    dump(rf_regressor, "rf_regressor.serialized")

    depths = [15,16]
    estimators = (94,95)
    #untrained_rf = rf_class_hyperparameters(X,Y,depths,estimators)
    untrained_rf = RandomForestClassifier(max_depth=15,n_estimators=95,n_jobs=-1)
    n_splits = 5 # optimal number of classifier splits found from tuning
    rf_classifier = train_rf_class(X_classifier,Y_classifier,untrained_rf,n_splits)
    dump(rf_classifier, "rf_classifier.serialized")

if __name__ == "__main__":
    main()
