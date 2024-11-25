import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin, check_array
import sklearn.metrics as metrics
from sklearn.pipeline import (
    Pipeline, 
    FeatureUnion
) 
from mtsa.features.stats import (
    MagnitudeMeanFeatureMfcc, 
    MagnitudeStdFeatureMfcc, 
    CorrelationFeatureMfcc,
    FEATURES,
    get_features
    )

from mtsa.features.mel import (
    Array2Mfcc,
    Array2Wavelet,
    Array2MfccWavelet,
    Array2Mfcc2Wavelet 
)
from mtsa.utils import (
    Wav2Array,
)

from mtsa.metrics import calculate_aucroc

from sklearn.ensemble import IsolationForest
from functools import reduce

class IForest(BaseEstimator, OutlierMixin):

    def __init__(self,
                 n_estimators=200,
                 max_samples="auto",
                 contamination='auto',
                 max_features=1.0,
                 bootstrap=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 final_model=None, 
                 features=FEATURES,
                 sampling_rate=None,
                 ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.sampling_rate = sampling_rate
        self.final_model = final_model
        self.features = features
        self.model_parameters_names = None
        self.fit_execution_time = None
        self.auc_execution_time = None
        self.execution_time = None
        self.acc = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.auc = None
        self.dataset_name = None
        self.experiment_dataframe = None
        self.model = self._build_model()
        self.model2 = self._build_model()

    @property
    def name(self):
        return "IsolationForest " + "+".join([f[0] for f in self.features])
        
    def fit(self, X, y=None):        
        return self.model.fit(X, y)

    def transform(self, X, y=None):
        l = list()
        l.append(X)
        l.extend(self.model.steps[:-1])
        Xt = reduce(lambda x, y: y[1].transform(x), l)
        return Xt
    
    def predict(self, X):
        predict = self.model.predict(X)                
        return np.where(predict == -1, 0, predict)
    
    def score(self, X, y=None):
        self.model.score(X)

    def score_samples(self, X):
        return self.model.score_samples(X=X)
    
    def decision_function(self, X):
        return self.model.decision_function(X=X)
    
    def __get_initialize_dataframe_experiments_result(self):
        parameters_columns = self.__get_parameters_columns()                              
        return pd.DataFrame(columns=parameters_columns)
    
    def __get_parameters_columns(self):
        parameters_columns = [
                              "actual_dataset",
                              "parameters_names",
                              "n_estimators",
                              "max_samples",
                              "contamination",
                              "max_features",
                              'fit_execution_time', 
                              'auc_execution_time', 
                              'execution_time',
                              'ACC',
                              'Precision',
                              'Recall',
                              'F1_Score',
                              "AUC_ROC",
                            ]
        return parameters_columns   

    def __create_dataframe(self):
        self.experiment_dataframe = self.__get_initialize_dataframe_experiments_result()
        self.experiment_dataframe.loc[len(self.experiment_dataframe)] = {
            "actual_dataset": self.dataset_name,
            "parameters_names": self.model_parameters_names,
            "n_estimators": self.n_estimators, 
            "max_samples": self.max_samples, 
            "contamination": self.contamination, 
            "max_features": self.max_features,
            'fit_execution_time': self.fit_execution_time, 
            'auc_execution_time': self.auc_execution_time, 
            'execution_time': self.execution_time,
            'ACC': self.acc,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1_Score': self.f1_score,
            "AUC_ROC": self.auc
            } 
    
    def get_experiment_dataframe(
            self, 
            dataset_name=None, 
            model_parameters_names=None, 
            fit_execution_time=None, 
            auc_execution_time=None, 
            execution_time=None, 
            acc=None, 
            precision=None,
            recall=None,
            f1_score=None,
            auc=None):
        self.dataset_name = dataset_name
        self.model_parameters_names = model_parameters_names
        self.auc = auc
        self.fit_execution_time = fit_execution_time
        self.auc_execution_time = auc_execution_time
        self.execution_time = execution_time
        self.acc = acc
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.__create_dataframe()
        return self.experiment_dataframe
    
    def evaluation(self, X_val,y_val):
        
        acc = metrics.accuracy_score(y_val,self.predict(X_val))
        precision = metrics.precision_score(y_val, self.predict(X_val))
        recall = metrics.recall_score(y_val,self.predict(X_val))
        f1_score = 2*precision*recall/(precision+recall)

        return {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def _build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate)
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        array2wavelet = Array2Wavelet(sampling_rate=self.sampling_rate)
        array2mfccwavelet = Array2MfccWavelet()
        array2mfcc2wavelet = Array2Mfcc2Wavelet()
        features = FeatureUnion(self.features)
        self.final_model = IsolationForest(
            n_estimators=self.n_estimators, 
            max_samples=self.max_samples, 
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs, 
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
        )
        
        model = Pipeline(
            steps=[
                ("wav2array", wav2array),
                #("array2wavelet", array2wavelet),
                #("array2mfcc", array2mfcc),
                #("array2mfccwavelet", array2mfccwavelet),
                ("array2mfcc2wavelet", array2mfcc2wavelet),
                #("features", features),
                ("final_model", self.final_model),
                ]
            )
        
        return model

