import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin, check_array
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
    Array2Mfcc 
)
from mtsa.utils import (
    Wav2Array,
)

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
        self.dataset_name = None
        self.experiment_dataframe = self.__get_initialize_dataframe_experiments_result()
        self.model = self._build_model()

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
        return self.model.predict(X)
    
    def score(self, X, y=None):
        self.model.score(X)

    def score_samples(self, X):
        return self.model.score_samples(X=X)
    
    def __get_initialize_dataframe_experiments_result(self):
        parameters_columns = self.__get_parameters_columns()                              
        return pd.DataFrame(columns=parameters_columns)
    
    def __get_parameters_columns(self):
        parameters_columns = ["actual_dataset",
                              "parameters_names",
                              "n_estimators",
                              "max_samples",
                              "contamination",
                              "max_features",
                              "RMSE",
                              "Score",
                              "AUC_ROC",
                            ]
        return parameters_columns   

    def __create_dataframe(self):
        self.experiment_dataframe.loc[len(self.experiment_dataframe)] = {
            "actual_dataset": self.dataset_name,
            "parameters_names": self.model_parameters_names,
            "n_estimators": self.n_estimators, 
            "max_samples": self.max_samples, 
            "contamination": self.contamination, 
            "max_features": self.max_features,
            "RMSE": None,
            "Score": None,
            "AUC_ROC": None
            } 
    
    def get_experiment_dataframe(self, dataset_name=None, model_parameters_names=None):
        self.dataset_name = dataset_name
        self.model_parameters_names = model_parameters_names
        self.__create_dataframe()
        return self.experiment_dataframe

    def _build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate)
        #array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        #features = FeatureUnion(self.features)
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
                #("array2mfcc", array2mfcc),
                #("features", features),
                ("final_model", self.final_model),
                ]
            )
        
        return model

