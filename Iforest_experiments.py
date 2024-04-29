import os
import tensorflow as tf
import pandas as pd
import numpy as np 
import sys
from multiprocessing import Process

import matplotlib.pyplot as plt
import scipy.stats as st
import librosa
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from mtsa.metrics import calculate_aucroc
from mtsa.utils import files_train_test_split

import librosa
from mtsa import IForest

def run_iforest_experiment():

    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) 

    '''

    path_input_fan_id_00 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "fan", "id_00")
    path_input_fan_id_02 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "fan", "id_02")
    path_input_fan_id_04 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "fan", "id_04")
    path_input_fan_id_06 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "fan", "id_06")

    path_input_pump_id_00 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "pump", "id_00")
    path_input_pump_id_02 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "pump", "id_02")
    path_input_pump_id_04 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "pump", "id_04")
    path_input_pump_id_06 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "pump", "id_06")

    path_input_slider_id_00 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "slider", "id_00")
    path_input_slider_id_02 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "slider", "id_02")
    path_input_slider_id_04 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "slider", "id_04")
    path_input_slider_id_06 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "slider", "id_06")

    path_input_valve_id_00 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "valve", "id_00")
    path_input_valve_id_02 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "valve", "id_02")
    path_input_valve_id_04 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "valve", "id_04")
    path_input_valve_id_06 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "valve", "id_06")

    datapaths = np.array([
        [path_input_fan_id_06,'path_input_fan_id_06'], 
        [path_input_slider_id_06,'path_input_slider_id_06']
        ])
    n_estimators = np.array([100,200,1000])
    contaminations = np.array([0.1, 0.172])
    max_samples_group = np.array([256, 128])
    max_features_group = np.array([0.3, 0.5, 0.8, 1.0])
    validation = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    experiment01 = np.array([100, 0.1, 128, 0.3])
    experiment02 = np.array([100, 0.1, 128, 0.5])
    experiment03 = np.array([100, 0.1, 128, 0.8])
    experiment04 = np.array([100, 0.1, 128, 1.0])

    experiment05 = np.array([100, 0.1, 256, 0.3])
    experiment06 = np.array([100, 0.1, 256, 0.5])
    experiment07 = np.array([100, 0.1, 256, 0.8])
    experiment08 = np.array([100, 0.1, 256, 1.0])

    experiment09 = np.array([200, 0.1, 128, 0.3])
    experiment10 = np.array([200, 0.1, 128, 0.5])
    experiment11 = np.array([200, 0.1, 128, 0.8])
    experiment12 = np.array([200, 0.1, 128, 1.0])

    experiment13 = np.array([200, 0.1, 256, 0.3])
    experiment14 = np.array([200, 0.1, 256, 0.5])
    experiment15 = np.array([200, 0.1, 256, 0.8])
    experiment16 = np.array([200, 0.1, 256, 1.0])

    

    experiments = np.array([
         experiment01, 
         experiment02, 
         experiment03, 
         experiment04, 
         experiment05, 
         experiment06, 
         experiment07, 
         experiment08
         ])    
    
    from mtsa import files_train_test_split
    X_train, X_test, y_train, y_test = files_train_test_split(path_input_fan_id_06)
    
    X = np.array(X_train)

    y = np.array(y_train) 

    k=10
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    kf.get_n_splits(X)

    for data in datapaths:
        for parameters in experiments: 
            
            result=[]
            print('---'*20)
            print('KFold\t|RMSE\t|Variance score\t|AUC\t|')
            for i, (train_index, val_index) in enumerate(kf.split(X)):

                model_iforest = IForest(n_estimators=int(parameters[0]), contamination=parameters[1], max_samples=int(parameters[2]), max_features=parameters[3])

                X_train = X[train_index]
                X_val = X[val_index]
                y_train = y[train_index]
                y_val = y[val_index]
                
                # Treinamento do modelo de regressão linear
                model_iforest.fit(X_train, y_train)

                # Predição no conjunto de validação
                preditions_val = model_iforest.predict(X_val)
                #new_predition_val = np.where(preditions_val == -1, preditions_val, preditions_val + 1)

                # Avaliação dos resultados
                ### COMENZAR O CODIGO AQUI ### 
                rmse = mean_squared_error(y_val, preditions_val, squared=False)
                score = r2_score(y_val, preditions_val)
                auc = calculate_aucroc(model_iforest, X_val, y_val)
                score_samples = model_iforest.score_samples(X_val)

                print(f'score_samples: \n{score_samples}')

                experiment_dataframe = model_iforest.get_experiment_dataframe(data[1], f'n_estimator-{parameters[0]}_contamination-{parameters[1]}_max_samples-{parameters[2]}_max_features-{parameters[3]}')
                experiment_dataframe.loc['AUC_ROC'] = auc
                experiment_dataframe.loc['RMSE'] = rmse
                experiment_dataframe.loc['Score'] = score


                if os.path.isfile(f'n_estimator-{parameters[0]}_contamination-{parameters[1]}_max_samples-{parameters[2]}_max_features-{parameters[3]}.csv'):
                    df_existente = pd.read_csv(f'n_estimator-{parameters[0]}_contamination-{parameters[1]}_max_samples-{parameters[2]}_max_features-{parameters[3]}.csv')
                    experiment_dataframe = pd.concat([df_existente, experiment_dataframe], ignore_index=True)
                    experiment_dataframe.to_csv(f'n_estimator-{parameters[0]}_contamination-{parameters[1]}_max_samples-{parameters[2]}_max_features-{parameters[3]}.csv', sep=',', encoding='utf-8')
                else:
                    experiment_dataframe.to_csv(f'n_estimator-{parameters[0]}_contamination-{parameters[1]}_max_samples-{parameters[2]}_max_features-{parameters[3]}.csv', sep=',', encoding='utf-8')

                result.append([rmse,score,auc])  
                print( f'K({i}):\t|{rmse:0.5f}\t|{score:0.5f}\t|{auc:0.5f}\t|' )
                print('---'*20)
                '''
                X_train, X_test, y_train, y_test = files_train_test_split(data[0])
                model_IForest = IForest(n_estimators=n_estimator, contamination=contamination, max_samples=max_samples, max_features=max_features, n_jobs=3)
                model_IForest.fit(X_train, y_train) 
                features = []
                audio, sample_rate = librosa.load(X_train)

                spectral_features = extract_spectral_features(audio, sample_rate)
                temporal_features = extract_temporal_features(audio)
                all_features = np.concatenate([spectral_features, temporal_features])
                features.append(all_features)
                features = np.array(features)
                df = pd.DataFrame(features)

                experiment_dataframe = model_IForest.get_experiment_dataframe(data[1], f'n_estimator-{n_estimator}_contamination-{contamination}_max_samples-{max_samples}_max_features-{max_features}')
                auc = calculate_aucroc(model_IForest, X_test, y_test)
                experiment_dataframe.loc['AUC_ROC'] = auc
                if os.path.isfile(f'n_estimator-{n_estimator}_contamination-{contamination}_max_samples-{max_samples}_max_features-{max_features}.csv'):
                    df_existente = pd.read_csv(f'n_estimator-{n_estimator}_contamination-{contamination}_max_samples-{max_samples}_max_features-{max_features}.csv')
                    experiment_dataframe = pd.concat([df_existente, experiment_dataframe], ignore_index=True)
                    experiment_dataframe.to_csv(f'n_estimator-{n_estimator}_contamination-{contamination}_max_samples-{max_samples}_max_features-{max_features}.csv', sep=',', encoding='utf-8')
                else:
                    experiment_dataframe.to_csv(f'n_estimator-{n_estimator}_contamination-{contamination}_max_samples-{max_samples}_max_features-{max_features}.csv', sep=',', encoding='utf-8')
                '''
def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

if __name__ == '__main__':
    info('main line')
    p = Process(target=run_iforest_experiment)
    p.start()
    p.join()