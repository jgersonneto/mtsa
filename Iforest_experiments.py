import os
import tensorflow as tf
import pandas as pd
import numpy as np 
import time
import sys
from multiprocessing import Process

import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from mtsa.metrics import calculate_aucroc
from mtsa.utils import files_train_test_split

import librosa
from mtsa.models import IForest
from mtsa.models import Hitachi

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
    path_input_fan_id_06 = os.path.join(os.getcwd(),  "MIMII", "fan", "id_06")

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
        ])
    n_estimators = np.array([100,200,1000])
    contaminations = np.array([0.1, 0.172])
    max_samples_group = np.array([256, 128])
    max_features_group = np.array([0.3, 0.5, 0.8, 1.0])
    validation = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    experiment01 = np.array([5, 0.1, 128, 0.3])
    experiment02 = np.array([5, 0.1, 128, 0.5])
    experiment03 = np.array([5, 0.1, 128, 0.8])
    experiment04 = np.array([5, 0.1, 128, 1.0])

    experiment05 = np.array([5, 0.1, 256, 0.3])
    experiment06 = np.array([5, 0.1, 256, 0.5])
    experiment07 = np.array([5, 0.1, 256, 0.8])
    experiment08 = np.array([5, 0.1, 256, 1.0])

    experiment09 = np.array([7, 0.1, 128, 0.3])
    experiment10 = np.array([7, 0.1, 128, 0.5])
    experiment11 = np.array([7, 0.1, 128, 0.8])
    experiment12 = np.array([7, 0.1, 128, 1.0])

    experiment13 = np.array([7, 0.1, 256, 0.3])
    experiment14 = np.array([7, 0.1, 256, 0.5])
    experiment15 = np.array([7, 0.1, 256, 0.8])
    experiment16 = np.array([7, 0.1, 256, 1.0])

    experiment17 = np.array([10, 0.1, 128, 0.3])
    experiment18 = np.array([10, 0.1, 128, 0.5])
    experiment19 = np.array([10, 0.1, 128, 0.8])
    experiment20 = np.array([10, 0.1, 128, 1.0])

    experiment21 = np.array([10, 0.1, 256, 0.3])
    experiment22 = np.array([10, 0.1, 256, 0.5])
    experiment23 = np.array([10, 0.1, 256, 0.8])
    experiment24 = np.array([10, 0.1, 256, 1.0])

    experiment25 = np.array([30, 0.1, 128, 0.3])
    experiment28 = np.array([30, 0.1, 128, 1.0])

    experiment29 = np.array([30, 0.1, 256, 0.3])
    experiment32 = np.array([30, 0.1, 256, 1.0])

    experiment33 = np.array([50, 0.1, 128, 0.3])
    experiment34 = np.array([50, 0.1, 128, 1.0])

    experiment35 = np.array([50, 0.1, 256, 0.3])
    experiment36 = np.array([50, 0.1, 256, 1.0])

    

    experiments = np.array([         
         experiment08    
         ])
    
    #from mtsa import files_train_test_split
    X_train, X_test, y_train, y_test = files_train_test_split(path_input_fan_id_06)
    
    X = np.array(X_train)
    y = np.array(y_train)
    X_valid = np.array(X_test)
    y_valid = np.array(y_test)
    model_iforest = None
    
    k=10
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    kf.get_n_splits(X)

    for data in datapaths:
        for parameters in experiments: 
            result=[]
            print('---'*20)
            print('KFold\t|AUC\t|FIT_EXECUTION_TIME\t|AUC_EXECUTION_TIME\t|')
            for i, (train_index, val_index) in enumerate(kf.split(X)):

                X_train = X[train_index]
                X_val = X_valid[val_index]
                y_train = y[train_index]
                y_val = y_valid[val_index]


                #IsolationForest(data, parameters, X, y, X_valid, y_valid, result)


                GetHitachi(data, parameters, X, y, X_valid, y_valid, result)

                
                

    dados = { 
        'y_val': y_valid, 
        'preditions_val': model_iforest.predict(X_valid) 
        } 
    # Criando o DataFrame 
    df = pd.DataFrame(dados)    

    df.to_csv(f'y_val.csv', sep=',', encoding='utf-8', index=False)    

def IsolationForest(data, parameters, X, y, X_valid, y_valid, result):
    model_iforest = IForest(n_estimators=int(parameters[0]), contamination=parameters[1], max_samples=int(parameters[2]), max_features=parameters[3])

    # Treinamento do modelo de Isolation Forest
    fit_execution_time = IsolationForesModelTrain(model_iforest,X)

    # Predição no conjunto de validação
    #preditions_val = model_iforest.predict(X_val)
    #new_predition_val = np.where(preditions_val == -1, preditions_val, preditions_val + 1)
    #print(f'preditions_val: \n{preditions_val}\n')       

    # Avaliação dos resultados
    ### COMENZAR O CODIGO AQUI ### 
    #rmse = mean_squared_error(y_val, preditions_val, squared=False)
    #score = r2_score(y_val, preditions_val)
    #score_samples = model_iforest.score_samples(X_val)

    #acc = model_iforest.evaluation(X_valid, y_valid)
    auc_execution_time_start = time.time()

    acc = metrics.accuracy_score(y_valid,model_iforest.predict(X_valid))
    precision = metrics.precision_score(y_valid, model_iforest.predict(X_valid))
    recall = metrics.recall_score(y_valid,model_iforest.predict(X_valid))
    f1_score = 2*precision*recall/(precision+recall)

    auc = calculate_aucroc(model_iforest, X_valid, y_valid)

    auc_execution_time_end = time.time()
    auc_execution_time = auc_execution_time_end - auc_execution_time_start

    printMetrics = {
                        'acc': acc,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score
                    }
    
    #experiment_dataframe.loc['AUC_ROC'] = auc

    file_path = f'n_estimator-{parameters[0]}_contamination-{parameters[1]}_max_samples-{parameters[2]}_max_features-{parameters[3]}.csv'
    if os.path.isfile(file_path):
        df_existente = pd.read_csv(file_path)
        df_existente.loc[len(df_existente)] = {
                                                'actual_dataset': data[1], 
                                                'parameters_names': f'n_estimator-{parameters[0]}_contamination-{parameters[1]}_max_samples-{parameters[2]}_max_features-{parameters[3]}', 
                                                'n_estimators': parameters[0], 
                                                'max_samples': parameters[2], 
                                                'contamination': parameters[1], 
                                                'max_features': parameters[3], 
                                                'fit_execution_time': fit_execution_time, 
                                                'auc_execution_time': auc_execution_time, 
                                                'execution_time': fit_execution_time + auc_execution_time,
                                                'ACC': acc,
                                                'Precision': precision,
                                                'Recall': recall,
                                                'F1_Score': f1_score,
                                                'AUC_ROC': auc
                                                }
        #experiment_dataframe = pd.concat([df_existente, experiment_dataframe], ignore_index=True)
        df_existente.to_csv(file_path, sep=',', encoding='utf-8', index=False)
    else:
        execution_time = fit_execution_time + auc_execution_time
        experiment_dataframe = model_iforest.get_experiment_dataframe(data[1], f'n_estimator-{parameters[0]}_contamination-{parameters[1]}_max_samples-{parameters[2]}_max_features-{parameters[3]}', fit_execution_time, auc_execution_time, execution_time, acc, precision, recall, f1_score, auc)
        experiment_dataframe.to_csv(file_path, sep=',', encoding='utf-8', index=False)

    result.append([auc])  
    print( f'K({i}):\t|{auc:0.5f}\t|{fit_execution_time:0.5f}\t|{auc_execution_time:0.5f}\t|' )
    print( f'K({i}):\t|{printMetrics}\t|' )
    print('---'*20)

def GetHitachi(data, parameters, X, y, X_valid, y_valid, result):
    model_hitachi = Hitachi()

     # Treinamento do modelo de Hitachi
    fit_execution_time = HitachiModelTrain(model_hitachi,X, y)

    auc_execution_time_start = time.time()

    acc = metrics.accuracy_score(y_valid,model_hitachi.predict(X_valid))
    precision = metrics.precision_score(y_valid, model_hitachi.predict(X_valid))
    recall = metrics.recall_score(y_valid,model_hitachi.predict(X_valid))
    f1_score = 2*precision*recall/(precision+recall)

    auc = calculate_aucroc(model_hitachi, X_valid, y_valid)

    auc_execution_time_end = time.time()
    auc_execution_time = auc_execution_time_end - auc_execution_time_start

    printMetrics = {
                        'acc': acc,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score
                    }
    
    #experiment_dataframe.loc['AUC_ROC'] = auc

    file_path = 'Hitachi.csv'
    if os.path.isfile(file_path):
        df_existente = pd.read_csv(file_path)
        df_existente.loc[len(df_existente)] = {
                                                'actual_dataset': data[1], 
                                                'parameters_names': f'hitachi', 
                                                'n_estimators': '', 
                                                'max_samples': '', 
                                                'contamination': '', 
                                                'max_features': '', 
                                                'fit_execution_time': fit_execution_time, 
                                                'auc_execution_time': auc_execution_time, 
                                                'execution_time': fit_execution_time + auc_execution_time,
                                                'ACC': acc,
                                                'Precision': precision,
                                                'Recall': recall,
                                                'F1_Score': f1_score,
                                                'AUC_ROC': auc
                                                }
        #experiment_dataframe = pd.concat([df_existente, experiment_dataframe], ignore_index=True)
        df_existente.to_csv(file_path, sep=',', encoding='utf-8', index=False)
    else:
        execution_time = fit_execution_time + auc_execution_time
        experiment_dataframe = model_iforest.get_experiment_dataframe(data[1], 'hitachi', fit_execution_time, auc_execution_time, execution_time, acc, precision, recall, f1_score, auc)
        experiment_dataframe.to_csv(file_path, sep=',', encoding='utf-8', index=False)

    result.append([auc])  
    print( f'K({i}):\t|{auc:0.5f}\t|{fit_execution_time:0.5f}\t|{auc_execution_time:0.5f}\t|' )
    print( f'K({i}):\t|{printMetrics}\t|' )
    print('---'*20)

def IsolationForesModelTrain(model, X):
    fit_execution_time_start = time.time()

    model.fit(X)

    fit_execution_time_end = time.time()

    return fit_execution_time_end - fit_execution_time_start

def HitachiModelTrain(model, X, y):
    fit_execution_time_start = time.time()

    model.fit(X, y)

    fit_execution_time_end = time.time()

    return fit_execution_time_end - fit_execution_time_start

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