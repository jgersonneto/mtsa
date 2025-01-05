import os
import tensorflow as tf
import pandas as pd 
import glob
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import itertools
import openpyxl
import scipy.stats as st


import sys
from multiprocessing import Process

import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)


def run_isolation_forest_result_show():

    
    path_csv1 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Test_Without_filter", "new", "100", "*.csv")
    path_csv2 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Test_Without_filter", "new", "50", "*.csv")
    path_csv3 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Test_Without_filter", "new", "30", "*.csv")
    path_csv4 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Test_Without_filter", "new", "10", "*.csv")
    path_csv5 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Test_Without_filter", "new", "7", "*.csv")
    path_csv6 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Test_Without_filter", "new", "5", "*.csv")
    path_csv7 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "OnlyWavelet", "WaveletDatails", "5", "*.csv")
    path_csv8 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "OnlyWavelet", "WaveletDatails", "30", "*.csv")
    path_csv9 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "OnlyWavelet", "WaveletDatails", "50", "*.csv")
    path_csv10 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "OnlyWavelet", "WaveletDetailsAndApproach", "*.csv")
    path_csv11 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "OnlyWavelet", "WaveletDatails", "5new01", "*.csv")
    path_csv12 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "OnlyWavelet", "WaveletDatails", "5new01", "y_val", "*.csv")
    path_csv13 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "MFCC", "01", "*.csv")
    path_csv14 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "MFCC", "01", "y_val", "*.csv")
    path_csv15 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "WaveletMfcc", "*.csv")
    path_csv16 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "WaveletMfcc", "y_val", "*.csv")
    path_csv17 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "WaveletMfcc", "01", "*.csv")
    path_csv18 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "WaveletMfcc", "01", "y_val", "*.csv")
    path_csv19 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "WaveletMfcc", "02", "*.csv")
    path_csv20 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "Wavelet", "WaveletMfcc", "02", "y_val", "*.csv")
    path_csv21 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "MFCC", "*.csv")
    path_csv22 = os.path.join(os.getcwd(),  "mtsa", "experiments_result", "MFCC", "y_val", "*.csv")
        
    files_csv1 = glob.glob(path_csv15)
    files_csv2 = glob.glob(path_csv16)
    #ShowTableEachParameterCombination(files_csv1)
    
    df_new, y_val, preditions_val = ShowTableEachNEstimator(files_csv1, files_csv2)

    ShowConfidenceInterval2(df_new)
    preditions_val = np.array(preditions_val)

    mc = metrics.confusion_matrix(y_val, preditions_val)

    classes = ['Anormal (0)', 'Normal (1)']
    #sns.heatmap(mc, cbar=False, annot=True, cmap="Blues", fmt="d")
    plot_confusion_matrix(mc, classes=classes, title='Matriz Confusão', normalize=False)
    

def ShowTableEachParameterCombination(files_csv):
    for data in files_csv:

        df = pd.read_csv(data)

        df_new = pd.DataFrame(columns=['Parameters', 'Time_Execution', 'ci_lower', 'ci_upper', 'ACC', 'Precision', 'Recall', 'F1_Score', 'AUC'])

        ci_lower, ci_upper = st.t.interval(confidence=0.95, 
                                       df=len(df['AUC_ROC'])-1, 
                                       loc=np.mean(df['AUC_ROC']), 
                                       scale=st.sem(df['AUC_ROC']))

        auc_mean = round(df['AUC_ROC'].mean(),2)
        time_execution_mean = round(df['execution_time'].mean(),2)
        acc = round(df['ACC'].mean(), 2)
        precision = round(df['Precision'].mean(), 2)
        recall = round(df['Recall'].mean(), 2)
        f1_socre = round(df['F1_Score'].mean(), 2)
        
        df_new.loc[len(df_new)] = {
            "Parameters": df['parameters_names'][0].split()[0],
            "Time_Execution": time_execution_mean,
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            'ACC': acc, 
            'Precision': precision, 
            'Recall': recall,
            'F1_Score': f1_socre,
            "AUC": auc_mean
            } 
                
        TablePlotterShow(df_new)


        df_new.to_excel('tabela_resultados.xlsx', index=False)

        parameter = df_new['Parameters'][0].split()[0]
        time = time_execution_mean
        auc = auc_mean

        print('---'*20)
        print('KFold\t|PARAMETERS\t|EXECUTION_TIME\t|AUC\t|')
        print( f'\t|{parameter}\t|{time}\t|{auc}\t|' )
        print('---'*20)

def ShowConfidenceInterval(df):
    test_name = ['test01','test02','test03','test04']
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_name, df['AUC'], marker="o", color="b")
    plt.fill_between(test_name, df['ci_lower'], df['ci_upper'], color="green", alpha=0.3)
    plt.xlabel("Test")
    plt.ylabel("AUC")
    plt.title("AUC confidence interval")
    plt.grid(True)
    plt.show()

def ShowConfidenceInterval2(df):
    test_name = ['n_estimator=5\nmax_samples=128\nmax_features=0.3','n_estimator=5\nmax_samples=128\nmax_features=1.0','n_estimator=5\nmax_samples=256\nmax_features=0.5','n_estimator=5\nmax_samples=256\nmax_features=1.0']
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_name, df['AUC'], marker="o", color="b")

    a=0
    # Adicionando linhas verticais para representar a média de AUC para cada modelo    
    for row in df.itertuples(index=False):
        plt.plot([a, a], [row.ci_lower, row.ci_upper], color='r', linestyle='-', lw=2, label=f'Segmento Vertical em X={a}')
        a=a+1
    

    plt.xlabel("Parameters")
    plt.ylabel("AUC")
    plt.title("AUC confidence interval")
    plt.grid(True)


    # Salvar a figura como uma imagem 
    parameter_name = 'MFCC'
    image_name = f'{parameter_name}.png'
    plt.savefig(image_name, bbox_inches='tight', dpi=500)

    plt.show()

def ShowTableEachNEstimator(files_csv1, files_csv2):


    df_new = pd.DataFrame(columns=['Parameters', 'Time_Execution', 'ci_lower', 'ci_upper', 'ACC', 'Precision', 'Recall', 'F1_Score', 'AUC'])

    for data in files_csv1:

        df = pd.read_csv(data)        

        ci_lower, ci_upper = st.t.interval(confidence=0.95, 
                                       df=len(df['AUC_ROC'])-1, 
                                       loc=np.mean(df['AUC_ROC']), 
                                       scale=st.sem(df['AUC_ROC']))

        auc_mean = round(df['AUC_ROC'].mean(),2)
        time_execution_mean = round(df['execution_time'].mean(),2)
        acc = round(df['ACC'].mean(), 2)
        precision = round(df['Precision'].mean(), 2)
        recall = round(df['Recall'].mean(), 2)
        f1_socre = round(df['F1_Score'].mean(), 2)
        
        df_new.loc[len(df_new)] = {
            "Parameters": df['parameters_names'][0].split()[0],
            "Time_Execution": time_execution_mean,
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            'ACC': acc, 
            'Precision': precision, 
            'Recall': recall,
            'F1_Score': f1_socre,
            "AUC": auc_mean
            }  
                

        parameter = df_new['Parameters'][0].split()[0]
        time = time_execution_mean
        auc = auc_mean

        print('---'*20)
        print('KFold\t|PARAMETERS\t|EXECUTION_TIME\t|AUC\t|')
        print( f'\t|{parameter}\t|{time}\t|{auc}\t|' )
        print('---'*20)

    TablePlotterShow(df_new)
    
    df_new.to_excel('tabela_resultados.xlsx', index=False)
    
    y_val, preditions_val = Get_yval_Preditions(files_csv2)

    return df_new, y_val, preditions_val

def Get_yval_Preditions(files_csv2):

    y_val = None
    preditions_val = None
    for data in files_csv2:
        df = pd.read_csv(data)
        y_val = np.array(df['y_val'])
        preditions_val = np.array(df['preditions_val'])

    return y_val, preditions_val


def TablePlotterShow(df):

    # Criar uma figura e um eixo
    fig, ax = plt.subplots(figsize=(16, 3))

    # Ocultar o eixo 
    ax.axis('tight') 
    ax.axis('off') 

    # Criar a tabela e adicioná-la à figura 
    tabela = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center') 

    tabela.scale(1, 1.5)

    for (i, j), cell in tabela.get_celld().items():
        cell.set_text_props(fontsize=18, fontweight='bold')

    # Salvar a figura como uma imagem 
    parameter_name = df['Parameters'][0].split()[0]
    image_name = f'{parameter_name}.png'
    plt.savefig(image_name, bbox_inches='tight', dpi=500) 
    
    # Mostrar a figura 
    plt.show()

def find_outlier_bounds_iqr(column): 
    Q1 = column.quantile(0.25) 
    Q3 = column.quantile(0.75) 
    IQR = Q3 - Q1 
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR 
    return lower_bound, upper_bound


import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    """
    Esta função imprime e plota a matriz de confusão.
    A normalização pode ser aplicada configurando `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')

    # Salvar a figura como uma imagem
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)

    plt.grid(False)
    plt.show()
    plt.clf()  # Limpar a figura após salvar


def graphAUC():
    plt.figure(figsize=(10, 6))
    sns.pointplot(x="parameters_names", y="mean_AUC_ROC", data=df, ci=None)
    plt.errorbar(range(len(df)), df["mean_AUC_ROC"], 
                yerr=[df["mean_AUC_ROC"] - df["ci_lower"], df["ci_upper"] - df["mean_AUC_ROC"]],
                fmt='o', color='black')
    plt.title('Mean AUC-ROC with Confidence Interval')
    plt.ylabel('Mean AUC-ROC')
    plt.xlabel('Parameters')
    plt.show()

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

if __name__ == '__main__':
    info('main line')
    p = Process(target=run_isolation_forest_result_show)
    p.start()
    p.join()