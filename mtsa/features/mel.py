"""Base classes for all mfcc estimators."""

import numpy as np
import sys
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats as st
import librosa as lib
import pywt 
import scipy
from librosa import util
import pywt as wavelet
from sklearn.preprocessing import StandardScaler
from functools import reduce

class Array2MelSpec(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 sampling_rate,
                 n_fft,
                 hop_length,
                 n_mels,
                 frames,
                 power
                 ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.frames = frames
        self.power = power

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        
        def normalize_melspec(mel_spectrogram):

            # 03 convert melspectrogram to log mel energy
            log_mel_spectrogram = 20.0 / self.power * np.log10(mel_spectrogram + sys.float_info.epsilon)

            # 04 calculate total vector size
            vectorarray_size = len(log_mel_spectrogram[0, :]) - self.frames + 1

            # 05 skip too short clips
            if vectorarray_size < 1:
                return np.empty((0, dims), float)

            # 06 generate feature vectors by concatenating multi_frames
            dims = self.n_mels * self.frames
            vectorarray = np.zeros((vectorarray_size, dims), float)
            for t in range(self.frames):
                vectorarray[:, self.n_mels * t: self.n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

            return vectorarray
            
        def extract_melspec(y):
            params = {
                'y':y, 
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                "n_mels": self.n_mels,
                "power": self.power
            }
            if self.sampling_rate:
                params['sr']=self.sampling_rate
            
            mel_spectrogram = lib.feature.melspectrogram(**params)
            
            normalized_melspec = normalize_melspec(mel_spectrogram)

            return normalized_melspec
        Xt = np.array(
            reduce(
                lambda n1, n2: np.concatenate([n1,n2]), 
                map(extract_melspec, X))
        )
        return Xt
    
    
class Array2Mfcc(BaseEstimator, TransformerMixin):
    """
     Gets a numpy array containing audio signals and transforms it into a numpy array containing mfcc signals
     
    """
    
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        def extract_mfcc(y):
            if self.sampling_rate: 
                return lib.feature.mfcc(y=y, sr=self.sampling_rate)
            else: 
                return lib.feature.mfcc(y=y)
        Xt = np.array(list(map(extract_mfcc, X)))
        return Xt
    
class Array2Wavelet(BaseEstimator, TransformerMixin):
    """
     Gets a numpy array containing audio signals and transforms it into a numpy array containing Wavelet signals
     
    """
    
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        def extract_wavelet(y):
            coeffs = wavelet.wavedec(data=y, wavelet='db4', level=3)
            cA = coeffs[0]
            cD1 = coeffs[1]
            cD2 = coeffs[2]
            cD3 = coeffs[3]
            cD = np.concatenate((cD1, cD2, cD3))
            return cD
        Xt = np.array(list(map(extract_wavelet, X)))  
        return Xt

class Array2MfccWavelet(BaseEstimator, TransformerMixin):
    """
     Gets a numpy array containing audio signals and transforms it into a numpy array containing mfcc signals
     
    """
    
    def __init__(self, lifter=0, n_mfcc=20, dct_type=2, norm="ortho", S=None,):
        self.lifter = lifter
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.S = S

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):

        def extract_mfcc_wavelet(y):

            if self.S is None:
                
                # Aplicar a transformada wavelet discreta usando db4
                coeffs = wavelet.wavedec(data=y, wavelet='db4', level=3)

                # Extrair coeficientes de detalhe (ignorando o coeficiente de aproximação)
                detail_coeffs = np.concatenate(coeffs[1:])

                sr = 22050
                # Calcular as frequências centrais dos filtros Mel 
                mel_filters = lib.filters.mel(sr=sr, n_fft=len(detail_coeffs), n_mels=40) 
                mel_coeffs = np.dot(mel_filters, np.abs(detail_coeffs[:len(mel_filters[0])]))

                # Calcular o log-power dos coeficientes de detalhe
                log_power_coeffs = np.log(np.abs(mel_coeffs) + 1e-10)

                # Aplicar a transformada de cosseno discreta (DCT) para obter coeficientes cepstrais
                num_ceps = 13  # Número de coeficientes desejados
                Xt = scipy.fftpack.dct(log_power_coeffs, type=2, norm='ortho')[:self.n_mfcc]
            return Xt
        
        Xt = np.array(list(map(extract_mfcc_wavelet, X)))
        #Xt = np.nan_to_num(Xt)

        if self.lifter > 0:
            
            # shape lifter for broadcasting
            LI = np.sin(np.pi * np.arange(1, 1 + self.n_mfcc, dtype=Xt.dtype) / self.lifter)
            LI = util.expand_to(LI, ndim=self.S.ndim, axes=-2)

            Xt *= 1 + (self.lifter / 2) * LI
            
            return Xt
        
        elif self.lifter == 0:
            return Xt
        else:
            raise lib.ParameterError(f"MFCC lifter={self.lifter} must be a non-negative number")
        

class Array2Mfcc2Wavelet(BaseEstimator, TransformerMixin):
    """
     Gets a numpy array containing audio signals and transforms it into a numpy array containing mfcc signals
     
    """
    
    def __init__(
                 self,
                 lifter=0, 
                 n_mfcc=20, 
                 dct_type=2, 
                 norm="ortho", 
                 S=None,
                 n_mels=40,
                 frames=5,
                 power=2.0):
        self.lifter = lifter
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.S = S
        self.n_mels = n_mels
        self.frames = frames
        self.power = power

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):

        def extract_mfcc_wavelet(y):

            if self.S is None:
                
                # Aplicar a transformada wavelet discreta usando db4
                coeffs = wavelet.wavedec(data=y, wavelet='db4', level=3)

                # Extrair coeficientes de detalhe (ignorando o coeficiente de aproximação)
                detail_coeffs = np.concatenate(coeffs[1:])

                sr = 22050
                # Calcular as frequências centrais dos filtros Mel 
                mel_filters = lib.filters.mel(sr=sr, n_fft=len(detail_coeffs), n_mels=40) 
                mel_coeffs = np.dot(mel_filters, np.abs(detail_coeffs[:len(mel_filters[0])]))

                # Calcular o log-power dos coeficientes de detalhe
                log_power_coeffs = np.log(np.abs(mel_coeffs) + 1e-10)

                # Aplicar a transformada de cosseno discreta (DCT) para obter coeficientes cepstrais
                num_ceps = 13  # Número de coeficientes desejados
                Xt = scipy.fftpack.dct(log_power_coeffs, type=2, norm='ortho')[:self.n_mfcc]
            return Xt
        
        def normalize_melspec(mel_spectrogram):

            sr = 22050

            mel_filters = lib.filters.mel(sr=sr, n_fft=len(mel_spectrogram), n_mels=40)

            mel_coeffs = np.dot(mel_filters, np.abs(mel_spectrogram[:len(mel_filters[0])]))

            # 03 convert melspectrogram to log mel energy
            log_mel_spectrogram = 20.0 / self.power * np.log10(mel_coeffs + sys.float_info.epsilon)

            
            return log_mel_spectrogram
            
        def extract_melspec(y=None):
            
            coeffs = wavelet.wavedec(data=y, wavelet='db4', level=3)
            cA = coeffs[0]
            cD1 = coeffs[1]
            cD2 = coeffs[2]
            cD3 = coeffs[3]
            cD = np.concatenate((cD1, cD2, cD3))
                        
            mel_spectrogram = np.abs(cD)
            
            normalized_melspec = normalize_melspec(mel_spectrogram)

            return normalized_melspec
        Xt = np.array(list(map(extract_melspec, X)))
        return Xt
        
        