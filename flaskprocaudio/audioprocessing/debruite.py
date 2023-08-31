import numpy as np
import warnings
import soundfile as snd
import resampy
from STFT import Stft, Istft
import os
from .snmf import BetanmfSparse


def rundebruitage(audio_file, model_file='model.npy'):
    print("Process denoising")
    # ########@# #
    # parameters #
    # ########@ ##
    # nb_iter = 500  # number of nmf iterations
    noise_floor = 1e-10  # small constant in IS nmf
    small_const_add_spec = 1e-10  # add this small constant to input spectrogram
    # to avoid numerical problems if spec == 0

    # 1) load data file
    try:
        w_music = np.load(model_file)
    except:
        warnings.warn(f'Failed to load model file {model_file}')
        return None

    # 2) load audio file
    try:
        y, fs = snd.read(audio_file)
    except:
        warnings.warn(f'Failed to read audio file {audio_file}')
        return None

    print('file : %s' % audio_file)
    print('sampling frequency : %d' % fs)

    # down to mono channel
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # resample if necessary
    if fs != 22050:
        print("resampling to 22kHz")
        y = resampy.resample(y, fs, 22050)
        fs = 22050

    # STFT parameters
    Ksic = 10
    Kspeech = 15
    Beta = 1.  # Kullback-Leibler divergence

    wlen_millisec = 50  # in milliseconds

    nb_iter = 500
    use_mu_nmf = True  # uses multiplicative update rules if True    
    wlen = int(np.floor(fs * wlen_millisec / 1000.))
    print('Window length %d ms, %d samples' % (int(np.floor(wlen_millisec)), wlen))

    hlen = wlen // 2
    n_fft = 1
    while n_fft < wlen:
        n_fft *= 2
    print('fft size : %d' % n_fft)

    # STFT
    X, f, t = Stft(y, wlen, hlen, n_fft, fs)
    spec = np.abs(X)  # magnitude spectrogram
    spec += small_const_add_spec
    # F, N = spec.shape
    print('running an input matrix of size {}'.format(spec.shape))

    # 3) global NMF
    F, N = spec.shape
    w_speech = np.abs(np.random.randn(F, Kspeech)) + 1e-8
    w_all_init = np.hstack((w_speech, w_music))
    w_all = np.copy(w_all_init)
    ind_w = range(Kspeech)  # update speech only

    w_all, h_all, v_all, _ = BetanmfSparse(spec, W=w_all, indW=ind_w,
                                            Beta=Beta, nbIter=nb_iter,
                                            noiseFloor=noise_floor,
                                            sparseType='None', LRupdate=False)
    # TODO : play with sparseType

    # 5) Wiener filtering
    w_speech_out = w_all[:, ind_w]
    h_speech_out = h_all[ind_w, :]
    v_speech_out = w_speech_out.dot(h_speech_out) + noise_floor

    X_speech = X*v_speech_out/v_all

    # 6) inverse STFT
    x_speech, _ = Istft(X_speech, wlen, hlen, n_fft, fs)

    # 7) save audio
    filename = os.path.basename(audio_file)
    outputFile = os.path.join(os.path.dirname(audio_file), filename[:-4] + '_denoised.wav')
    snd.write(outputFile, x_speech, fs)
    return 'success'
