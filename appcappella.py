# -*- coding: utf-8 -*-
"""
GUI implementation of NMF applied to Acappella denoising.
"""

# Author : Elie Laurent Benaroya
# laurent.benaroya@gmail.com
# 3/2019

# license : GNU GPL v3

import wx
import soundfile as snd
import resampy

import os.path
import numpy as np

from STFT import Stft, Istft
from nmf.snmf import BetanmfSparse


MAIN_WINDOW_DEFAULT_SIZE = (600, 280)

# TODO : mettre sur gitlab et mail à Acappella ou à Ben Mazue
# TODO : jouer avec les parametres de la NMF (sur sommeil_heureux.wav)
# TODO : le probleme apparait avec les silences (quand il y a plusieurs phrases)


class Frame(wx.Frame):
    """
    Frame class. Contains the GUI, with buttons
    
    Attributes :
        trainFile : string
            music file
        speechFile : string
            file to denoise
        model_filename : string
            = "model.npy"
        model_file = string
            outputDir + model_filename
        outputDir : string
            output folder for model file and denoised file
        outputFile : string
            output file!

        Ksic : int = 10 
            number of nmf components for music training file
        Kspeech  : int = 15
            number of nmf components for speech
        Beta : float = 1.
            Beta-divergence, Beta = 1. => Kullback-Leibler divergence
        wlen_millisec : float = 50
            window size in milliseconds
    """

    def __init__(self, parent, id, title):
        """

        Parameters
        ----------
        parent
        id
        title

        set 7 buttons and two labels (StaticText)
        """
        style = wx.MAXIMIZE_BOX | wx.RESIZE_BORDER | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX
        wx.Frame.__init__(self, parent, id, title=title, style=style, size=MAIN_WINDOW_DEFAULT_SIZE)
        self.Center()

        # (longueur, hauteur)
        # self.panel = wx.Panel(self)
        # self.panel.SetBackgroundColour("gray")

        # training

        lbl = wx.StaticText(self, id=wx.ID_ANY, style=wx.ALIGN_CENTER,
                            pos=(100, 30))
        font = wx.Font(18, wx.ROMAN, wx.ITALIC, wx.NORMAL)
        lbl.SetFont(font)
        lbl.SetLabel("Train")

        self.button1 = wx.Button(self, id=wx.ID_ANY, label="file for training",
                                 pos=(100, 60), size=(105, 20))
        self.button1.Bind(wx.EVT_BUTTON, self.onButton1)

        self.button4 = wx.Button(self, id=wx.ID_ANY, label="model and speech output folder",
                                 pos=(160, 90), size=(210, 20))
        self.button4.Bind(wx.EVT_BUTTON, self.onButton4)

        self.buttonRunTrain = wx.Button(self, id=wx.ID_ANY, label="Process train", pos=(100, 120))
        self.buttonRunTrain.Bind(wx.EVT_BUTTON, self.trainMusic)

        # ####
        # ####
        # processing noisy file

        lbl2 = wx.StaticText(self, id=wx.ID_ANY, style=wx.ALIGN_CENTER,
                             pos=(350, 30))
        # font = wx.Font(18, wx.ROMAN, wx.ITALIC, wx.NORMAL)
        lbl2.SetFont(font)
        lbl2.SetLabel("Process noisy file")

        self.button3 = wx.Button(self, id=wx.ID_ANY, label="noisy file",
                                 pos=(350, 60), size=(70, 20))
        self.button3.Bind(wx.EVT_BUTTON, self.onButton3)

        self.buttonRunDenoise = wx.Button(self, id=wx.ID_ANY, label="Process file", pos=(350, 120))
        self.buttonRunDenoise.Bind(wx.EVT_BUTTON, self.speech_separation)

        # ##### CHECK

        self.buttonCheck = wx.Button(self, id=wx.ID_ANY, label="Check", pos=(300, 200))
        self.buttonCheck.Bind(wx.EVT_BUTTON, self.onButtonCheck)

        # ##### EXIT
        self.buttonExit = wx.Button(self, id=wx.ID_ANY, label="Exit", pos=(50, 200))
        self.buttonExit.Bind(wx.EVT_BUTTON, self.onButtonExit)

        # init attributes
        self.trainFile = ""
        self.speechFile = ""
        self.model_filename = "model.npy"
        self.model_file = ""
        self.outputDir = ""
        self.outputFile = ""
        # number of nmf components for each source
        self.Ksic = 10
        self.Kspeech = 15
        self.Beta = 1.  # Kullback-Leibler divergence

        self.wlen_millisec = 50  # in milliseconds

    def OnExit(self, event):
        self.Destroy()

    def onButtonExit(self, event):
        self.OnExit(event)

    def select_file(self, default_file, name='Select file'):
        wildcard = "audio file wav (*.wav)|*.wav"
        selection = wx.FileDialog(self, name, defaultFile=default_file,
                                  wildcard=wildcard, style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        selection.ShowModal()
        filename = selection.GetPath()
        print(selection.GetPath())
        if os.path.isfile(filename) == 1:
            return filename
        else:
            wx.MessageBox('Invalid file', 'Error', wx.OK | wx.ICON_ERROR)
            return ""

    def select_dir(self, default_folder, name='Select folder'):
        selection = wx.DirDialog(self, name, defaultPath=default_folder,
                                 style=wx.DD_DEFAULT_STYLE)
        selection.ShowModal()
        dirname = selection.GetPath()
        print(selection.GetPath())
        if os.path.isdir(dirname) == 1:
            return dirname
        else:
            wx.MessageBox('Invalid folder', 'Error', wx.OK | wx.ICON_ERROR)
            return ""

    def onButton1(self, event):
        print("Train file button pressed!")
        sfile = self.select_file("", "Select train file")
        if sfile != "":
            self.trainFile = sfile

    def onButton3(self, event):
        print("Noisy file button pressed!")
        sfile = self.select_file("", "Select noisy file")
        if sfile != "":
            self.speechFile = sfile

    def onButton4(self, event):
        print("Model and speech output folder button pressed!")
        out_dir = self.select_dir("", "Select model and speech output folder")
        if out_dir != "":
            self.outputDir = out_dir

    def onButtonCheck(self, event):
        print("Check button pressed!")
        print("")
        print("train file")
        print(self.trainFile)
        print("noisy speech file")
        print(self.speechFile)
        print("output dir")
        print(self.outputDir)
        print('number music components %d' % self.Ksic)
        print('number speech components %d' % self.Kspeech)

    def trainMusic(self, event):

        print("Process training")
        # ########@
        # parameters
        # ########@
        nb_iter = 500  # number of nmf iterations
        noise_floor = 1e-10  # small constant in IS nmf
        small_const_add_spec = 1e-10  # add this small constant to input spectrogram
        # to avoid numerical problems if spec == 0

        # read audio
        try:
            y, fs = snd.read(self.trainFile)
        except:
            wx.MessageBox('Failed to read audio file', 'Error', wx.OK | wx.ICON_ERROR)
            return

        print('file : %s' % self.trainFile)
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
        wlen = int(np.floor(fs * self.wlen_millisec / 1000.))
        print('Window length %d ms, %d samples' % (int(np.floor(self.wlen_millisec)), wlen))

        hlen = wlen // 2
        n_fft = 1
        while n_fft < wlen:
            n_fft *= 2
        print('fft size : %d' % n_fft)

        # STFT
        X, f, t = Stft(y, wlen, hlen, n_fft, fs)
        spec = np.abs(X)  # magnitude spectrogram
        spec += small_const_add_spec

        print('running an input matrix of size {}'.format(spec.shape))

        w_music, _, _, _ = BetanmfSparse(spec, W=self.Ksic, Beta=self.Beta, nbIter=nb_iter,
                                         noiseFloor=noise_floor, sparseType='None', LRupdate=False)

        self.model_file = os.path.join(self.outputDir, self.model_filename)
        np.save(self.model_file, w_music)

        wx.MessageBox('Training done!', 'Job''s done!', wx.OK)

        return

    def speech_separation(self, event):

        print("Process denoising")
        # ########@# #
        # parameters #
        # ########@ ##
        nb_iter = 500  # number of nmf iterations
        noise_floor = 1e-10  # small constant in IS nmf
        small_const_add_spec = 1e-10  # add this small constant to input spectrogram
        # to avoid numerical problems if spec == 0

        # 1) load data file
        self.model_file = os.path.join(self.outputDir, self.model_filename)
        try:
            w_music = np.load(self.model_file)
        except:
            wx.MessageBox('Failed to load model file', 'Error', wx.OK | wx.ICON_ERROR)
            return

        # 2) load audio file

        # read audio
        # audio_dir = os.path.dirname(self.trainFile)
        # filename = os.path.basename(self.trainFile)
        try:
            y, fs = snd.read(self.speechFile)
        except:
            wx.MessageBox('Failed to read audio file', 'Error', wx.OK | wx.ICON_ERROR)
            return

        print('file : %s' % self.speechFile)
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
        wlen = int(np.floor(fs * self.wlen_millisec / 1000.))
        print('Window length %d ms, %d samples' % (int(np.floor(self.wlen_millisec)), wlen))

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
        w_speech = np.abs(np.random.randn(F, self.Kspeech)) + 1e-8
        w_all_init = np.hstack((w_speech, w_music))
        w_all = np.copy(w_all_init)
        ind_w = range(self.Kspeech)  # update speech only
        w_all, h_all, v_all, _ = BetanmfSparse(spec, W=w_all, indW=ind_w, Beta=self.Beta,
                                               nbIter=nb_iter, noiseFloor=noise_floor,
                                               sparseType='None', LRupdate=True)

        # TODO : play with sparseType

        # 5) Wiener filtering
        w_speech_out = w_all[:, ind_w]
        h_speech_out = h_all[ind_w, :]
        v_speech_out = w_speech_out.dot(h_speech_out) + noise_floor

        X_speech = X*v_speech_out/v_all

        # 6) inverse STFT
        x_speech, _ = Istft(X_speech, wlen, hlen, n_fft, fs)

        # 7) save audio
        filename = os.path.basename(self.speechFile)
        self.outputFile = os.path.join(self.outputDir, filename[:-4] + 'denoised.wav')
        snd.write(self.outputFile, x_speech, fs)

        wx.MessageBox('Denoising done!', 'Job''s done!', wx.OK)

        return


if __name__ == "__main__":
    app = wx.App()
    c_frame = Frame(None, -1, 'acappella')
    c_frame.Show(True)

    app.MainLoop()
