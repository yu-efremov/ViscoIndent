# -*- coding: utf-8 -*-
"""
@author: Yuri
"""

import sys
import os
import glob
import inspect
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, \
    QVBoxLayout, QHBoxLayout, QLabel,\
    QPushButton, QFileDialog, QListView, QAbstractItemView, QTreeView

import numpy as np

from Pars_class import Pars_gen
from import_AFM_data import Bruker_import
from import_ARDF import ARDF_import
from import_Asylum_ibw import import_multifiles_ibw
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir+ '/projects/microtester')
sys.path.insert(0, parentdir+ '/projects/biomomentum')
from Biomomentum_import_FC import import_multifiles_BM
from import_Microtester import import_Microtester_toVI_multi
from make_Results import make_Results
from utils_ViscoIndent import load_AFM_data_pickle, load_AFM_data_pickle_short


class selection_win1(QMainWindow):


    def __init__(self, parent=None):
        super(selection_win1, self).__init__(parent)
        self.selection_win1_gui = parent  # share variables
        if not hasattr(self.selection_win1_gui, 'commongui'):
            self.initUI()


    def initUI(self):

        self.btn_loadSPM = QPushButton("Load .spm (Bruker)", self)
        self.btn_loadSPM.clicked.connect(self.load_spm)

        self.btn_loadARDF = QPushButton("Load .ARDF (Asylum)", self)
        self.btn_loadARDF.clicked.connect(self.load_ardf)

        self.btn_load_ibw = QPushButton("Load .ibw (Asylum)", self)
        self.btn_load_ibw.clicked.connect(self.load_ibw)

        self.btn_loadBiomomentumFC = QPushButton("Load Biomomentum data", self)
        self.btn_loadBiomomentumFC.clicked.connect(self.load_biomomentum)

        self.btn_loadMicrotesterFC = QPushButton("Load Microtester data", self)
        self.btn_loadMicrotesterFC.clicked.connect(self.load_microtester)

        self.btn_loadDATshort = QPushButton("Load .dat (processed file for spm)", self)
        self.btn_loadDATshort.clicked.connect(self.load_dat_short)

        self.btn_loadDATfull = QPushButton("Load .dat (full processed file)", self)
        self.btn_loadDATfull.clicked.connect(self.load_dat_full)

        self.btn_loadWorkspace = QPushButton("Load from workspace", self)
        self.btn_loadWorkspace.clicked.connect(self.load_workspace)

        self.btn_Exit = QPushButton("Exit", self)
        self.btn_Exit.clicked.connect(self.btn_Exit_pressed)
        self.lookfolder = "examples/"
        try:
            self.lookfolder = self.selection_win1_gui.start_folder
        except:
            pass
        if 'Data' in globals() or hasattr(self.selection_win1_gui, 'Data'):
            print('Data were found in workspace')
            strData = 'Data found in workspace. Reuse or open new?'
            try:
                self.lookfolder = os.path.dirname(self.selection_win1_gui.Pars.filedir[0])
            except:
                pass
            # choice = QMessageBox.question(msg_box, "Choose Data", strData, QMessageBox.Yes | QMessageBox.Open | QMessageBox.Cancel)
        else:
            print('no Data were found in workspace')
            strData = 'Open data file or exit'
            self.btn_loadWorkspace.setEnabled(False)
            # choice = QMessageBox.question(msg_box, "Choose Data", strData, QMessageBox.Open | QMessageBox.Cancel)
        self.label_textabove = QLabel(strData)

        self.setGeometry(300, 300, 900, 100)
        self.setWindowTitle('Select data to open')
        layoutA = QHBoxLayout()
        layoutA.addWidget(self.btn_loadSPM)
        layoutA.addWidget(self.btn_loadARDF)
        layoutA.addWidget(self.btn_load_ibw)
        layoutA.addWidget(self.btn_loadBiomomentumFC)
        layoutB = QHBoxLayout()
        layoutA.addWidget(self.btn_loadMicrotesterFC)
        layoutB.addWidget(self.btn_loadDATshort)
        layoutB.addWidget(self.btn_loadDATfull)
        layoutB.addWidget(self.btn_loadWorkspace)
        layoutC = QHBoxLayout()
        layoutC.addWidget(self.btn_Exit)
        layoutM = QVBoxLayout()
        layoutM.addWidget(self.label_textabove)
        layoutM.addLayout(layoutA)
        layoutM.addLayout(layoutB)
        layoutM.addLayout(layoutC)

        widget = QWidget()
        widget.setLayout(layoutM)
        self.setCentralWidget(widget)
        self.show()
        self.activateWindow()


    def load_spm(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Select Bruker file", self.lookfolder, "Bruker files (*.spm)", options=options)
        if self.fileName != "":
            filedir0 = self.fileName
            self.supress_ROIquestion = 0

            fnameshort = filedir0.split("/")
            fnameshort = fnameshort[-1]
            Pars = Pars_gen()
            Pars.probe_shape = 'sphere'
            Pars.probe_dimension = 5000
            Pars.Poisson = 0.5         # Poisson's ratio of the sample
            Pars.dT = 1e-3                 # Sampling time
            Pars.height = 0
            Pars.HeightfromZ = 0
            Pars.viscomodel = 'elastic'
            Pars.hydro.corr = 1
            Pars.hydro.corr_type = 2
            Pars.hydro.speedcoef = 5.0e-7
            Pars.cp_adjust = 1

            Pars.filedir = []
            Pars.filedir.append(filedir0)
            Pars.fnameshort = []
            Pars.fnameshort.append(fnameshort)
            self.Data = {}
            Bruker_data = Bruker_import(Pars)
            print('data loaded from file')
            self.Data = Bruker_data.Data
            self.Pars = Bruker_data.Pars
            self.Results = make_Results(np.shape(self.Data)[0])
            print(self.Data[0])
            self.Results.Pixel = self.Data[:, 0]  # remove nans
            if hasattr(self.selection_win1_gui, 'commongui'):
                # self.selection_win1_gui.supress_ROIquestion = 1
                if hasattr(self.selection_win1_gui, 'config'):
                    delattr(self.selection_win1_gui, 'config')
                self.selection_win1_gui.Pars = self.Pars
                self.selection_win1_gui.Data = self.Data
                self.selection_win1_gui.Results = self.Results
                self.close()
                self.selection_win1_gui.initUI2()


    def load_ardf(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Select Asylum ARDF file", self.lookfolder, "AR Files (*.ardf)", options=options)
        if self.fileName != "":
            filedir0 = self.fileName
            self.supress_ROIquestion = 0

            fnameshort = filedir0.split("/")
            fnameshort = fnameshort[-1]
            Pars = Pars_gen()
            Pars.probe_shape = 'sphere'
            Pars.probe_dimension = 5000
            Pars.Poisson = 0.5         # Poisson's ratio of the sample
            Pars.dT = 1e-3             # Sampling time
            Pars.height = 0
            Pars.HeightfromZ = 0
            Pars.viscomodel = 'elastic'
            Pars.hydro.corr = 1
            Pars.hydro.corr_type = 2
            Pars.hydro.speedcoef = 4.0e-7
            Pars.cp_adjust = 1

            Pars.filedir = []
            Pars.filedir.append(filedir0)
            Pars.fnameshort = []
            Pars.fnameshort.append(fnameshort)
            self.Data = {}
            [Pars, Data] = ARDF_import(Pars)
            print('data loaded from ARDF file')
            self.Data = Data
            self.Pars = Pars
            self.Results = make_Results(np.shape(self.Data)[0])
            print(self.Data[0])
            self.Results.Pixel = self.Data[:, 0]  # remove nans
            if hasattr(self.selection_win1_gui, 'commongui'):
                self.selection_win1_gui.Pars = self.Pars
                self.selection_win1_gui.Data = self.Data
                self.selection_win1_gui.Results = self.Results
                self.close()
                self.selection_win1_gui.initUI2()

    def load_ibw(self):
        options = QFileDialog.Options()
        self.fileNames, _ = QFileDialog.getOpenFileNames(self, "Select Asylum .ibw files", self.lookfolder, "All Files (*.ibw);", options=options)
        # filenames = []  # for several folder selction
        # ij = 2
        # while self.fileNames:
        #     # print(w.fileNames)
        #     # w.close()
        #     filenamesH = self.fileNames
        #     for filename in filenamesH:
        #         filenames.append(filename)
        #         window_title2 = "Select Biomomentum file" + ' from folder_' + str(ij) + ' or cancel'
        #     ij = ij + 1
        #     self.fileNames, _ = QFileDialog.getOpenFileNames(self, window_title2, self.lookfolder, "All Files (*.txt);;Python Files (*.py)", options=options)
        # self.fileNames = filenames
        # print(self.fileNames)
        # if self.fileNames != "":
        if len(self.fileNames)>0:
            Pars, Data, Results = import_multifiles_ibw(self.fileNames)
            Pars.topo = np.ones([1, np.shape(Data)[0]])
            print('Asylum .ibw data loaded')
            self.Data = Data
            self.Pars = Pars
            self.Results = Results
            self.supress_ROIquestion = 1
            # self.Results = make_Results(np.shape(self.Data)[0])
            # print(self.Data[0])
            self.Results.Pixel = self.Data[:, 0]  # remove nans
            if hasattr(self.selection_win1_gui, 'commongui'):
                self.selection_win1_gui.Pars = self.Pars
                self.selection_win1_gui.Data = self.Data
                self.selection_win1_gui.Results = self.Results
                self.selection_win1_gui.supress_ROIquestion = self.supress_ROIquestion
                from VI_config import biomomentum as config  # biomomentum config for ibw
                self.selection_win1_gui.config = config
                self.close()
                self.selection_win1_gui.initUI2()

    def load_biomomentum(self):
        options = QFileDialog.Options()
        self.fileNames, _ = QFileDialog.getOpenFileNames(self, "Select Biomomentum file", self.lookfolder, "All Files (*.ibw);;Python Files (*.py)", options=options)
        filenames = []
        ij = 2
        while self.fileNames:
            # print(w.fileNames)
            # w.close()
            filenamesH = self.fileNames
            for filename in filenamesH:
                filenames.append(filename)
                window_title2 = "Select Biomomentum file" + ' from folder_' + str(ij) + ' or cancel'
            ij = ij + 1
            self.fileNames, _ = QFileDialog.getOpenFileNames(self, window_title2, self.lookfolder, "All Files (*.txt);;Python Files (*.py)", options=options)
        self.fileNames = filenames
        # print(self.fileNames)
        # if self.fileNames != "":
        if len(self.fileNames)>0:
            num_regs = 0
            Pars, Data, Results = import_multifiles_BM(self.fileNames, num_regs)
            Pars.topo = np.ones([1, np.shape(Data)[0]])
            print('Biomomenum data loaded')
            self.Data = Data
            self.Pars = Pars
            self.Results = Results
            self.supress_ROIquestion = 1
            # self.Results = make_Results(np.shape(self.Data)[0])
            print(self.Data[0])
            self.Results.Pixel = self.Data[:, 0]  # remove nans
            if hasattr(self.selection_win1_gui, 'commongui'):
                self.selection_win1_gui.Pars = self.Pars
                self.selection_win1_gui.Data = self.Data
                self.selection_win1_gui.Results = self.Results
                self.selection_win1_gui.supress_ROIquestion = self.supress_ROIquestion
                from VI_config import biomomentum as config  # biomomentum cells config
                self.selection_win1_gui.config = config
                self.close()
                self.selection_win1_gui.initUI2()


    def load_microtester(self):
        start_folder = 'D:/MailCloud/Microsquisher'
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_dialog.setDirectory(start_folder)
        file_view = file_dialog.findChild(QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            pathnames = file_dialog.selectedFiles()

        if 'pathnames' in locals():
            fileNames = []
            for pathnamec in pathnames:
                tfiles = glob.glob(pathnamec+"/"+"*.csv")
                if len(tfiles)>0:
                    fileNames.append(tfiles[0])
    
            Pars, Data, Results = import_Microtester_toVI_multi(fileNames)
            print('Microtester data loaded')
            self.Data = Data
            self.Pars = Pars
            self.Results = Results
            self.Results.Pixel = self.Data[:, 0]  # remove nans
            self.supress_ROIquestion = 1
            if hasattr(self.selection_win1_gui, 'commongui'):
                self.selection_win1_gui.Pars = self.Pars
                self.selection_win1_gui.Data = self.Data
                self.selection_win1_gui.Results = self.Results
                self.selection_win1_gui.supress_ROIquestion = self.supress_ROIquestion
                self.close()
                self.selection_win1_gui.initUI2()


    def load_dat_short(self):
        print('load dat short')
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Selecd .dat file (for .spm file)", self.lookfolder, "Data Files (*.dat);;Python Files (*.py)", options=options)

        if self.fileName != "":
            # self.Pars, self.Data, self.Results = load_AFM_data_pickle(self.fileName)
            self.Pars, self.Data, self.Results = load_AFM_data_pickle_short(self.fileName)
            if hasattr(self.selection_win1_gui, 'commongui'):
                self.selection_win1_gui.loaded = 1
                self.selection_win1_gui.Pars = self.Pars
                self.selection_win1_gui.Data = self.Data
                self.selection_win1_gui.Results = self.Results
                self.selection_win1_gui.supress_ROIquestion = 1  # rewrited by config!!!
                self.close()
                self.selection_win1_gui.initUI2()

    def load_dat_full(self):
        print('load dat full')
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Selecd .dat file", self.lookfolder, "Data Files (*.dat);;Python Files (*.py)", options=options)
    
        if self.fileName != "":
            self.Pars, self.Data, self.Results = load_AFM_data_pickle(self.fileName)
            if hasattr(self.selection_win1_gui, 'commongui'):
                self.selection_win1_gui.loaded = 1
                self.selection_win1_gui.Pars = self.Pars
                self.selection_win1_gui.Data = self.Data
                self.selection_win1_gui.Results = self.Results
                self.selection_win1_gui.supress_ROIquestion = 1  # rewrited by config!!!
                self.close()
                self.selection_win1_gui.initUI2()

    def load_workspace(self):
        self.supress_ROIquestion = 1
        if 'Data' in globals():  # and choice == QMessageBox.Yes:
            print('here1')
            global Pars, Results, Data
            if 'Results' not in globals():
                self.Results = make_Results(np.shape(Data)[0])
            else:
                self.Results = Results
            self.Data = Data
            self.Pars = Pars
            self.close()
        elif hasattr(self.selection_win1_gui, 'Data'):
            self.selection_win1_gui.loaded = 1
            self.Pars = self.selection_win1_gui.Pars
            self.Data = self.selection_win1_gui.Data
            self.Results = self.selection_win1_gui.Results
            self.selection_win1_gui.supress_ROIquestion = 1
            self.close()
            self.selection_win1_gui.initUI2()

    def btn_Exit_pressed(self):
        if hasattr(self.selection_win1_gui, 'commongui'):
            self.selection_win1_gui.close()
            self.close()
        else:
            self.close()

    def closeEvent(self, event):

        if not hasattr(self.selection_win1_gui, 'commongui'):
            # print('here')
            if hasattr(self, 'Pars'):
                # print('here2')
                global Pars, Data, Results
                Pars = self.Pars
                Data = self.Data
                Results = self.Results
            print('module exit')
            QApplication.quit()
        else:
            if hasattr(self, 'Pars'):
                self.selection_win1_gui.Pars = self.Pars
                self.selection_win1_gui.Data = self.Data
                self.selection_win1_gui.Results = self.Results
            else:
                self.selection_win1_gui.supress_ROIquestion = 1
            # self.selection_win1_gui.curveviewer()
            # print('module_PLpart_finished')
            self.close()
            # if 'SPYDER_ENCODING' in os.environ.keys():
            #     self.selection_win1_gui.close()  # initUI2
            # else:
            #     self.close()
            print('module_DataLoad_finished')


if __name__ == '__main__':
    try:
        del app
    except:
        print('noapp')

    app = QApplication(sys.argv)
    main = selection_win1()
    main.show()
    sys.exit(app.exec_())
