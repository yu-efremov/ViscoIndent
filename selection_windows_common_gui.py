# -*- coding: utf-8 -*-
"""
@author: Yuri
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, \
    QVBoxLayout, QHBoxLayout, QLabel,\
    QPushButton, QFileDialog

import numpy as np

from Pars_class import Pars_gen
from import_AFM_data import Bruker_import
from make_Results import make_Results
from utils_ViscoIndent import load_AFM_data_pickle, load_AFM_data_pickle_short


class selection_win1(QMainWindow):

    def __init__(self, parent=None):
        super(selection_win1, self).__init__(parent)
        self.selection_win1_gui = parent  # share variables
        if not hasattr(self.selection_win1_gui, 'commongui'):
            self.initUI()

    def initUI(self):

        self.btn_loadSPM = QPushButton("Load .spm", self)
        self.btn_loadSPM.clicked.connect(self.load_spm)

        self.btn_loadDAT = QPushButton("Load .dat", self)
        self.btn_loadDAT.clicked.connect(self.load_dat)

        self.btn_loadWorkspace = QPushButton("Load from workspace", self)
        self.btn_loadWorkspace.clicked.connect(self.load_workspace)

        self.btn_Exit = QPushButton("Exit", self)
        self.btn_Exit.clicked.connect(self.btn_Exit_pressed)

        if 'Data' in globals() or hasattr(self.selection_win1_gui, 'Data'):
            print('Data were found in workspace')
            strData = 'Data found in workspace. Reuse or open new?'
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
        layoutA.addWidget(self.btn_loadDAT)
        layoutA.addWidget(self.btn_loadWorkspace)
        layoutA.addWidget(self.btn_Exit)
        layoutM = QVBoxLayout()
        layoutM.addWidget(self.label_textabove)
        layoutM.addLayout(layoutA)

        widget = QWidget()
        widget.setLayout(layoutM)
        self.setCentralWidget(widget)
        self.show()
        self.activateWindow()

    def load_spm(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "examples/", "All Files (*);;Python Files (*.py)", options=options)
        filedir0 = self.fileName
        self.supress_ROIquestion = 0

        # filedir0 = 'D:/MailCloud/AFM_data/BrukerResolve/MISIS/20190928_PC3_MMAE_ACALold2/area1-01.0_00000.spm'
        # filedir0 = 'examples/Bruker_forcevolume_cells.spm'
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
        Pars.hydro.speedcoef = 4.0e-7
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
            self.selection_win1_gui.Pars = self.Pars
            self.selection_win1_gui.Data = self.Data
            self.selection_win1_gui.Results = self.Results
            self.close()
            self.selection_win1_gui.initUI2()

    def load_dat(self):
        print('load dat')
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "examples/", "All Files (*.dat);;Python Files (*.py)", options=options)

        # self.Pars, self.Data, self.Results = load_AFM_data_pickle(self.fileName)

        self.Pars, self.Data, self.Results = load_AFM_data_pickle_short(self.fileName)
        # self.supress_ROIquestion = 1
        if hasattr(self.selection_win1_gui, 'commongui'):
            self.selection_win1_gui.Pars = self.Pars
            self.selection_win1_gui.Data = self.Data
            self.selection_win1_gui.Results = self.Results
            self.selection_win1_gui.supress_ROIquestion = 1
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
            self.Pars = self.selection_win1_gui.Pars
            self.Data = self.selection_win1_gui.Data
            self.Results = self.selection_win1_gui.Results
            self.selection_win1_gui.supress_ROIquestion = 1
            self.close()
            self.selection_win1_gui.initUI2()

    def btn_Exit_pressed(self):
        self.close()

    def closeEvent(self, event):

        if not hasattr(self.selection_win1_gui, 'commongui'):
            if hasattr(self, 'Pars'):
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
            # self.close()
            # self.selection_win1_gui.close()  # initUI2
            print('module_PLpart_finished')


if __name__ == '__main__':
    try:
        del app
    except:
        print('noapp')

    app = QApplication(sys.argv)
    main = selection_win1()
    main.show()
    sys.exit(app.exec_())
