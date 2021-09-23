# -*- coding: utf-8 -*-
"""
gui - common gui for data import and processing (force volume)
with PyQt5
@author: Yuri Efremov, yu.efremov@gmail.com
"""


import sys
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, \
    QVBoxLayout, QHBoxLayout, QLineEdit, QTableView, \
    QPushButton, QFileDialog, QMessageBox, QComboBox, QSlider, QSizePolicy
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# from Pars_class import Pars_gen
# from import_AFM_data import Bruker_import
from make_Results import make_Results
from tingprocessing_class4 import tingsprocessingd1
from flattenAFMwrap_commongui import flattenAFMwrap
from select_regions_on_image_commongui import selectROIs
from maps_view1_commongui import maps_view
from set_Pars_commongui import set_Pars
from utils_ViscoIndent import save_AFM_data_pickle, \
    save_AFM_data_pickle_short, curve_from_saved_pars
from selection_windows_common_gui import selection_win1
from config import config as config


def dicttolist(Dict):
    temp = []
    dictlist = []
    for key, value in Dict.items():
        temp = [key, str(value)]
        dictlist.append(temp)
    return dictlist


class App(QMainWindow):

    def read_data(self):
        model = self.table.model()
        Pars2 = []
        for row in range(model.rowCount(0)):
            Pars2.append([])
            for column in range(model.columnCount(0)):
                index = model.index(row, column)
                try:
                    Pars2[row].append(float(model.data(index, 0)))
                except:
                    Pars2[row].append((model.data(index, 0)))
        first_curve = int(self.first_curve.text())
        last_curve = int(self.last_curve.text())
        self.Pars.curve_range = np.array([first_curve, last_curve])
        # print(Pars.curve_range)
        # print(Pars)

    def button1(self):
        first_curve = int(self.first_curve.text())
        last_curve = int(self.last_curve.text())
        self.Pars.curve_range = np.array([first_curve, last_curve])
        if self.Data.shape[1] < 3:
            self.Data = np.insert(self.Data, 2, np.zeros((self.Data.shape[0])), axis=1)
        for kk in range(self.Pars.curve_range[0], self.Pars.curve_range[1]+1):
            Results_temp, self.Data[kk][0:2], DFL_corrs = tingsprocessingd1(self.Pars, self.Data[kk][0:2])
            self.Data[kk][2] = DFL_corrs
            self.Results.loc[kk, :] = Results_temp.loc[0, :]

    def button_save_click(self):
        self.Pars.save_short = 1
        print('save start')
        if self.Pars.save_short == 0:
            filename = QFileDialog.getSaveFileName(self, 'Save File')[0]
            if os.path.isfile(filename):
                self.Rewrite_q = QMessageBox.question(QMessageBox(), "saved file was found", 'saved file was found. rewrite?', QMessageBox.Yes | QMessageBox.No)
                if self.Rewrite_q == QMessageBox.Yes:
                    os.remove(filename)  # delete the file
            else:
                self.Rewrite_q = QMessageBox.Yes
            if self.Rewrite_q == QMessageBox.Yes:
                print('saving ' + filename)
                for ii in range(self.Data.shape[0]):
                    self.Data[ii][1] = self.Data[ii][1].astype('float32')  # reduces size 2x, fast enough
                save_AFM_data_pickle(filename, self.Pars, self.Data, self.Results)
            else:
                for ii in range(self.Data.shape[0]):
                    self.Data[ii][1] = self.Data[ii][1].astype('float32')  # reduces size 2x, fast enough
                save_AFM_data_pickle(filename, self.Pars, self.Data, self.Results)
        elif self.Pars.save_short == 1:
            filename = self.Pars.filedir[0][:-3] + 'dat'
            if os.path.isfile(filename):
                self.Rewrite_q = QMessageBox.question(QMessageBox(), "saved file was found", 'saved file was found. rewrite?', QMessageBox.Yes | QMessageBox.No)
                if self.Rewrite_q == QMessageBox.Yes:
                    os.remove(filename)  # delete the file
            else:
                self.Rewrite_q = QMessageBox.Yes
            if self.Rewrite_q == QMessageBox.Yes:
                print('saving ' + filename)
                for ii in range(self.Data.shape[0]):
                    self.Data[ii][1] = self.Data[ii][1].astype('float32')  # reduces size 2x, fast enough
                save_AFM_data_pickle_short(filename, self.Pars, self.Data, self.Results)
                print('data saved')

    def button_save_excel_click(self):
        filename = QFileDialog.getSaveFileName(self, 'Save File')[0]
        self.Results.to_excel(filename)

    def button_load_click(self):
        self.selection_win1.initUI()

    def button_maps_click(self):
        self.MAPviewdialog.initUI()

    def button_changePars_click(self):
        self.changeParsdialog.initUI()

    def changefigaxis(self):
        self.Pars.graph = str(self.graphT.currentText())
        self.curvedata = PlotCanvas.plot(self.m, self.Pars, self.Data, self.Results)
        self.show()

    def changeValue(self, value):
        # print(value)
        self.Pars.kk = value
        self.curvedata = PlotCanvas.plot(self.m, self.Pars, self.Data, self.Results)
        listResults = dicttolist(self.Results.iloc[value].to_dict())
        self.model_Res = TableModel(listResults)
        self.table_Res.setModel(self.model_Res)

    def sldDisconnect(self):
        self.slider.valueChanged.disconnect()

    def sldReconnect(self):
        self.slider.valueChanged.connect(self.changeValue)
        self.slider.valueChanged.emit(self.slider.value())

    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.left = 50
        self.top = 50
        self.title = 'GUI for force curve processing with ViscoIndent'
        self.width = 1140
        self.height = 750
        self.commongui = 1
        self.MAPdialog = flattenAFMwrap(self)
        self.ROIdialog = selectROIs(self)
        self.MAPviewdialog = maps_view(self)
        self.changeParsdialog = set_Pars(self)
        self.selection_win1 = selection_win1(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.supress_ROIquestion = 0
        if 'Data' in globals():
            print('Data were found in workspace')
            global Pars, Results, Data
            if 'Results' not in globals():
                self.Results = make_Results(np.shape(Data)[0])
            else:
                self.Results = Results
            self.Data = Data
            self.Pars = Pars
        self.selection_win1.initUI()

    def initUI2(self):
        if self.supress_ROIquestion == 0:
            self.MAPdialog.initUI()
        else:
            self.curveviewer()

    def regionselection(self):
        self.ROIproc = QMessageBox.question(QMessageBox(), "select ROIs?", 'select ROIs?', QMessageBox.Yes | QMessageBox.No)
        if self.ROIproc == QMessageBox.Yes:
            self.ROIdialog.initUI()
        else:
            self.curveviewer()

    def curveviewer(self):
        if hasattr(self.Pars.ROIPars, 'reg_nums') and self.Data.shape[0] == len(self.Pars.ROIPars.reg_nums_all):
            indcells = self.Pars.ROIPars.reg_nums_all > 0
            Datasel = self.Data[indcells, :]
            self.Data = Datasel
            self.Results = self.Results.drop(self.Results.index[np.size(self.Data, 0):])  # size correction
            self.Results.Pixel = self.Data[:, 0]  # remove nans

        listPars = self.Pars.class2list()  # dicttolist(Pars)
        listResults = dicttolist(self.Results.iloc[0].to_dict())

        self.table = QTableView()
        self.model = TableModel(listPars)
        self.table.setModel(self.model)
        self.table_Res = QTableView()
        self.model_Res = TableModel(listResults)
        self.table_Res.setModel(self.model_Res)

        self.graphT = QComboBox()
        self.graphT.addItems(['raw', 'elastic', 'viscoelastic'])
        self.graphT.setCurrentIndex(2)
        self.graphT.currentIndexChanged.connect(self.changefigaxis)

        self.Pars.graph = str(self.graphT.currentText())  # 'raw'

        self.first_curve = QLineEdit(self)
        self.first_curve.setText('0')
        self.last_curve = QLineEdit(self)
        self.last_curve.setText(str(len(self.Data)-1))

        self.Pars.kk = 0
        self.read_data()
        self.m = PlotCanvas(self, self.Pars, self.Data, width=5, height=4)
        self.curvedata = PlotCanvas.plot(self.m, self.Pars, self.Data, self.Results)

        button_launch = QPushButton('Start processing', self)
        button_launch.setToolTip('Start the processing for selected data and model')
        button_launch.clicked.connect(self.button1)

        button_maps = QPushButton('Show maps', self)
        button_maps.setToolTip('Show data maps')
        button_maps.clicked.connect(self.button_maps_click)

        button_save = QPushButton('Save to .dat', self)
        button_save.setToolTip('Save .dat')
        button_save.clicked.connect(self.button_save_click)

        button_save_excel = QPushButton('Save to excel', self)
        button_save_excel.setToolTip('Save .xlsx')
        button_save_excel.clicked.connect(self.button_save_excel_click)

        button_load = QPushButton('Load', self)
        button_load.setToolTip('Load new data')
        button_load.clicked.connect(self.button_load_click)

        button_changePars = QPushButton('Modify parameters', self)
        button_changePars.setToolTip('Check and modify parameters for the processing')
        button_changePars.clicked.connect(self.button_changePars_click)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.Data)-1)
        # self.slider.valueChanged[int].connect(self.changeValue)
        self.slider.valueChanged.connect(self.changeValue)  # 3xcombo for slider
        self.slider.sliderPressed.connect(self.sldDisconnect)
        self.slider.sliderReleased.connect(self.sldReconnect)

        layoutM = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()
        layoutR = QVBoxLayout()

        layout2.addWidget(self.table)
        layout_curve_numbers = QHBoxLayout()
        layout_curve_numbers.addWidget(self.first_curve)
        layout_curve_numbers.addWidget(self.last_curve)
        layout2.addLayout(layout_curve_numbers)
        layout2.addWidget(button_changePars)
        layout2.addWidget(button_launch)

        layoutM.addLayout(layout2)

        layout3.addWidget(self.graphT)
        layout3.addWidget(self.m)
        layout3.addWidget(self.slider)

        layoutR.addWidget(self.table_Res)
        layoutR.addWidget(button_maps)
        layoutR.addWidget(button_load)
        layoutR.addWidget(button_save)
        layoutR.addWidget(button_save_excel)

        layoutM.addLayout(layout3)
        layoutM.addLayout(layoutR)

        # three commands to apply layout
        widget = QWidget()
        widget.setLayout(layoutM)
        self.setCentralWidget(widget)

        self.show()
        self.activateWindow()

    def closeEvent(self, event):
        print("The program was shut down.")
        if hasattr(self, 'Pars'):
            global Pars, Data, Results
            Pars = self.Pars
            Data = self.Data
            Results = self.Results
        QApplication.quit()


class PlotCanvas(FigureCanvas):

    def __init__(self, QWidget, Pars, Data, parent=None, width=5, height=4, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.plot(Pars)

    def plot(self, Pars, Data, Results):

        kk = Pars.kk
        cResults = Results.loc[kk, :]

        if hasattr(Pars, 'save_short') and Pars.save_short == 1 and np.shape(Data)[1] >= 3 and type(Data[kk][2]) is list:
            currentcurve3 = curve_from_saved_pars(Pars, Data[kk], Results.loc[kk, :])
        else:
            currentcurve3 = Data[kk][1]
        rawZ = currentcurve3[:, 0]
        rawDFL = currentcurve3[:, 1]
        if np.shape(currentcurve3)[1] > 4:
            ind_Hertz = currentcurve3[:, 2]
            force = currentcurve3[:, 3]
            fit_Hertz = currentcurve3[:, 4]
            ind_visco = ind_Hertz
            fit_visco = np.nan*ind_visco
        if np.shape(currentcurve3)[1] > 5 and ~np.isnan(cResults['cpTing']):
            ind_visco = ind_Hertz - (cResults['cpTing']-cResults['cpHertz'])
            fit_visco = currentcurve3[:, 5]

        ax = self.axes
        ax.clear()
        if Pars.graph == 'raw':
            ax.plot(rawZ, rawDFL)
        elif Pars.graph == 'elastic' and np.shape(currentcurve3)[1] > 4:
            ax.plot(ind_Hertz, force)
            ax.plot(ind_Hertz, fit_Hertz)
        elif Pars.graph == 'viscoelastic' and np.shape(currentcurve3)[1] > 4:
            ax.plot(ind_visco, force)
            ax.plot(ind_visco, fit_Hertz)
            if np.shape(currentcurve3)[1] > 5:
                ax.plot(ind_visco, fit_visco)
        else:
            ax.plot(currentcurve3[:, 0], currentcurve3[:, 1])
            # plt.plot will open new window
        # if Pars.graph == 'Force versus Indentation':
        #     ax.plot(ind, force, 'r-')
        #     ax.set_title('Force vs indentation')
        # elif Pars.graph == 'Force versus Time':
        #     ax.plot(time, force, 'r-')
        #     ax.set_title('Force vs Time')
        self.draw()


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # Get the raw value
            value = self._data[index.row()][index.column()]

            if isinstance(value, str):
                # Render strings with quotes
                return '%s' % value

            return value

        if role == Qt.EditRole or role == Qt.DisplayRole:
            return QtCore.QVariant(self._data[index.row()][index.column()])

        return QtCore.QVariant()

    def setData(self, index, value, role):
        self._data[index.row()][index.column()] = value  # or float(value)
        return True

    def flags(self, index):
        if index.column() == 1:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if index.column() == 0:  # make first column read only
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._data[0])


if __name__ == '__main__':
    try:
        del app
    except:
        print('noapp')
    app = QApplication(sys.argv)
    ex = App()
    app.exec()