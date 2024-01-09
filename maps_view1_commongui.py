# -*- coding: utf-8 -*-
"""
maps viewer
to use with common pyQt5 gui
"""


import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QCheckBox,\
                            QFileDialog, QMessageBox, QLabel, QComboBox,\
                            QRadioButton, QButtonGroup,\
                            QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt

from matplotlib import cm
from matplotlib.colors import ListedColormap # LinearSegmentedColormap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np

from utils_ViscoIndent import load_AFM_data_pickle_short


class maps_view(QMainWindow):

    def __init__(self, parent=None):
        super(maps_view, self).__init__(parent)
        self.MAPviewdialog = parent  # share variables
        if not hasattr(self.MAPviewdialog, 'commongui'):
            self.initUI()

    def initUI(self):
        if not hasattr(self.MAPviewdialog, 'commongui'):
            if 'Pars' in globals():
                print('Data found in workspace')
                strData = 'Data found in workspace. Reuse or open new?'
                choice = QMessageBox.question(QMessageBox(), "Choose Data", strData, QMessageBox.Yes | QMessageBox.Open)
                if choice == QMessageBox.Yes:
                    global Pars, Results, Data
                    self.Pars = Pars
                    self.Results = Results
                    self.Data = Data
                else:
                    options = QFileDialog.Options()
                    self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)", options=options)
                    self.Pars, self.Data, self.Results = load_AFM_data_pickle_short(self.fileName)
            else:
                options = QFileDialog.Options()
                self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)", options=options)
                # self.Pars, self.Data, self.Results = open_vars(self.fileName[:-4])
                self.Pars, self.Data, self.Results = load_AFM_data_pickle_short(self.fileName)

            # self.Results = self.Results[:np.size(self.Data, 0)]  # size correction
            # self.Results = self.Results.drop(self.Results.index[np.size(self.Data, 0):])  # size correction
            # self.Results.Pixel = self.Data[:, 0]  # remove nans
        else:
            self.Pars = self.MAPviewdialog.Pars
            self.Data = self.MAPviewdialog.Data
            self.Results = self.MAPviewdialog.Results
            # self.Results = self.Results[:np.size(self.Data,0)]  # size correction
            # self.Results.Pixel = self.Data[:, 0]  # remove nans

        self.cmaplist = plt.colormaps()
        top = cm.get_cmap('Oranges', 64)
        bottom = cm.get_cmap('viridis', 512)
        # top = plt.colormaps['Oranges']
        # bottom = plt.colormaps['viridis']
        newcolors = np.vstack((top(np.linspace(0, 1, 64)),
                               bottom(np.linspace(0, 1, 512))))
        self.newcmp = ListedColormap(newcolors, name='viridis_topo')
        self.cmaplist.append('viridis_topo')

        # self.pixN = np.asarray(self.Data[:, 0], int) # take from results somehow...
        self.pixN = np.asarray(self.Results.Pixel, int)

        # self.Res_fields = list(self.Results.columns)
        self.Res_fields = list(self.Results.select_dtypes(include=['float64', 'float32']).columns)

        self.fig = fig = Figure()
        self.canvas1 = FigureCanvasQTAgg(fig)
        self.ax1 = fig.add_subplot(111, title="Map")
        img1 = self.ax1.imshow(self.Pars.topo, interpolation='nearest', cmap='viridis', origin='lower')
        self.cb = self.fig.colorbar(img1)

        self.boxROI = QCheckBox("Show ROIs", self)
        self.boxROI.stateChanged.connect(self.clickBoxROI)

        self.labelLog = QLabel('Log or linear scale:')
        self.btngroupLogopts = [QRadioButton("linear"),
                                QRadioButton("log")]
        self.btngroupLog = QButtonGroup()

        self.cbRes = QComboBox()
        self.cbRes.addItems(self.Res_fields)
        self.cbRes.addItems(['topography', 'corrected topography'])
        self.cbRes.currentIndexChanged.connect(self.cbRes_changed)
        self.cbRes.setCurrentIndex(1)

        self.layoutLog = QVBoxLayout()
        self.layoutLog.addWidget(self.labelLog)
        for i in range(len(self.btngroupLogopts)):
            self.layoutLog.addWidget(self.btngroupLogopts[i])
            self.btngroupLog.addButton(self.btngroupLogopts[i], i)
            self.btngroupLogopts[i].clicked.connect(self.cbRes_changed)
        self.btngroupLogopts[0].setChecked(True)

        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('Maps')

        layoutA = QVBoxLayout()
        layoutA.addWidget(self.canvas1)

        layoutB = QVBoxLayout()
        layoutB.addWidget(self.cbRes)
        layoutB.addWidget(self.boxROI)
        layoutB.addLayout(self.layoutLog)
        layoutB.setAlignment(Qt.AlignTop)  # alignment of widgets

        layoutM = QHBoxLayout()
        layoutM.addLayout(layoutA)
        layoutM.addLayout(layoutB)

        widget = QWidget()
        widget.setLayout(layoutM)
        self.setCentralWidget(widget)
        self.show()
        self.activateWindow()
        self.select_pts()

    def cbRes_changed(self):

        # self.ax1.clear()
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(111, title="Map")
        if self.cbRes.currentText() == 'topography':
            self.currentmap = self.Pars.topo
            cmapS = self.newcmp
        elif self.cbRes.currentText() == 'corrected topography':
            self.currentmap = self.Pars.topo2
            cmapS = self.newcmp
        else:
            self.res_to_map(self.cbRes.currentText())
            cmapS = 'viridis'
        # self.currentmap = self.Pars.topo2
        if self.btngroupLog.checkedId() == 1:
            img1 = self.ax1.imshow(np.log(self.currentmap), interpolation='nearest', cmap=cmapS, origin='lower')  # 'jet'
        else:
            img1 = self.ax1.imshow(self.currentmap, interpolation='nearest', cmap=cmapS, origin='lower')  # 'jet'
        # self.cb.remove()
        # del self.cb
        self.cb = self.fig.colorbar(img1)
        self.canvas1.draw()
        if self.boxROI.isChecked():
            self.showROI()

    def res_to_map(self, keyV):
        fullResults = np.full([self.Pars.topo.size], np.nan)
        # pixN = Results.Pixel[:np.size(Data,0)].values
        # pixN = np.asarray(Results.Pixel[1:np.size(Data,0)], int)

        # fullResults[pixN] = Results.EHertz.values[:np.size(Data,0)]
        fullResults[self.pixN] = self.Results[keyV].values
        self.currentmap = np.reshape(fullResults, self.Pars.topo.shape)

    def clickBoxROI(self, state):

        if state == QtCore.Qt.Checked:
            # self.showROI()
            self.cbRes_changed()
        else:
            self.cbRes_changed()
            pass

    def showROI(self):
        num_of_ROIs = np.max(self.Pars.ROIPars.reg_nums_all)  # self.Pars.ROIPars.num_of_ROIs
        # regions = self.Pars.ROIPars.reg_nums
        region_numbers = self.Pars.ROIPars.reg_nums_all
        # map_regions = np.reshape(region_numbers, self.Pars.topo.shape)
        for ii in range(1, num_of_ROIs+1):
            xc = np.arange(self.Pars.topo.shape[0])
            yc = np.arange(self.Pars.topo.shape[1])
            region = np.copy(region_numbers)
            region = np.where(region != ii, 0, region)
            self.ax1.contour(xc, yc, np.reshape(region, self.Pars.topo.shape))
            map_regions = np.reshape(np.copy(region_numbers), self.Pars.topo.shape)
            map_regions[map_regions != ii] = 0
            labelXcoorF = np.maximum.reduce(map_regions, 0)
            labelXcoor = np.mean(xc[(labelXcoorF == ii)])
            labelYcoorF = np.maximum.reduce(map_regions, 1)
            labelYcoor = np.mean(yc[(labelYcoorF == ii)])
            self.ax1.text(labelXcoor, labelYcoor, str(ii),
                          horizontalalignment='center',
                          verticalalignment='center', fontsize=20)
            self.canvas1.draw()

    def select_pts(self):
        self._cid = self.fig.canvas.mpl_connect("axes_enter_event", self._start_ginput)

    def _start_ginput(self, event):
        self.fig.canvas.mpl_disconnect(self._cid)
        # Workaround for now; patch makes this unnecessary.
        self.fig.show = lambda: None
        for ii in range(1, 100):
            self.pts = self.fig.ginput(1, timeout=0)
            pts = np.asarray(self.pts).squeeze()
            pts = pts.round()  # round pts coordinates
            pts = pts.astype(int)  # coordinates to int
            print(pts)
            if len(pts) > 0:
                PixN = pts[0] + (pts[1])*self.Pars.topo.shape[0]
                print(PixN)
            else:
                break
            try:
                numinarray = np.where(self.pixN == PixN)[0][0]
                print(numinarray)
                if hasattr(self.MAPviewdialog, 'commongui'):
                    self.MAPviewdialog.slider.setValue(numinarray)
                else:
                    print(PixN)
            except:
                print('data point is not in the array')
            ii = ii+1
        # self.select_pts()

    def closeEvent(self, event):

        if not hasattr(self.MAPviewdialog, 'commongui'):
            # global Pars = self.Pars
            print('module exit')
            QApplication.quit()
        else:
            self.MAPviewdialog.Pars = self.Pars
            print('module_PLpart_finished')
            self.close()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    try:
        del app
    except:
        print('noapp')

    app = QApplication(sys.argv)
    main = maps_view()
    main.show()
    sys.exit(app.exec_())
