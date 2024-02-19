# -*- coding: utf-8 -*-
"""
gui for viscoindent, v. January-2024
with images for viscomodels
gui_Viscoindent
"""

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QSizePolicy,\
                            QVBoxLayout, QHBoxLayout, QLabel,\
                            QPushButton, QComboBox, QDoubleSpinBox, \
                            QItemEditorFactory, QStyledItemDelegate
from PyQt5.QtCore import Qt, QPersistentModelIndex, QModelIndex, QVariant

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
import numpy as np
# import csv

from tingFC_constructor import tingFC_constructor
from relaxation_functions import relaxation_function, modellist


def dicttolist(Dict):
    temp = []
    dictlist = []
    for key, value in Dict.items():
        temp = [key, value]
        dictlist.append(temp)
    return dictlist


class App(QMainWindow):

    def read_data(self):
        model = self.table.model()  # general parameters (indenter, etc.)
        model2 = self.table2.model()  # indentation history
        model3 = self.table3.model()  # viscoelastic model (relaxation function) parameters
        Pars2 = []
        for row in range(model.rowCount(0)):
            Pars2.append([])
            for column in range(model.columnCount(0)):
                index = model.index(row, column)
                try:
                    Pars2[row].append(float(model.data(index, 0)))
                except:
                    Pars2[row].append((model.data(index, 0)))

        indpars = []
        for row in range(model2.rowCount(0)):
            indpars.append([])
            column = 1
            index = model2.index(row, column)
            try:
                indpars[row].append(float(model2.data(index, 0)))
            except:
                indpars[row].append((model2.data(index, 0)))
        # print(indpars)
        indpars = np.squeeze(indpars)

        vpars = []
        for row in range(model3.rowCount(0)):
            vpars.append([])
            column = 1
            index = model3.index(row, column)
            try:
                vpars[row].append(float(model3.data(index, 0)))
            except:
                vpars[row].append((model3.data(index, 0)))
        # print(vpars)
        vpars = np.squeeze(vpars)

        Pars = dict(Pars2)
        Pars['indpars'] = indpars
        Pars['Vpars'] = vpars
        Pars['graph'] = str(self.graphT.currentText())
        # print(Pars)
        self.ParsCur = Pars
        indentationfull = np.array([0, 1])
        time, ind, force, cradius, indentationfullL, forceL = tingFC_constructor(Pars, indentationfull)
        self.curve_data = [time, ind, force, cradius, indentationfullL, forceL]

    def button1(self):
        self.read_data()
        curve_data = self.curve_data
        Pars = self.ParsCur
        print(Pars)
        viscpars = relaxation_function(Pars['Vpars'], Pars['viscomodel'], np.ones(1))[2]
        self.labelViscoHelp.setText(Pars['viscomodel'] + ' pars: ' + str(viscpars))
        self.curvedata = PlotCanvas.plot(self.m, Pars, curve_data)
        self.show()

    def button2(self):
        Pars = self.ParsCur
        # print(self.curvedata)
        arr = np.vstack([self.curve_data[0], self.curve_data[1], self.curve_data[2]])
        np.savetxt('force_curve_data.csv', arr.transpose(), delimiter=',',
                   fmt=['%f', '%f', '%f'], header=str(Pars) + '\n' +
                   'time; indentation; force')
        # with open('force_curve_data.csv', mode='w', newline='') as csv_file:
        #     wr = csv.writer(csv_file)
        #     wr.writerow(self.curvedata)

    def changeviscomodel(self):
        viscomodel = str(self.cbDel2.currentText())
        viscpars = relaxation_function([0.4, 0.3, 0.2, 0.1, 0], viscomodel, np.ones(1))[2]
        self.labelViscoHelp.setText(viscomodel + ' pars: ' + str(viscpars))

        for ij in range(len(viscpars)):
            indx = self.table3.model().index(ij, 0)
            self.table3.model().setData(indx, viscpars[ij], 0)
            self.table3.selectRow(ij)
            self.table3.setRowHidden(ij, False)
        self.table3.clearSelection()
        if ij < 4:
            for ik in range(4, ij, -1):
                self.table3.setRowHidden(ik, True)
        try:
            filevname = "images/" + viscomodel + ".png"
            pixmap = QtGui.QPixmap(filevname)
            pixmap2 = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            self.VmodelImage.setPixmap(pixmap2)
        except:
            print('image for the viscomodel does not exist')

    def changefigaxis(self):
        self.curvedata = PlotCanvas.plot(self.m, self.ParsCur, self.curve_data)
        self.show()

    def __init__(self):
        super().__init__()
        self.left = 50
        self.top = 50
        self.title = 'PyQt5 gui for ViscIndent'
        self.width = 1200
        self.height = 750
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        Pars = {}
        Pars['probe_shape'] = 'sphere'
        Pars['probe_dimension'] = 5000
        Pars['Poisson'] = 0.5         # Poisson's ratio of the sample
        Pars['dT'] = 1e-3                 # Sampling time
        Pars['height'] = 0
        Pars['viscomodel'] = 'sPLR'
        # Pars['indpars'] = np.array([1, 50, 50, 1000, 1]
        # Pars['Vpars'] = np.array([1000, 0.8, 0, 20])
        Pars['noise'] = 0.0 # % noise level from median force
        Pars['hydrodrag'] = 0.000  # [nN*s/nm] coefficient of viscous drag
        Pars['adhesion'] = 0.000

        IndPars = {}  # [yes/no; depth; speed; numpoimts; ramp/sin; dwell_time];
        IndPars['define_indentation'] = 1  # Pars['indpars'][0]
        IndPars['depth (nm)'] = 50  # Pars['indpars'][1]
        IndPars['speed (nm/s)'] = 50  # Pars['indpars'][2]
        IndPars['number of pts'] = 1000  # Pars['indpars'][3]
        IndPars['tri(0) or sin(1)'] = 0  # Pars['indpars'][4]
        IndPars['dwell_time (s)'] = 0.0  # Pars['indpars'][5]

        ViscoPars = {}
        ViscoPars['visco-par1'] = float(1000)  # Pars['Vpars'][0]
        ViscoPars['visco-par2'] = float(0.2)  # Pars['Vpars'][1]
        ViscoPars['visco-par3'] = float(0)  # Pars['Vpars'][2]
        ViscoPars['visco-par4'] = float(0)  # Pars['Vpars'][3]
        ViscoPars['visco-par5'] = float(0)  # Pars['Vpars'][3]

        # Pars.pop('Vpars', None)
        # Pars.pop('indpars', None)
        listPars = dicttolist(Pars)
        listInd = dicttolist(IndPars)
        listVisco = dicttolist(ViscoPars)

        Pars['indpars'] = np.squeeze(list(IndPars.values()))
        Pars['Vpars'] = np.squeeze(list(ViscoPars.values()))

        styledItemDelegate=QStyledItemDelegate()
        styledItemDelegate.setItemEditorFactory(ItemEditorFactory())

        self.table = QtWidgets.QTableView()
        self.model = TableModel(listPars)
        self.table.setItemDelegate(styledItemDelegate)
        self.table.setModel(self.model)

        indx = self.table.model().index(0, 1)
        pix = QPersistentModelIndex(indx)
        cbDel = QComboBox()
        cbDel.currentIndexChanged[str].connect(lambda txt, pix=pix: self.table.model().setData(QModelIndex(pix), txt, 0))
        cbDel.addItems(['sphere', 'cone', 'cylinder'])
        self.table.setIndexWidget(indx, cbDel)

        indx2 = self.table.model().index(5, 1)
        pix2 = QPersistentModelIndex(indx2)
        self.cbDel2 = QComboBox()
        self.cbDel2.currentIndexChanged[str].connect(lambda txt, pix2=pix2: self.table.model().setData(QModelIndex(pix2), txt, 0))
        self.cbDel2.addItems(modellist())
        self.cbDel2.setCurrentIndex(9)
        self.cbDel2.currentIndexChanged.connect(self.changeviscomodel)
        self.table.setIndexWidget(indx2, self.cbDel2)
        self.table.setRowHidden(3, True)

        self.table2 = QtWidgets.QTableView()
        self.model2 = TableModel(listInd)
        self.table2.setItemDelegate(styledItemDelegate)
        self.table2.setModel(self.model2)
        self.table2.setRowHidden(0, True)

        self.table3 = QtWidgets.QTableView()
        self.model3 = TableModel(listVisco)
        self.table3.setItemDelegate(styledItemDelegate)
        self.table3.setModel(self.model3)

        self.graphT = QComboBox()
        self.graphT.addItems(['Force versus Indentation', 'Force versus Time', 'Force versus Displacement'])
        self.graphT.currentIndexChanged.connect(self.button1)

        self.labelViscoHelp = QLabel(str(self.cbDel2.currentText()) + ' pars: E1, alpha', self)
        # self.changeviscomodel()
        # self.model3 = TableModel(listVisco)

        Pars['graph'] = 'Force versus Indentation'
        self.read_data()
        curve_data = self.curve_data
        self.m = PlotCanvas(self, Pars, curve_data, width=5, height=4)
        self.curvedata = PlotCanvas.plot(self.m, Pars, curve_data)

        button = QPushButton('Update', self)
        button.setToolTip('Update the grapg')
        button.clicked.connect(self.button1)

        button2 = QPushButton('Export csv', self)
        button2.clicked.connect(self.button2)

        self.VmodelImage = QLabel(self)
        self.VmodelImage.setScaledContents(True)
        pixmap = QtGui.QPixmap("images/sPLR.png")
        pixmap2 = pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio)
        self.VmodelImage.setPixmap(pixmap2)

        self.changeviscomodel()
        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()
        layoutV = QVBoxLayout()
        layout3 = QVBoxLayout()

        layout2.addWidget(button)
        layout2.addWidget(self.table)
        layout2.addWidget(self.table2)

        layoutV.addWidget(self.labelViscoHelp)
        layoutV.addWidget(self.table3)
        layoutV.addWidget(self.VmodelImage)

        layout3.addWidget(self.graphT)
        layout3.addWidget(self.m)
        layout3.addWidget(button2)

        layout1.addLayout(layout2)
        layout1.addLayout(layoutV)
        layout1.addLayout(layout3)

        # three commands to apply layout
        widget = QWidget()
        widget.setLayout(layout1)
        self.setCentralWidget(widget)

        self.show()
        self.activateWindow()

    def closeEvent(self, event):
        QApplication.quit()


class PlotCanvas(FigureCanvas):

    def __init__(self, QWidget, Pars, curve_data, parent=None, width=5, height=4, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.plot(Pars)

    def plot(self, Pars, curve_data):

        time, ind, force, cradius, indentationfullL, forceL = curve_data
        ax = self.axes
        ax.clear()
        if Pars['graph'] == 'Force versus Indentation':
            ax.plot(ind, force, 'r-')
            ax.set_title('Force versus Indentation')
            ax.set_xlabel('Indentation, nm')
            ax.set_ylabel('Force, nN')
        elif Pars['graph'] == 'Force versus Time':
            ax.plot(time, force, 'r-')
            ax.set_title('Force vs Time')
            ax.set_xlabel('Time, s')
            ax.set_ylabel('Force, nN')
        elif Pars['graph'] == 'Force versus Displacement':
            ax.plot(indentationfullL, forceL, 'r-')
            ax.set_title('Force versus Displacement')
            ax.set_xlabel('Displacement, nm')
            ax.set_ylabel('Force, nN')
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

        if role == Qt.EditRole or role == Qt.DisplayRole:  # edit without clear
            return QtCore.QVariant(self._data[index.row()][index.column()])

        # return QtCore.QVariant()

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

class ItemEditorFactory(QItemEditorFactory):  # http://doc.qt.io/qt-5/qstyleditemdelegate.html#subclassing-qstyleditemdelegate    It is possible for a custom delegate to provide editors without the use of an editor item factory. In this case, the following virtual functions must be reimplemented:
    def __init__(self):
        super().__init__()

    def createEditor(self, userType, parent):
        if userType == QVariant.Double:
            doubleSpinBox = QDoubleSpinBox(parent)
            doubleSpinBox.setDecimals(6)
            doubleSpinBox.setMaximum(1e12)  # The default maximum value is 99.99
            return doubleSpinBox
        else:
            return super().createEditor(userType, parent)


if __name__ == '__main__':
    try:
        del app
    except:
        print('noapp')
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
