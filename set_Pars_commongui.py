# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:34:15 2020

@author: Yuri
to use with common pyQt5 gui
"""


import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, \
    QVBoxLayout, QHBoxLayout, QLineEdit, \
    QPushButton,  QComboBox, QCheckBox, QTableView
from PyQt5.QtCore import Qt

import numpy as np

# from Pars_class import Pars_gen
from relaxation_functions import relaxation_function
# from common_gui_PyQt5 import TableModel


def dicttolist(Dict):
    temp = []
    dictlist = []
    for key, value in Dict.items():
        temp = [key, value]
        dictlist.append(temp)
    return dictlist


class set_Pars(QMainWindow):

    def __init__(self, parent=None):
        super(set_Pars, self).__init__(parent)
        self.set_Pars_gui = parent  # share variables
        if not hasattr(self.set_Pars_gui, 'commongui'):
            # from common_gui_PyQt5 import TableModel
            self.initUI()

    def initUI(self):
        if not hasattr(self.set_Pars_gui, 'commongui'):
            # self.open_vars()
            global Pars, Results, Data
            self.Pars = Pars
            self.Results = Results
            self.Data = Data
            self.Results = self.Results.drop(self.Results.index[np.size(self.Data, 0):])  # size correction
            self.Results.Pixel = self.Data[:, 0]  # remove nans
        else:
            self.Pars = self.set_Pars_gui.Pars
            self.Data = self.set_Pars_gui.Data
            self.Results = self.set_Pars_gui.Results

        # self.initial_check_fill()
        self.labelInd = QLabel('Indenter parameters')
        self.experiment_name = QLabel(self.Pars.fnameshort[0])
        self.probe_type_label = QLabel('Indenter type: ')
        self.probe_type = QLabel(self.Pars.TipType)
        self.probe_geoms_label = QLabel('Indenter geometry')
        self.probe_geoms = ['sphere', 'cone', 'cylinder', 'unsupported']
        self.cbProbe = QComboBox()
        self.cbProbe.addItems(self.probe_geoms)

        self.label_R = QLabel()  # self.probe_geom_shape
        self.le_R = QLineEdit(self)
        # self.le_R.setText(str(self.Pars.probe_dimension))
        self.label_R_units = QLabel()  # self.probe_geom_units
        self.cbProbe.currentIndexChanged.connect(self.change_geometry)

        self.label_k = QLabel('Indenter stiffness')
        self.le_k = QLineEdit(self)
        # self.le_k.setText(str(self.Pars.k))
        self.label_k_units = QLabel('[N/m]')

        self.label_InvOLS = QLabel('InvOLS')
        self.le_InvOLS = QLineEdit(self)
        # self.le_InvOLS.setText(str(self.Pars.InvOLS))
        self.label_InvOLS_units = QLabel('[nm/unit]')

        self.label_dT = QLabel('Sampling time')
        self.le_dT = QLineEdit(self)
        self.label_dT_units = QLabel('[s]')

        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('Modify processing parameters')

        self.labelSample = QLabel('Sample')
        self.label_models = QLabel('Model')
        self.vmodels = ['elastic', 'SLS', 'sPLR', 'sPLReta', 'sPLRetatest']
        self.cb_models = QComboBox()
        self.cb_models.addItems(self.vmodels)
        # self.cb_models.currentIndexChanged.connect(self.update_Pars)
        self.cb_models.currentIndexChanged.connect(self.changeviscomodel)
        self.label_Poisson = QLabel("Poisson's ratio")
        self.le_Poisson = QLineEdit(self)
        # self.le_Poisson.setText(str(self.Pars.Poisson))

        self.VmodelImage = QLabel(self)
        self.VmodelImage.setScaledContents(True)
        pixmap = QtGui.QPixmap("images/elastic.png")
        pixmap2 = pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio)
        self.VmodelImage.setPixmap(pixmap2)
        self.table3 = QTableView()
        listVisco = [['visco-par1', float(1000)],
                     ['visco-par2', float(0.2)],
                     ['visco-par3', float(0)],
                     ['visco-par4', float(0)],
                     ['visco-par5', float(0)]]
        self.model3 = TableModel(listVisco)
        self.table3.setModel(self.model3)
        self.fixpars_label = QLabel("Fixed parameters")
        self.fixpars_box1 = QCheckBox('#1', self)
        self.fixpars_box2 = QCheckBox('#2', self)
        self.fixpars_box3 = QCheckBox('#3', self)
        # self.fixpars_box1.stateChanged.connect(self.update_Pars)
        # self.fixpars_box2.stateChanged.connect(self.update_Pars)
        # self.fixpars_box3.stateChanged.connect(self.update_Pars)

        self.labelOptions = QLabel('Processing options')
        self.downsampling_label = QLabel('Downsampling')
        self.downsampling_types = ['off', 'moderate', 'strong']
        self.cb_downsampling_types = QComboBox()
        self.cb_downsampling_types.addItems(self.downsampling_types)
        self.hydro_box = QCheckBox('Hydrodynamic correction', self)
        # self.hydro_box.toggle()
        # self.hydro_box.stateChanged.connect(self.update_Pars)
        self.hydro_types_label = QLabel('hydro. correction type')
        self.hydro_types = ['find corr. coeff. automatically', 'use pre-defined corr. coeff.', 'biomomentum']
        self.cb_hydro_types = QComboBox()
        self.cb_hydro_types.addItems(self.hydro_types)
        # self.cb_hydro_types.currentIndexChanged.connect(self.update_Pars)
        self.hydro_coeff_label = QLabel('Coefficient of viscous drag: ')
        self.hydro_coeff_le = QLineEdit(self)
        # self.hydro_coeff_le.setText(str(self.Pars.hydro.speedcoef))

        self.BEC_box = QCheckBox('Bottom-effect correction', self)
        # self.BEC_box.toggle()
        # self.BEC_box.stateChanged.connect(self.update_Pars)

        self.AdjCP_box = QCheckBox('Adjust contact point position for viscoelastic models', self)
        # self.AdjCP_box.toggle()

        # self.cbProbe.currentIndexChanged.connect(self.update_Pars)
        # self.AdjCP_box.stateChanged.connect(self.update_Pars)

        self.labelFitlimits = QLabel('Limits for the elastic fit (% of max indentation)')
        self.labellowerlimit = QLabel('Lower')
        self.labelupperlimit = QLabel('Upper')
        self.le_depth_start = QLineEdit(self)  # Pars.depth_start
        self.le_depth_end = QLineEdit(self)

        self.OK_btn = QPushButton("Accept", self)
        self.OK_btn.clicked.connect(self.close)

        layoutA = QVBoxLayout()
        layoutAA = QHBoxLayout()
        layoutAAB = QHBoxLayout()
        layoutAB = QHBoxLayout()
        layoutAC = QHBoxLayout()
        layoutAD = QHBoxLayout()
        layoutAE = QHBoxLayout()
        layoutAA.addWidget(self.probe_type_label)
        layoutAA.addWidget(self.probe_type)
        layoutAAB.addWidget(self.probe_geoms_label)
        layoutAAB.addWidget(self.cbProbe)
        layoutAB.addWidget(self.label_R)
        layoutAB.addWidget(self.le_R)
        layoutAB.addWidget(self.label_R_units)
        layoutAC.addWidget(self.label_k)
        layoutAC.addWidget(self.le_k)
        layoutAC.addWidget(self.label_k_units)
        layoutAD.addWidget(self.label_InvOLS)
        layoutAD.addWidget(self.le_InvOLS)
        layoutAD.addWidget(self.label_InvOLS_units)
        layoutAE.addWidget(self.label_dT)
        layoutAE.addWidget(self.le_dT)
        layoutAE.addWidget(self.label_dT_units)

        layoutA.addWidget(self.labelInd)
        layoutA.addWidget(self.experiment_name)
        layoutA.addLayout(layoutAA)
        layoutA.addLayout(layoutAAB)
        layoutA.addLayout(layoutAB)
        layoutA.addLayout(layoutAC)
        layoutA.addLayout(layoutAD)
        layoutA.addLayout(layoutAE)

        layoutB = QVBoxLayout()
        layoutBA = QHBoxLayout()
        layoutBB = QHBoxLayout()
        layoutBC = QHBoxLayout()
        layoutBA.addWidget(self.label_models)
        layoutBA.addWidget(self.cb_models)
        layoutBB.addWidget(self.label_Poisson)
        layoutBB.addWidget(self.le_Poisson)
        layoutBC.addWidget(self.fixpars_label)
        layoutBC.addWidget(self.fixpars_box1)
        layoutBC.addWidget(self.fixpars_box2)
        layoutBC.addWidget(self.fixpars_box3)

        layoutB.addWidget(self.labelSample)
        layoutB.addLayout(layoutBA)
        layoutB.addLayout(layoutBB)
        layoutB.addWidget(self.table3)
        layoutB.addLayout(layoutBC)
        layoutB.addWidget(self.VmodelImage)

        layoutC = QVBoxLayout()
        layoutCA = QHBoxLayout()
        layoutCA2 = QHBoxLayout()
        layoutCA.addWidget(self.hydro_types_label)
        layoutCA.addWidget(self.cb_hydro_types)
        layoutCA2.addWidget(self.hydro_coeff_label)
        layoutCA2.addWidget(self.hydro_coeff_le)
        layoutCB = QHBoxLayout()
        layoutCB.addWidget(self.labellowerlimit)
        layoutCB.addWidget(self.le_depth_start)
        layoutCB.addWidget(self.labelupperlimit)
        layoutCB.addWidget(self.le_depth_end)
        layoutDS = QHBoxLayout()
        layoutDS.addWidget(self.downsampling_label)
        layoutDS.addWidget(self.cb_downsampling_types)

        layoutC.addWidget(self.labelOptions)
        layoutC.addLayout(layoutDS)
        layoutC.addWidget(self.hydro_box)
        layoutC.addLayout(layoutCA)
        layoutC.addLayout(layoutCA2)
        layoutC.addWidget(self.BEC_box)
        layoutC.addWidget(self.AdjCP_box)
        layoutC.addWidget(self.labelFitlimits)
        layoutC.addLayout(layoutCB)
        layoutC.addWidget(self.OK_btn)

        layoutA.setAlignment(Qt.AlignTop)  # alignment of widgets
        layoutB.setAlignment(Qt.AlignTop)  # alignment of widgets
        layoutC.setAlignment(Qt.AlignTop)  # alignment of widgets

        layoutM = QHBoxLayout()
        layoutM.addLayout(layoutA)
        layoutM.addLayout(layoutB)
        layoutM.addLayout(layoutC)

        widget = QWidget()
        widget.setLayout(layoutM)
        self.initial_check_fill()
        self.changeviscomodel()
        self.setCentralWidget(widget)
        self.show()
        self.activateWindow()

    def initial_check_fill(self):

        if hasattr(self.Pars, 'probe_shape'):
            if self.Pars.probe_shape == 'sphere':
                self.probe_geom_ndx = 0
                self.probe_geom_shape = 'radius'
                self.probe_geom_units = '[nm]'
            elif self.Pars.probe_shape == 'cone':
                self.probe_geom_ndx = 1
                self.probe_geom_shape = 'half-opening angle'
                self.probe_geom_units = '[degrees]'
            elif self.Pars.probe_shape == 'cylinder':
                self.probe_geom_ndx = 2
                self.probe_geom_shape = 'radius'
                self.probe_geom_units = '[nm]'
            else:
                self.probe_geom_ndx = 3
        else:
            self.probe_geom_ndx = 0
            self.probe_geom_shape = 'radius'
            self.probe_geom_units = '[nm]'
        self.cbProbe.setCurrentIndex(self.probe_geom_ndx)
        self.label_R.setText(self.probe_geom_shape)
        self.le_R.setText(str(self.Pars.probe_dimension))
        self.label_R_units.setText(str(self.probe_geom_units))
        self.le_k.setText(str(self.Pars.k))
        self.le_InvOLS.setText(str(self.Pars.InvOLS))
        self.le_dT.setText(str(self.Pars.dT))
        self.le_Poisson.setText(str(self.Pars.Poisson))
        self.cb_models_ndx = self.vmodels.index(self.Pars.viscomodel)
        self.cb_models.setCurrentIndex(self.cb_models_ndx)
        if not hasattr(self.Pars, 'fixed_values'):
            self.Pars.fixed_values = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)  # complex
        self.fixedinds = np.array(self.Pars.fixed_values[0], dtype=int)

        listVisco = [['visco-par1', float(self.Pars.fixed_values[1, 0])],
                     ['visco-par2', float(self.Pars.fixed_values[1, 1])],
                     ['visco-par3', float(self.Pars.fixed_values[1, 2])],
                     ['visco-par4', float(0)],
                     ['visco-par5', float(0)]]
        self.model3 = TableModel(listVisco)
        self.table3.setModel(self.model3)
        if self.fixedinds[0] == 1:
            self.fixpars_box1.setChecked(True)
        else:
            self.fixpars_box1.setChecked(False)
        if self.fixedinds[1] == 1:
            self.fixpars_box2.setChecked(True)
        else:
            self.fixpars_box2.setChecked(False)
        if self.fixedinds[2] == 1:
            self.fixpars_box3.setChecked(True)
        else:
            self.fixpars_box3.setChecked(False)
        if self.Pars.hydro.corr == 1:
            self.hydro_box.setChecked(True)
        else:
            self.hydro_box.setChecked(False)
        if self.Pars.hydro.corr_type == 1:
            self.cb_hydro_types.setCurrentIndex(0)
        elif self.Pars.hydro.corr_type == 2:
            self.cb_hydro_types.setCurrentIndex(1)
        elif self.Pars.hydro.corr_type == 3:
                self.cb_hydro_types.setCurrentIndex(2)
        self.hydro_coeff_le.setText(str(self.Pars.hydro.speedcoef))
        if self.Pars.HeightfromZ == 1 or self.Pars.height > 0:
            self.BEC_box.setChecked(True)
        else:
            self.BEC_box.setChecked(False)
        if self.Pars.cp_adjust == 1:
            self.AdjCP_box.setChecked(True)
        else:
            self.AdjCP_box.setChecked(False)
        if not hasattr(self.Pars, 'depth_start'):
            self.Pars.depth_start = 5
            self.Pars.depth_end = 95
        self.le_depth_start.setText(str(self.Pars.depth_start))
        self.le_depth_end.setText(str(self.Pars.depth_end))
        
        if not hasattr(self.Pars, 'downsampling'):
            self.Pars.downsampling = 0
        self.cb_downsampling_types.setCurrentIndex(self.Pars.downsampling)        

    def update_Pars(self):
        self.changeviscomodel()
        self.Pars.probe_shape = str(self.cbProbe.currentText())
        self.Pars.probe_dimension = float(self.le_R.text())
        self.Pars.k = float(self.le_k.text())
        self.Pars.InvOLS = float(self.le_InvOLS.text())
        self.Pars.dT = float(self.le_dT.text())
        self.Pars.Poisson = float(self.le_Poisson.text())
        self.Pars.viscomodel = str(self.cb_models.currentText())
        model3 = self.table3.model()  # viscoelastic parameters
        vpars = []
        for row in range(3):
            vpars.append([])
            column = 1
            index = model3.index(row, column)
            try:
                vpars[row].append(float(model3.data(index, 0)))
            except:
                vpars[row].append((model3.data(index, 0)))
        print(vpars)
        vpars = np.squeeze(vpars)
        self.Pars.fixed_values[1, :] = vpars
        if self.fixpars_box1.isChecked():
            self.Pars.fixed_values[0, 0] = 1
        else:
            self.Pars.fixed_values[0, 0] = 0
        if self.fixpars_box2.isChecked():
            self.Pars.fixed_values[0, 1] = 1
        else:
            self.Pars.fixed_values[0, 1] = 0
        if self.fixpars_box3.isChecked():
            self.Pars.fixed_values[0, 2] = 1
        else:
            self.Pars.fixed_values[0, 2] = 0

        if self.hydro_box.isChecked():
            self.Pars.hydro.corr = 1
        else:
            self.Pars.hydro.corr = 0
        if self.cb_hydro_types.currentIndex() == 0:
            self.Pars.hydro.corr_type = 1
        elif self.cb_hydro_types.currentIndex() == 1:
            self.Pars.hydro.corr_type = 2
        self.Pars.hydro.speedcoef = float(self.hydro_coeff_le.text())
        if self.BEC_box.isChecked():  # split heightfromZ, height
            self.Pars.HeightfromZ = 1
        else:
            self.Pars.HeightfromZ = 0
        if self.AdjCP_box.isChecked():  # split heightfromZ, height
            self.Pars.cp_adjust = 1
        else:
            self.Pars.cp_adjust = 0
        self.Pars.downsampling = self.cb_downsampling_types.currentIndex()
        self.Pars.depth_start = float(self.le_depth_start.text())
        self.Pars.depth_end = float(self.le_depth_end.text())
        # print (self.Pars.probe_shape)
        self.initial_check_fill()
        # self.initUI()
        # pass

    def change_geometry(self):
        self.probe_geom_ndx = self.cbProbe.currentIndex()
        if self.probe_geom_ndx == 0:
            self.probe_geom_shape = 'radius'
            self.probe_geom_units = '[nm]'
        elif self.probe_geom_ndx == 1:
            self.probe_geom_shape = 'half-opening angle'
            self.probe_geom_units = '[degrees]'
        elif self.probe_geom_ndx == 2:
            self.probe_geom_shape = 'radius'
            self.probe_geom_units = '[nm]'
        self.label_R.setText(self.probe_geom_shape)
        self.label_R_units.setText(str(self.probe_geom_units))

    def changeviscomodel(self):
        self.viscomodel = str(self.cb_models.currentText())
        viscpars = relaxation_function([0.4, 0.3, 0.2, 0.1, 0], self.viscomodel, np.ones(1))[2]
        # self.labelViscoHelp.setText(viscomodel + ' pars: ' + str(viscpars))

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
            filevname = "images/" + self.viscomodel + ".png"
            pixmap = QtGui.QPixmap(filevname)
            pixmap2 = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            self.VmodelImage.setPixmap(pixmap2)
        except:
            print('image for the viscomodel does not exist')

    def closeEvent(self, event):
        self.update_Pars()
        if not hasattr(self.set_Pars_gui, 'commongui'):
            global Pars
            Pars = self.Pars
            print('module exit')
            QApplication.quit()
        else:
            self.set_Pars_gui.Pars = self.Pars
            listPars = self.Pars.class2list()  # dicttolist(Pars)
            self.set_Pars_gui.model = TableModel(listPars)
            self.set_Pars_gui.table.setModel(self.set_Pars_gui.model)
            print('module_setPars_finished')
            self.close()


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
    from utils_ViscoIndent import load_AFM_data_pickle, load_AFM_data_pickle_short
    try:
        del app
    except:
        print('noapp')

    if 'Pars' not in globals():
        from file_import_qt5simple import file_import_dialog_qt5
        file_import_dialog_qt5
        filename = file_import_dialog_qt5()[0]
        Pars, Data, Results = load_AFM_data_pickle_short(filename)
    Results = Results.drop(Results.index[np.size(Data, 0):])  # size correction
    Results.Pixel = Data[:, 0]  # remove nans
    app = QApplication(sys.argv)
    main = set_Pars()
    main.show()
    sys.exit(app.exec_())
