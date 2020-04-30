# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:10:02 2020


gui for viscoindent, from 21-Apr-20
"""

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
# from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QPersistentModelIndex, QModelIndex

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import csv

from tingFC_constructor import tingFC_constructor
from relaxation_functions import relaxation_function, modellist

def dicttolist(Dict):
    temp = []
    dictlist = []   
    for key, value in Dict.items():
        temp = [key,value]
        dictlist.append(temp)
    return dictlist  

class App(QMainWindow):
 
    
    def button1(self):
        #print('slot method called.')
        model = self.table.model()
        model2 = self.table2.model()
        model3 = self.table3.model()
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
            column=1
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
            column=1
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
        print(Pars)
        self.ParsCur = Pars
        viscpars = relaxation_function(Pars['Vpars'], Pars['viscomodel'], np.zeros(1))[2]
        self.labelViscoHelp.setText(Pars['viscomodel'] + ' pars: '+ str(viscpars))
        self.curvedata = PlotCanvas.plot(self.m, Pars)
        self.show()
    
    def button2(self):
        Pars = self.ParsCur
        print(self.curvedata)
        arr = np.vstack([self.curvedata[0], self.curvedata[1], self.curvedata[2]])
        np.savetxt('force_curve_data.csv', arr.transpose(), delimiter=',', fmt=['%f' , '%f', '%f'] , header=str(Pars)+ '\n'+'time; indentation; force')
        # with open('force_curve_data.csv', mode='w', newline='') as csv_file:
        #     wr = csv.writer(csv_file)
        #     wr.writerow(self.curvedata)
        
    def changeviscomodel(self):
        viscomodel = str(self.cbDel2.currentText())
        viscpars = relaxation_function([1, 1, 1, 1], viscomodel, np.zeros(1))[2]
        self.labelViscoHelp.setText(viscomodel + ' pars: '+ str(viscpars))
        
            

    def __init__(self):
        super().__init__()
        self.left = 50
        self.top = 50
        self.title = 'PyQt5 gui for ViscIndent'
        self.width = 840
        self.height = 700
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        
        Pars={}
        Pars['probe_shape'] = 'sphere'
        Pars['probe_dimension'] = 5000
        Pars['Poisson'] = 0.5         # Poisson's ratio of the sample
        Pars['dT'] = 1e-3                 # Sampling time
        Pars['height'] = 0
        Pars['viscomodel'] = 'sPLR'
        # Pars['indpars'] = np.array([1, 50, 50, 1000, 1])
        # Pars['Vpars'] = np.array([1000, 0.8, 0, 20]) 
        
        IndPars = {} # [yes/no; depth; speed; numpoimts; ramp/sin];
        IndPars['define_indentation'] = 1 # Pars['indpars'][0]
        IndPars['depth'] = 50 # Pars['indpars'][1]
        IndPars['speed'] = 50 # Pars['indpars'][2]
        IndPars['number of pts'] = 1000 # Pars['indpars'][3]
        IndPars['tri(0) or sin(1)'] = 0 # Pars['indpars'][4]
        
        ViscoPars = {}
        ViscoPars['visco-par1'] = 1000 # Pars['Vpars'][0]
        ViscoPars['visco-par2'] = 0.5 # Pars['Vpars'][1]
        ViscoPars['visco-par3'] = 0 # Pars['Vpars'][2]
        ViscoPars['visco-par4'] = 0 # Pars['Vpars'][3]

        # Pars.pop('Vpars', None)
        # Pars.pop('indpars', None)
        data = dicttolist(Pars)
        dataind = dicttolist(IndPars)
        datavisco = dicttolist(ViscoPars)
        
        Pars['indpars'] = np.squeeze(list(IndPars.values()))
        Pars['Vpars'] = np.squeeze(list(ViscoPars.values()))
        
        self.table = QtWidgets.QTableView()
        self.model = TableModel(data)
        self.table.setModel(self.model)
        
        indx = self.table.model().index(0, 1)
        pix = QPersistentModelIndex(indx)
        cbDel = QComboBox()
        cbDel.currentIndexChanged[str].connect(lambda txt, pix=pix:self.table.model().setData(QModelIndex(pix), txt, 0))
        cbDel.addItems(['sphere','cone','cylinder'])
        self.table.setIndexWidget(indx,cbDel)        
        
        indx2 = self.table.model().index(5, 1)
        pix2 = QPersistentModelIndex(indx2)
        self.cbDel2 = QComboBox()
        self.cbDel2.currentIndexChanged[str].connect(lambda txt, pix2=pix2:self.table.model().setData(QModelIndex(pix2), txt, 0))
        self.cbDel2.addItems(modellist())
        self.cbDel2.setCurrentIndex(9)
        self.cbDel2.currentIndexChanged.connect(self.changeviscomodel)
        self.table.setIndexWidget(indx2,self.cbDel2)
        self.table.setRowHidden(3, True)
        
        self.table2 = QtWidgets.QTableView()
        self.model2 = TableModel(dataind)
        self.table2.setModel(self.model2)
        
        self.table2.setRowHidden(0, True)
        
        self.table3 = QtWidgets.QTableView()
        self.model3 = TableModel(datavisco)
        self.table3.setModel(self.model3)
        Pars['graph'] = 'Force versus Indentation'

        self.m = PlotCanvas(self, Pars, width=5, height=4)
        self.curvedata = PlotCanvas.plot(self.m, Pars)
        
        button = QPushButton('Update', self)
        button.setToolTip('Update the grapg')
        button.clicked.connect(self.button1)
        
        button2 = QPushButton('Export csv', self)
        button2.clicked.connect(self.button2)
        
        self.graphT = QComboBox()
        self.graphT.addItems(['Force versus Indentation','Force versus Time'])
        
        self.labelViscoHelp = QLabel(str(self.cbDel2.currentText()) + ' pars: E1, alpha', self)
        
        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()
        
        layout2.addWidget(button)
        layout2.addWidget(self.table)
        layout2.addWidget(self.table2)
        layout2.addWidget(self.labelViscoHelp)
        layout2.addWidget(self.table3)
        
        layout1.addLayout(layout2)
        
        layout3.addWidget(self.graphT)
        layout3.addWidget(self.m)
        layout3.addWidget(button2)
        
        layout1.addLayout(layout3)
        
        
        # three commands to apply layout
        widget = QWidget()
        widget.setLayout(layout1)
        self.setCentralWidget(widget)
        
        self.show()
        self.activateWindow()
        
    def closeEvent(self,event):
         QApplication.quit()



class PlotCanvas(FigureCanvas):

    def __init__(self, QWidget, Pars, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        #self.plot(Pars)


    def plot(self, Pars):
        
        indentationfull = np.array([0, 1])
        time, ind, force, cradius, indentationfullL, forceL = tingFC_constructor(Pars, indentationfull)
        ax = self.axes
        ax.clear()
        if Pars['graph'] == 'Force versus Indentation':
            ax.plot(ind, force, 'r-')
            ax.set_title('Force vs indentation')
        elif Pars['graph'] == 'Force versus Time':
            ax.plot(time, force, 'r-')
            ax.set_title('Force vs Time')    
        self.draw()
        curvedata = [time, ind, force, cradius, indentationfullL, forceL]
        return curvedata
        
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
        
    def setData(self, index, value, role):
        self._data[index.row()][index.column()] = value # or float(value)
        return True
    
    def flags(self, index):
        return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

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
    sys.exit(app.exec_())
