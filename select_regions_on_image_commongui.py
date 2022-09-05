# -*- coding: utf-8 -*-
"""
@author: Yuri
to use with common pyQt5 gui
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, \
    QRadioButton, QVBoxLayout, QHBoxLayout, QLineEdit, QButtonGroup, \
    QPushButton, QSpinBox
from PyQt5.QtCore import Qt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np

from Pars_class import Pars_gen


class selectROIs(QMainWindow):

    def __init__(self, parent=None):
        super(selectROIs, self).__init__(parent)
        self.ROIdialog = parent  # share variables
        if not hasattr(self.ROIdialog, 'commongui'):
            self.initUI()

    def initUI(self):
        # print('ROI_winodw_test1')
        if not hasattr(self.ROIdialog, 'commongui'):
            global Pars
            self.Pars = Pars
        else:
            self.Pars = self.ROIdialog.Pars
        self.topo2 = self.Pars.topo2
        self.ROImode = self.Pars.ROIPars.ROImode
        self.numofselections = 1
        self.mask_ar = [None]
        self.reg_nums = [None]

        self.fig = fig = Figure()
        self.canvas1 = FigureCanvasQTAgg(fig)
        self.ax1 = fig.add_subplot(111, title="Original image")

        self.sp01 = QSpinBox(self)
        self.sp01.setValue(1)
        self.sp01.valueChanged.connect(self.radio_button_clicked)

        self.le01 = QLineEdit(self)
        self.le01.setText(str(self.Pars.ROIPars.h_level))
        self.le01.editingFinished.connect(self.edit_height)

        self.label01 = QLabel('ROI selection options')
        self.btngroup1opts = [QRadioButton("no selection"),
                              QRadioButton("select above h-level"),
                              QRadioButton("draw"),
                              QRadioButton("draw and select above h-level")]
        self.btngroup1 = QButtonGroup()
        layoutA = QVBoxLayout()
        layoutA1 = QVBoxLayout()
        layoutA1.addWidget(self.label01)
        for i in range(len(self.btngroup1opts)):
            layoutA1.addWidget(self.btngroup1opts[i])
            self.btngroup1.addButton(self.btngroup1opts[i], i)
            self.btngroup1opts[i].clicked.connect(self.radio_button_clicked)
        self.btngroup1opts[self.ROImode].setChecked(True)

        layoutA.addLayout(layoutA1)

        layoutA2 = QVBoxLayout()
        self.labelhlevel = QLabel('Insert h-level and press enter:')
        layoutA2.addWidget(self.labelhlevel)
        layoutA2.addWidget(self.le01)
        layoutA3 = QVBoxLayout()
        self.label02 = QLabel('Number of ROIs')
        layoutA3.addWidget(self.label02)
        layoutA3.addWidget(self.sp01)
        layoutA.addLayout(layoutA2)
        layoutA.addLayout(layoutA3)
        self.Reselect_btn = QPushButton("Reselect ROIs", self)
        self.Reselect_btn.clicked.connect(self.Reselect_btn_clicked)

        self.OK_btn = QPushButton("Continue", self)
        # self.OK_btn.clicked.connect(self.OK_btn_clicked)
        self.OK_btn.clicked.connect(self.close)

        layoutB = QHBoxLayout()
        layoutB.addWidget(self.Reselect_btn)
        layoutB.addWidget(self.OK_btn)
        layoutA.addLayout(layoutB)

        # self.fig = fig = Figure()
        # self.canvas1 = FigureCanvasQTAgg(fig)
        # self.ax1 = fig.add_subplot(111, title="Original image")
        img1 = self.ax1.imshow(self.topo2, interpolation='nearest',cmap='viridis', origin='lower')  # 'jet'
        self.fig.colorbar(img1)

        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('Regions of interest (ROIs) selection')

        layoutM = QHBoxLayout()
        layoutM.addLayout(layoutA)
        layoutM.addWidget(self.canvas1)
        layoutA1.setAlignment(Qt.AlignTop)
        layoutA2.setAlignment(Qt.AlignTop)
        layoutA3.setAlignment(Qt.AlignTop)

        widget = QWidget()
        widget.setLayout(layoutM)
        self.setCentralWidget(widget)
        self.show()
        self.activateWindow()
        # print('ROI_winodw_test2')

    def Reselect_btn_clicked(self):

        self.ax1.clear()
        img1 = self.ax1.imshow(self.topo2, interpolation='nearest', cmap='viridis', origin='lower')
        # self.fig.colorbar(img1)
        self.canvas1.draw()
        self.numofselections = 1
        self.radio_button_clicked()

    def OK_btn_clicked(self):

        self.closeEvent(1)
        # pass

    def edit_height(self):
        self.radio_button_clicked()

    def radio_button_clicked(self):

        print(self.btngroup1.checkedId())
        self.ax1.clear()
        img1 = self.ax1.imshow(self.topo2, interpolation='nearest', cmap='viridis', origin='lower')
        # self.fig.colorbar(img1)
        self.canvas1.draw()

        self.Pars.ROIPars.ROImode = self.btngroup1.checkedId()
        self.Pars.ROIPars.num_of_ROIs = self.sp01.value()
        self.numcells = self.Pars.ROIPars.num_of_ROIs
        self.h_level = float(self.le01.text())
        self.mask_ar = [None]
        self.reg_nums = [None]

        if self.btngroup1.checkedId() == 0:
            self.mask_ar[0] = np.full(self.topo2.shape, True)  # np.ones(self.topo.shape, dtype=bool)
            self.reg_nums[0] = 1*np.reshape(self.mask_ar[0], self.topo2.size)
        elif self.btngroup1.checkedId() == 1:
            self.numofselections = 1
            print(self.h_level)
            self.mask_ar[0] = (self.topo2 > self.h_level)
            self.reg_nums[0] = 1*np.reshape(self.mask_ar[0], self.topo2.size)
            self.canvas1.draw()
            ar = self.mask_ar[0]
            ar[ar == 1] = np.nan
            self.ax1.imshow(ar, alpha=0.3, cmap="Reds", origin='lower')
            self.canvas1.draw()
        elif self.btngroup1.checkedId() in [2, 3]:
            self.mask_ar = [None]*self.numcells
            self.reg_nums = [None]*self.numcells
            self.lasso = LassoSelector(self.ax1, onselect=self.onselect)

    def onselect(self, verts):
        p = Path(verts)
        # Pixel coordinates
        xc = np.arange(self.topo2.shape[0])
        yc = np.arange(self.topo2.shape[1])
        xv, yv = np.meshgrid(xc, yc)
        pix = np.vstack((xv.flatten(), yv.flatten())).T
        maskC = p.contains_points(pix, radius=0.26)  # check radius
        mask = np.reshape(maskC, self.topo2.shape)
        if self.btngroup1.checkedId() == 3:
            mask = mask*(self.topo2 > self.h_level)
            maskC = maskC*(self.topo2.reshape(self.topo2.size) > self.h_level)
        # print(mask)
        # mask_ar=mask_ar.append(mask)
        # print(mask_ar)
        self.mask_ar[self.numofselections-1] = mask
        self.reg_nums[self.numofselections-1] = maskC*1*[self.numofselections]
        self.canvas1.draw_idle()

        self.ax1.contour(xc, yc, np.reshape(self.reg_nums[self.numofselections-1], self.topo2.shape))
        labelXcoorF = np.maximum.reduce(self.numofselections*mask, 0)
        labelXcoor = np.mean(xc[(labelXcoorF == self.numofselections)])
        labelYcoorF = np.maximum.reduce(self.numofselections*mask, 1)
        labelYcoor = np.mean(yc[(labelYcoorF == self.numofselections)])
        self.ax1.text(labelXcoor, labelYcoor, str(self.numofselections),
                      horizontalalignment='center', verticalalignment='center',
                      fontsize=20)

        self.numofselections = self.numofselections+1
        if self.numofselections == self.numcells+1:
            print("Done")
            self.disconnect()
            # ax.set_title("")
            # fig.canvas.draw()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas1.draw_idle()
        # for ii in self.mask_ar:
        #     ar = ii
        #     ar[ar == 1] = np.nan
        #     self.ax1.imshow(ar, alpha=0.075, cmap="Reds", origin='lower')

    def closeEvent(self, event):
        self.Pars.ROIPars.mask_ar = self.mask_ar
        self.Pars.ROIPars.reg_nums = self.reg_nums
        if len(self.reg_nums) > 1:
            # self.Pars.ROIPars.reg_nums_all = np.maximum(*self.reg_nums)
            self.Pars.ROIPars.reg_nums_all = np.maximum.reduce([*self.reg_nums])
        else:
            self.Pars.ROIPars.reg_nums_all = self.reg_nums[0]

        if not hasattr(self.ROIdialog, 'commongui'):
            global Pars
            Pars = self.Pars
            QApplication.quit()
            print('module_ROIs_finished(not common)')
        else:
            self.ROIdialog.Pars = self.Pars
            print('module_ROIs_finished(common gui)')
            self.close()
            self.ROIdialog.curveviewer()


if __name__ == '__main__':
    try:
        del app
    except:
        print('noapp')
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    X, Y = np.meshgrid(x, y)
    f = lambda x, y: np.sin(np.sqrt(x ** 2 + y ** 2)) - x/5 - y/5
    if 'Pars' not in globals():
        Pars = Pars_gen()
        Pars.topo2 = f(X, Y)

    app = QApplication(sys.argv)
    main = selectROIs()
    main.show()
    sys.exit(app.exec_())
