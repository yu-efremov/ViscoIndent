# -*- coding: utf-8 -*-
"""
@author: Yuri
to use with common pyQt5 gui
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, \
    QRadioButton, QVBoxLayout, QHBoxLayout, QLineEdit, QButtonGroup, \
    QPushButton, QSpinBox, QCheckBox, QFrame
from PyQt5.QtCore import Qt, QEvent
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
from scipy import ndimage

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

        self.le01 = QLineEdit(self)  # threshold level of height from topography (only higher points are selected)
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
        self.label02 = QLabel('Number of ROIs:')
        layoutA3.addWidget(self.label02)
        #layoutA3.addWidget(self.sp01)
        self.text_reg_nums = QLineEdit(self)
        self.text_reg_nums.setText(str(self.numofselections-1)) # change
        self.text_reg_nums.setEnabled(False)
        layoutA3.addWidget(self.text_reg_nums)
        layoutA.addLayout(layoutA2)
        layoutA.addLayout(layoutA3)
        self.separateROIs_box1 = QCheckBox('separate not-touching ROIs', self)
        self.separateROIs_box1.stateChanged.connect(self.split_ROIs)
        self.label_min_elements = QLabel('Smallest number of pixels in ROI:')
        self.le_min_elements = QLineEdit(self)
        self.le_min_elements.setText(str(3)) # self.Pars.ROIPars.min_elements_in_ROI
        self.FrameA4 = QFrame(self)
        self.FrameA4.setFrameShape(QFrame.StyledPanel)
        layoutA4 = QVBoxLayout(self.FrameA4)
        layoutA4.addWidget(self.separateROIs_box1)
        layoutA4.addWidget(self.label_min_elements)
        layoutA4.addWidget(self.le_min_elements)
        # layoutA.addLayout(layoutA4)
        layoutA.addWidget(self.FrameA4)
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
        self.radio_button_clicked()
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
        # self.radio_button_clicked()
        self.h_level = float(self.le01.text())
        pass

    def radio_button_clicked(self):

        print(self.btngroup1.checkedId())
        self.ax1.clear()
        img1 = self.ax1.imshow(self.topo2, interpolation='nearest', cmap='viridis', origin='lower')
        # self.fig.colorbar(img1)
        self.canvas1.draw()

        self.Pars.ROIPars.ROImode = self.btngroup1.checkedId()
        # self.Pars.ROIPars.num_of_ROIs = self.sp01.value()
        # self.numcells = self.Pars.ROIPars.num_of_ROIs
        self.maxRois = 100 # max Roi to not overload program
        self.h_level = float(self.le01.text())
        self.mask_ar = [None]
        self.reg_nums = [None]

        if self.btngroup1.checkedId() == 0:  # "no selection"
            self.mask_ar[0] = np.full(self.topo2.shape, True)  # np.ones(self.topo.shape, dtype=bool)
            self.reg_nums[0] = 1*np.reshape(self.mask_ar[0], self.topo2.size)
        elif self.btngroup1.checkedId() == 1:  # "select above h-level"
            self.numofselections = 1
            # print(self.h_level)
            self.mask_ar[0] = (self.topo2 > self.h_level)
            self.reg_nums[0] = 1*np.reshape(self.mask_ar[0], self.topo2.size)
            self.canvas1.draw()
            ar = self.mask_ar[0]
            ar[ar == 1] = np.nan
            self.ax1.imshow(ar, alpha=0.3, cmap="Reds", origin='lower')
            self.canvas1.draw()
        elif self.btngroup1.checkedId() in [2, 3]:  # "draw"
            self.ax1.set_title("Select ROIs by LMB, stop selection by RMB")
            self.mask_ar = [None]*self.maxRois
            self.reg_nums = [None]*self.maxRois
            self.lasso = LassoSelector(self.ax1, onselect=self.onselect)

    def onselect(self, verts):
        self.canvas1.installEventFilter(self)
        p = Path(verts)
        # Pixel coordinates
        xc = np.arange(self.topo2.shape[0])
        yc = np.arange(self.topo2.shape[1])
        xv, yv = np.meshgrid(xc, yc)
        pix = np.vstack((xv.flatten(), yv.flatten())).T
        maskC = p.contains_points(pix, radius=0.26)  # check radius
        mask = np.reshape(maskC, self.topo2.shape)
        if self.btngroup1.checkedId() == 3:
            # self.h_level = float(self.le01.text())
            print(self.h_level)
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
        self.text_reg_nums.setText(str(self.numofselections-1)) # change
        if self.numofselections == self.maxRois+1:
            print("Done")
            self.text_reg_nums.setText(str(self.numofselections-1))
            self.disconnect()
            # ax.set_title("")
            # fig.canvas.draw()


    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.RightButton:
                print("RMB pressed")
                self.disconnect()
                self.numcells = self.numofselections-1
                self.Pars.ROIPars.num_of_ROIs = self.numofselections-1
                self.reg_nums = self.reg_nums[0:self.numofselections-1]
                print(self.reg_nums)
                self.ax1.set_title("ROI selection stopped")
                self.text_reg_nums.setText(str(self.numofselections-1))
                return True  # Event handled
        return super().eventFilter(obj, event)


    def split_ROIs(self):
        if self.separateROIs_box1.isChecked():
            print(self.mask_ar[0])
            min_elements = float(self.le_min_elements.text())
            if hasattr(self, 'mask_ar'):
                print("split")
                labeled, num_features = ndimage.label(self.mask_ar[0]) # Label connected components in the mask
                # labeled, num_features = ndimage.label(mask, structure=np.ones((3,3))) # considers orthogonal connectivity
                print(num_features)
            init_mask = self.mask_ar[0]
            self.mask_ar = [None]*num_features
            self.reg_nums = [None]*num_features
            xc = np.arange(self.topo2.shape[0])
            yc = np.arange(self.topo2.shape[1])
            xv, yv = np.meshgrid(xc, yc)
            counter = 1
            for i in range(1, num_features+1):
                region_mask = labeled == i
                mask = np.where(region_mask, init_mask, 0)  # keep original values
                mask = mask.astype(bool)
                print(np.count_nonzero(mask))
                if np.count_nonzero(mask)>min_elements:  # TODO place in GUI
                    self.mask_ar[i-1] = mask
                    self.reg_nums[i-1] = np.reshape(mask*1*[counter], self.topo2.size)
                    # self.ax1.imshow(init_mask, alpha=0.3, cmap="Reds", origin='lower')
                    self.canvas1.draw_idle()
                    self.ax1.contour(xc, yc, np.reshape(self.reg_nums[i-1], self.topo2.shape))
                    labelXcoorF = np.maximum.reduce((i)*mask, 0)
                    labelXcoor = np.mean(xc[(labelXcoorF == (i))])
                    labelYcoorF = np.maximum.reduce((i)*mask, 1)
                    labelYcoor = np.mean(yc[(labelYcoorF == (i))])
                    self.ax1.text(labelXcoor, labelYcoor, str((counter)),
                                  horizontalalignment='center', verticalalignment='center',
                                  fontsize=20)
                    counter = counter+1
            # filtered1 = self.mask_ar([x for x in self.mask_ar if x is not None])
            # filtered2 = self.reg_nums([x for x in self.reg_nums if x is not None])
            self.mask_ar[:] = [x for x in self.mask_ar if x is not None]
            self.reg_nums[:] = [x for x in self.reg_nums if x is not None]
            self.numofselections = len(self.mask_ar) # num_features
            self.Pars.ROIPars.num_of_ROIs = len(self.mask_ar)
            self.text_reg_nums.setText(str(self.numofselections))

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas1.draw_idle()
        # for ii in self.mask_ar:
        #     ar = ii
        #     ar[ar == 1] = np.nan
        #     self.ax1.imshow(ar, alpha=0.075, cmap="Reds", origin='lower')

    def closeEvent(self, event):
        self.Pars.ROIPars.mask_ar = self.mask_ar
        self.Pars.ROIPars.h_level = self.h_level
        # print(self.numofselections)
        # print(self.reg_nums)
        self.reg_nums = self.reg_nums[0:self.numofselections-1]
        # print(self.reg_nums)
        self.Pars.ROIPars.reg_nums = self.reg_nums
        if len(self.reg_nums) > 1:
            #self.Pars.ROIPars.reg_nums_all = np.maximum(*self.reg_nums)
            #self.Pars.ROIPars.reg_nums_all = np.maximum(self.reg_nums)
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
    # plt.plot(Pars.ROIPars.reg_nums_all)
