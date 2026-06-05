# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:19:52 2020

@author: Yuri
to use with common pyQt5 gui
"""


import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QRadioButton, \
    QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QButtonGroup, QPushButton
# from PyQt5.QtCore import pyqtSlot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from Pars_class import Pars_gen
from window_check_zero_level import check_zero_level


class flattenAFMwrap(QMainWindow):

    def __init__(self, parent=None):
        super(flattenAFMwrap, self).__init__(parent)
        self.MAPdialog = parent  # share variables
        if not hasattr(self.MAPdialog, 'commongui'):
            self.initUI()

    def initUI(self):
        if not hasattr(self.MAPdialog, 'commongui'):
            global Pars
            self.Pars = Pars
        else:
            self.Pars = self.MAPdialog.Pars
            # self.commongui = self.MAPdialog.commongui
        self.CheckZeroLevel = check_zero_level(self)
        self.topo = self.Pars.topo
        self.Pars.topo2 = self.Pars.topo  # in case no selection made
        self.PL_strategy = self.Pars.PLpars.strategy
        self.height = 0

        self.fig = fig = Figure()
        self.canvas1 = FigureCanvasQTAgg(fig)
        self.ax1 = fig.add_subplot(111, title="Original image")

        self.label01 = QLabel('Bottom effect correction (BEC) strategy')
        self.btngroup1opts = [QRadioButton("no BEC"),
                              QRadioButton("locate zero height level"),
                              QRadioButton("use thickness value")]
        self.btngroup1 = QButtonGroup()
        layoutA = QVBoxLayout()
        layoutA.addWidget(self.label01)
        for i in range(len(self.btngroup1opts)):
            layoutA.addWidget(self.btngroup1opts[i])
            self.btngroup1.addButton(self.btngroup1opts[i], i)
            self.btngroup1opts[i].clicked.connect(self.radio_button_clicked)
        if hasattr(self.Pars, 'HeightfromZ') and self.Pars.HeightfromZ == 1:
            self.btngroup1opts[1].setChecked(True)
        elif hasattr(self.Pars, 'height') and self.Pars.height > 0:
            self.btngroup1opts[2].setChecked(True)
            self.height = self.Pars.height
        else:
            self.btngroup1opts[0].setChecked(True)
        self.labelThickess = QLabel('Thickness')
        self.le01 = QLineEdit(self)  # thickness
        self.le01.setText(str(self.height))
        self.le01.editingFinished.connect(self.edit_height)
        layoutAB = QHBoxLayout()
        layoutAB.addWidget(self.labelThickess)
        layoutAB.addWidget(self.le01)
        layoutA.addLayout(layoutAB, 1)

        self.label21 = QLabel('Zero height level location strategy (activate by "locate zero height level")')
        self.btngroup2opts = [QRadioButton("plane by 3 points"),
                              QRadioButton("horizontal plane by 1 point"),
                              QRadioButton("line-by-line polynomial subtraction")]
                              # QRadioButton("reseved for another option")]
        self.btngroup2 = QButtonGroup()

        layoutB = QVBoxLayout()
        layoutB.addWidget(self.label21)
        for i in range(len(self.btngroup2opts)):
            layoutB.addWidget(self.btngroup2opts[i])
            self.btngroup2.addButton(self.btngroup2opts[i], i)
            self.btngroup2opts[i].clicked.connect(self.radio_button_clicked)
        self.btngroup2opts[0].setChecked(True)
        self.leMaskh = QLineEdit(self)  # thickness
        self.leMaskh.setText(str(100))
        self.Maskh = float(self.leMaskh.text())
        self.leMaskh.editingFinished.connect(self.edit_Maskh)
        layoutBB = QHBoxLayout()
        self.labelMask = QLabel('Mask above')
        layoutBB.addWidget(self.labelMask)
        layoutBB.addWidget(self.leMaskh)
        layoutB.addLayout(layoutBB, 1)
        # self.label_emptyB = QLabel('')
        # layoutB.addWidget(self.label_emptyB)

        self.Selectpts_btn = QPushButton("Re-select points / Update", self)
        self.Selectpts_btn.clicked.connect(self.Selectpts_btn_clicked)

        self.Continue_btn = QPushButton("Continue", self)
        # self.Continue_btn.clicked.connect(self.Continue_btn_clicked)
        self.Continue_btn.clicked.connect(self.close)

        img1 = self.ax1.imshow(self.topo, interpolation='nearest', cmap='viridis', origin='lower')  # or 'jet'
        fig.colorbar(img1)
        self.setGeometry(300, 300, 1000, 650)
        self.setWindowTitle('Topography correction')

        self.fig2 = fig2 = Figure()
        self.canvas2 = FigureCanvasQTAgg(fig2)
        self.ax2 = self.fig2.add_subplot(111, title="Processed image")
        # self.ax2.imshow(self.Pars.topo, interpolation='nearest',cmap='viridis', origin='lower') #'jet'

        layout3 = QHBoxLayout()
        # layout3.addWidget(self.le1)
        layout3.addWidget(self.canvas1)
        layout3.addWidget(self.canvas2)

        # three commands to apply layout
        widget = QWidget()
        layoutM1 = QHBoxLayout()
        layoutM1.addLayout(layoutA, 1)
        layoutM1.addLayout(layoutB, 1)
        layoutM = QVBoxLayout()
        layoutM.addLayout(layoutM1)
        layoutM.addLayout(layout3, 2)
        layoutM.addWidget(self.Selectpts_btn)
        layoutM.addWidget(self.Continue_btn)
        widget.setLayout(layoutM)
        self.setCentralWidget(widget)
        self.show()
        self.activateWindow()
        self.radio_button_clicked()


    def Selectpts_btn_clicked(self):
        # print(self.le1.text())
        print(self.btngroup1.checkedId())
        self.fig2.clear()
        self.ax2.clear()
        self.ax2.set_title("Processed image")
        self.ax2 = self.fig2.add_subplot(111, title="Processed image")
        # img2 = self.ax2.imshow(self.Pars.topo2, interpolation='nearest', cmap='viridis', origin='lower')
        self.canvas2.draw()
        self._cid = self.fig.canvas.mpl_connect(
            "axes_enter_event", self._start_ginput)
        if hasattr(self.Pars.PLpars, 'PLmode') and self.Pars.PLpars.PLmode == 3:
            img2 = self.ax2.imshow(self.Pars.topo2, interpolation='nearest', cmap='viridis', origin='lower')
            self.fig2.colorbar(img2, ax=self.ax2)
            self.ax2.set_title("Processed image")
            self.canvas2.draw()
        print("test1")
        # self.dialog.initUI(self.le1)
        # self.le1.setText(self.strtemp)
        # self.dialog.show()

    def Continue_btn_clicked(self):
        # pass
        self.closeEvent(1)  # not good - fires twice

    def edit_height(self):
        self.radio_button_clicked()
        
    def edit_Maskh(self):
        self.radio_button_clicked()

    def points_no_pressbutton(self):
        self._cid = self.fig.canvas.mpl_connect(
            "axes_enter_event", self._start_ginput)
        # print('here1')
        # self.dialog.initUI(self.le1)
        # self.le1.setText(self.strtemp)
        # self.dialog.show()
        
    def radio_button_clicked(self):
        for button in  self.btngroup2.buttons():
            button.setEnabled(True)
        print(self.btngroup1.checkedId())
        if self.btngroup1.checkedId() == 0:
            str_fig_add1 = 'no topography correction'
            self.Pars.PLpars.strategy = 0
            self.Pars.HeightfromZ = 0
            for button in  self.btngroup2.buttons():
                button.setEnabled(False)
        elif self.btngroup1.checkedId() == 1:
            self.Pars.HeightfromZ = 1
            self.Pars.PLpars.strategy = 3
            if self.btngroup2.checkedId() == 0:
                str_2 = 'select 3 or more points and press RMB'
                self.Pars.PLpars.PLmode = 1
            elif self.btngroup2.checkedId() == 1:
                str_2 = 'select 1 or more points and press RMB'
                self.Pars.PLpars.PLmode = 2
            elif self.btngroup2.checkedId() == 2:
                str_2 = 'select masks if needed'
                self.Pars.PLpars.PLmode = 3
            else:
                str_2 = ''
            str_fig_add1 = 'find zero level \n' + str_2
            self.points_no_pressbutton()
        elif self.btngroup1.checkedId() == 2:
            str_fig_add1 = 'use thickness value'
            self.Pars.HeightfromZ = 0
            self.Pars.height = float(self.le01.text())
        self.ax1.set_title("Original image \n" + str_fig_add1)
        self.canvas1.draw()
        if hasattr(self.Pars.PLpars, 'PLmode') and self.Pars.PLpars.PLmode == 3:
            print("Test_line_by_line")
            mapsize = self.topo.shape
            numxlines = mapsize[0]
            numylines = mapsize[1]
            porder = 2
            bas = np.ones([mapsize[0], mapsize[1]])
            for i in range(0, numxlines):
               x=np.array(range(numylines))
               xLine=self.topo[i,:]
               # plt.plot(x, xLine)
               nanInds = np.where(np.isnan(xLine))[0]
               pval = np.polyfit(np.delete(x, nanInds), np.delete(xLine, nanInds), porder)
               corrline = np.polyval(pval,x)
               # plt.plot(x,xLine,x, corrline)
               bas[i, :]=corrline
            self.Pars.PL = bas
            self.Pars.topo2 = self.topo - bas
            self.Maskheight = float(self.leMaskh.text())
            if self.Maskheight!=0.001:
                print(self.Maskheight)
                self.topo_temp = self.Pars.topo2
                bas2 = np.ones([mapsize[0], mapsize[1]])
                for i in range(0, numxlines):
                   x=np.array(range(numylines))
                   xLine=self.topo_temp[i,:]
                   # plt.plot(x, xLine)
                   nanInds = np.where(xLine>self.Maskheight)[0]
                   pval = np.polyfit(np.delete(x, nanInds), np.delete(xLine, nanInds), porder)
                   corrline = np.polyval(pval,x)
                   # plt.plot(x,xLine,x, corrline)
                   bas2[i, :]=corrline
                self.Pars.topo2 = self.topo - bas - bas2
                self.Pars.PL = bas + bas2
            self.ax2.clear()
            img2 = self.ax2.imshow(self.Pars.topo2, interpolation='nearest', cmap='viridis', origin='lower')
            self.fig2.colorbar(img2, ax=self.ax2)
            self.ax2.set_title("Processed image")
            self.canvas2.draw()
        self._cid2 = self.fig2.canvas.mpl_connect(
                "axes_enter_event", self._start_ginput2)
        self.Selectpts_btn_clicked()

    def _start_ginput(self, event):
        self.fig.canvas.mpl_disconnect(self._cid)
        # Workaround for now; patch makes this unnecessary.
        mapsize = self.topo.shape
        numxlines = mapsize[0]
        numylines = mapsize[1]
        if hasattr(self.Pars.PLpars, 'PLmode') and self.Pars.PLpars.PLmode in [1,2]:
            self.fig.show = lambda: None
            self.pts = self.fig.ginput(-1, timeout=0, mouse_pop=2, mouse_stop=3)  # 2 is MMB, 3 is RMB
            # print(self.pts)
            pts = np.asarray(self.pts)
            pts = pts.round()  # round pts coordinates
            pts = pts.astype(int)  # coordinates to integer
            pont = pts.shape[0]
            zin = np.ones((pont, 1))
            m = np.concatenate((pts, zin), axis=1)
            for i in range(pont):
                zin[i] = self.topo[pts[i, 1], pts[i, 0]]  # checked
                vx = np.linspace(1, mapsize[0], mapsize[1])  # start, end, number of elements
                vy = np.linspace(1, mapsize[0], mapsize[1])
    
            if self.Pars.PLpars.PLmode == 1:
                v = np.linalg.lstsq(m, zin, rcond=None)[0]
                x, y = np.meshgrid(vx, vy)
                bas = v[0]*x+v[1]*y+v[2]  # checked
                # print('here2')
            elif self.Pars.PLpars.PLmode == 2:
                zin = np.mean(zin)
                bas = zin*np.ones([mapsize[0], mapsize[1]])
            # coorection for deflection in curves
            if hasattr(self.MAPdialog, 'Data'):
                print('DFL correction applied')
                PixN = []
                for ii in range(pont):
                    PixN.append(pts[ii, 0] + (pts[ii, 1])*self.Pars.topo.shape[0])
                print(PixN)
                DFLm = []
                for ii in range(len(PixN)):
                    DFLc = self.MAPdialog.Data[PixN[ii]][1][:, 1]
                    DFLm.append(np.max(DFLc) - np.mean(DFLc[0:len(DFLc)//4])) # max DFL minus zero level
                DFLs = np.mean(DFLm)
                bas = bas + DFLs
        if self.Pars.PLpars.PLmode == 3:  # line-by-line
            porder = 1
            bas = np.ones([mapsize[0], mapsize[1]])
            for i in range(0, numxlines):
               x=np.array(range(numylines))
               xLine=self.topo[i,:]
               # plt.plot(x, xLine)
               nanInds = np.where(np.isnan(xLine))[0]
               pval = np.polyfit(np.delete(x, nanInds), np.delete(xLine, nanInds), porder)
               corrline = np.polyval(pval,x)
               # plt.plot(x,xLine,x, corrline)
               bas[i, :]=corrline

        self.Pars.topo2 = self.topo - bas
        self.Pars.PL = bas
        # self.Pars.PLpars.strategy = PL_strategy
        # self.Pars.PLpars.PLmode = PLmode
        # self.fig2.clear()
        self.ax2.clear()
        # self.ax2 = self.fig2.add_subplot(111, title = "Processed image")
        img2 = self.ax2.imshow(self.Pars.topo2, interpolation='nearest', cmap='viridis', origin='lower')
        self.fig2.colorbar(img2, ax=self.ax2)
        self.ax2.set_title("Processed image")
        self.canvas2.draw()
        # self.show()

    def _start_ginput2(self, event):
        self.fig2.canvas.mpl_disconnect(self._cid2)
        # self.canvas2.draw()
        if hasattr(self.Pars, 'topo2'):
            # Crosshair lines
            self.ax2.clear()
            img2 = self.ax2.imshow(self.Pars.topo2, interpolation='nearest', cmap='viridis', origin='lower')
            # self.fig2.colorbar(img2, ax=self.ax2)
            self.ax2.set_title("Processed image")
            self.canvas2.draw()
            if not hasattr(self, 'vline'):
                self.vline = self.ax2.axvline(0, color='r', linestyle='--')
                self.hline = self.ax2.axhline(0, color='r', linestyle='--')
            else:
                # self.vline.set_xdata([self.xCS])
                # self.hline.set_ydata([self.yCS])
                self.vline = self.ax2.axvline(self.xCS, color='r', linestyle='--')
                self.hline = self.ax2.axhline(self.yCS, color='r', linestyle='--')
                self.canvas2.draw()

            self.pts = self.fig2.ginput(1, timeout=0, mouse_pop=2, mouse_stop=3)  # 2 is MMB, 3 is RMB

            # --- Profiles figure ---
            
            # self.CheckZeroLevel.initUI()
            pts = np.asarray(self.pts)
            pts = pts.round()  # round pts coordinates
            pts = pts.astype(int)  # coordinates to integer
            print(pts)
            if len(pts)>0:
                pts2 = pts.tolist()[0]
                print(pts2)
                self.xCS = pts2[0]
                self.yCS = pts2[1]
                print(f"Clicked at: x={self.xCS}, y={self.yCS}")
    
                # --- Update crosshairs ---
                self.vline.set_xdata([self.xCS])
                self.hline.set_ydata([self.yCS])
                # fig.canvas.draw_idle()
    
                # --- Extract profiles ---
                row_profile = self.Pars.topo2[self.yCS, :]
                col_profile = self.Pars.topo2[:, self.xCS]
                self.row_profile = row_profile
                self.col_profile = col_profile
                self.CheckZeroLevel.initUI()

                self.canvas2.draw()
                self._cid2 = self.fig2.canvas.mpl_connect(
                    "axes_enter_event", self._start_ginput2)  
        
        
    def closeEvent(self, event):
        if not hasattr(self.MAPdialog, 'commongui'):
            print('module BEC exit (not common)')
            print('Pars.Height = ' + str(self.Pars.height))
            QApplication.quit()
        else:
            self.MAPdialog.Pars = self.Pars
            self.close()
            self.MAPdialog.regionselection()
            print('module BEC exit (common)')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    try:
        del app
    except:
        print('noapp')
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    X, Y = np.meshgrid(x, y)
    f = lambda x, y: np.sin(np.sqrt(x ** 2 + y ** 2)) - x/5 - y/5
    topo = f(X, Y)
    if 'Pars' not in globals():
        Pars = Pars_gen()
        Pars.ScanSize = 10
        Pars.topo = topo
        Pars.height = 0
        Pars.HeightfromZ = 1
        Pars.PLpars.strategy = 1  # % 1-ask question, 2 - reuse, 3 - new, not reuse; 0 - no PL substraction
        Pars.PLpars.PLmode = 1
        Pars.PLpars.PLadj=1

    app = QApplication(sys.argv)
    main = flattenAFMwrap()
    main.show()
    sys.exit(app.exec_())