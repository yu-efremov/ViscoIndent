# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:26:53 2026

@author: Yuri
"""

import sys
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget,\
                            QVBoxLayout, QHBoxLayout
# from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class check_zero_level(QMainWindow):

    def __init__(self, parent=None):
        super(check_zero_level, self).__init__(parent)
        self.CheckZeroLevel = parent  # share variables
        if not hasattr(self.CheckZeroLevel, 'commongui'):
            print("no main GUI")
        if not hasattr(self.CheckZeroLevel, 'Pars'):
            print("no main GUI")
            self.initUI()

    def initUI(self):
        #self.Pars = self.CheckZeroLevel.Pars


        self.fig = fig = Figure()
        self.canvas1 = FigureCanvasQTAgg(fig)
        self.ax1 = fig.add_subplot(121) #, title="Profiles")
        self.ax2 = fig.add_subplot(122) #, title="Profiles")
        if hasattr(self.CheckZeroLevel, 'row_profile'):
            row_profile = self.CheckZeroLevel.row_profile
            col_profile = self.CheckZeroLevel.col_profile
        else:
            row_profile = [1, 2, 3]
            col_profile = [1, 2, 3]
            print("no profiles found")
        x1 = np.arange(len(row_profile))
        self.ax1.plot(x1, row_profile, lw=2)
        self.ax1.plot(x1, np.zeros(np.size(row_profile)), 'r--')

        x2 = np.arange(len(col_profile))
        self.ax2.plot(x2, col_profile, lw=2)
        self.ax2.plot(x2, np.zeros(np.size(col_profile)), 'r--')

        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('Profiles')
        self.ax1.set_title("Row profile")
        self.ax1.set_xlabel("X position")
        self.ax1.set_ylabel("Value")

        self.ax2.set_title("Column profile")
        self.ax2.set_xlabel("Y position")
        self.ax2.set_ylabel("Value")
    
        self.fig.tight_layout()
        self.toolbar = NavigationToolbar(self.canvas1, self)

        layoutA = QVBoxLayout()
        # layoutA.addWidget(self.toolbar)
        layoutA.addWidget(self.canvas1)


        layoutM = QHBoxLayout()
        layoutM.addLayout(layoutA)

        widget = QWidget()
        widget.setLayout(layoutM)
        self.setCentralWidget(widget)
        self.show()
        self.activateWindow()

    def closeEvent(self, event):

        if not hasattr(self.CheckZeroLevel, 'Pars'):
            # global Pars = self.Pars
            print('module exit')
            QApplication.quit()
        else:
            # self.CheckZeroLevel.Pars = self.Pars
            print('module_CheckZeroLevel_finished')
            self.close()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    try:
        del app
    except:
        print('noapp')
    app = QApplication(sys.argv)
    main = check_zero_level()
    main.show()
    sys.exit(app.exec_())