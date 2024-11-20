# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:45:53 2024

@author: Yuri
"""
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton,\
    QComboBox, QVBoxLayout, QLabel

class config_gui(QMainWindow):
    
    def __init__(self, parent=None):
        super(config_gui, self).__init__(parent)
        self.config_gui = parent  # share variables
        if not hasattr(self.config_gui, 'commongui'):
            self.initUI()


    def initUI(self):
        import VI_config
        self.configs = {
               name: value 
               for name, value in vars(VI_config).items() 
               if not name.startswith('__') 
            }
        configs_list = [
               name 
               for name, value in vars(VI_config).items() 
               if not name.startswith('__') 
            ]
        self.configs_names = configs_list
        self.cb_configs_names = QComboBox()
        self.cb_configs_names.addItems(self.configs_names)
        self.OK_btn = QPushButton("Accept", self)
        # self.OK_btn.clicked.connect(self.close)
        self.OK_btn.clicked.connect(self.continuescript)
        self.label1 = QLabel('Select basic settings for the data processing')
        layoutA = QVBoxLayout()
        layoutA.addWidget(self.label1)
        layoutA.addWidget(self.cb_configs_names)
        layoutA.addWidget(self.OK_btn)
        layoutM = QVBoxLayout()
        layoutM.addLayout(layoutA)
        
        widget = QWidget()
        widget.setLayout(layoutM)
        self.setWindowTitle('ViscoIndent settings')
        self.setCentralWidget(widget)
        self.show()
        self.activateWindow()
        QApplication.alert(self, 5000)

    def closeEvent(self, event):
        self.selected_config = str(self.cb_configs_names.currentText())
        if not hasattr(self.config_gui, 'commongui'):
            # print('here')
            if hasattr(self, 'Pars'):
                print('here2')
            print('module exit')
            QApplication.quit()
        else:
            self.config_gui.selected_config_name = self.selected_config
            self.config_gui.selected_config = self.configs[self.selected_config]
            # self.close()
            self.config_gui.selection_win1.initUI()
            print('module_config_gui_finished')
    
    def continuescript(self):
        self.selected_config = str(self.cb_configs_names.currentText())
        if not hasattr(self.config_gui, 'commongui'):
            # print('here')
            if hasattr(self, 'Pars'):
                print('here2')
            print('module exit')
            QApplication.quit()
        else:
            self.config_gui.selected_config_name = self.selected_config
            self.config_gui.selected_config = self.configs[self.selected_config]
            self.hide()
            self.config_gui.selection_win1.initUI()
            print('module_config_gui_finished')



if __name__ == '__main__':
    # from config_gui import config_gui
    try:
        del app
    except:
        print('noapp')

    app = QApplication(sys.argv)
    main = config_gui()
    main.show()
    sys.exit(app.exec_())