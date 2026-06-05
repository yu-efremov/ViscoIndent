# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:45:53 2024

@author: Yuri
"""
import sys, os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton,\
    QComboBox, QVBoxLayout, QLabel
import configparser
import ast


class config_gui(QMainWindow):
    
    def smart_parse(self, value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    def get_script_dir(self):  # path
        if getattr(sys, 'frozen', False):
            self.scriptdir = os.path.dirname(sys.executable)
        else:
            self.scriptdir = os.path.dirname(os.path.abspath(__file__))
        print("Script folder")
        print(self.scriptdir)

    def __init__(self, parent=None):
        super(config_gui, self).__init__(parent)
        self.config_gui = parent  # share variables
        if not hasattr(self.config_gui, 'commongui'):
            self.initUI()

    def initUI(self):
        self.get_script_dir()
        self.config_path = os.path.join(self.scriptdir, 'config.ini')
        if not os.path.exists(self.config_path):
            import VI_config
            config_dict = {
                   name: value 
                   for name, value in vars(VI_config).items() 
                   if not name.startswith('__') 
                   }
            print('no config file')
        else:
            config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
            config.optionxform = str 
            config.read(self.config_path)
            #print (config.items('default'))
            print('config file loaded')
            config_dict = {
            section: {key: self.smart_parse(val) for key, val in config.items(section)}
            for section in config.sections()
            }
            print(config_dict)

        configs_list = list(config_dict.keys())

        self.configs = config_dict
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
            # self.close()
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