# -*- coding: utf-8 -*-
"""
file with basic cellsuration settings


"""
from Pars_class import Pars_gen
import types

config = {}
# start folder
config['start_folder'] = ''  # write data containing folder here
config['supress_ROIquestion'] = 0  # 0 - ask question, 1 - response no, 2 - yes
config['HeightfromZ'] = 0
config['ROImode'] = 0    # 0 - "no selection", 1 - "select above h-level",
                        # 2 - "draw", 3- "draw and select above h-level"
config['h_level'] = -10000  # nm, low value to select all points
config['save_short'] = 0


cells = {}
# start folder
cells['start_folder'] = ''  # write data containing folder here
cells['supress_ROIquestion'] = 2  # 0 - ask question, 1 - response no, 2 - yes
cells['HeightfromZ'] = 1
cells['ROImode'] = 3    # 0 - "no selection", 1 - "select above h-level",
                        # 2 - "draw", 3- "draw and select above h-level"
cells['h_level'] = 200  # nm, select points above substarte level
cells['save_short'] = 1

biomomentum = {}
biomomentum['start_folder'] = 'D:/MailCloud/BioMomentum'  # write data containing folder here
biomomentum['supress_ROIquestion'] = 3  # 0 - ask question, 1 - response no, 2 - yes, 3 - no flatten
biomomentum['HeightfromZ'] = 1
biomomentum['ROImode'] = 0    # 0 - "no selection", 1 - "select above h-level",
                        # 2 - "draw", 3- "draw and select above h-level"
biomomentum['h_level'] = 0  # nm, select points above substarte level
biomomentum['save_short'] = 1

load_config = {}
load_config['supress_ROIquestion'] = 1  # 0 - ask question, 1 - response no, 2 - yes

cells_YE = {}
# start folder
cells_YE['start_folder'] = ''  # write data containing folder here
cells_YE['supress_ROIquestion'] = 2  # 0 - ask question, 1 - response no, 2 - yes
cells_YE['HeightfromZ'] = 1
cells_YE['ROImode'] = 1    # 0 - "no selection", 1 - "select above h-level",
                        # 2 - "draw", 3- "draw and select above h-level"
cells_YE['h_level'] = 100  # nm, select points above substarte level
cells_YE['save_short'] = 1
# cells_YE['ReplacePars'] = types.SimpleNamespace()
# cells_YE['ReplacePars'].hydro.speedcoef = 5.0e-7