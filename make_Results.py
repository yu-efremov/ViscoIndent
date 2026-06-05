# -*- coding: utf-8 -*-
"""
make pandas Results array for Tingsprocessing code
"""

import numpy as np
import pandas as pd


def make_Results(testlen):
    # testlen = 4096
    Results = pd.DataFrame(columns=['Name', 'Pixel', 'model', 'cpHertz',
                                    'EHertz', 'hysteresis', 'AdjRsqHertz',
                                    'max_force', 'max_adh', 'ind_depth',
                                    'Height', 'cpTing', 'resnorm', 'S',
                                    'AdjRsq', 'contact_timeH', 'EHertzBEC',
                                    'E0', 'Einf', 'alpha_tau', 'contact_timeT',
                                    'Freq', 'Estor', 'Eloss', 'E0BEC',
                                    'EinfBEC', 'alpha_tauBEC', 'E_adhesion', 'adhesion',
                                    'comment'], index=range(testlen))
    Results = Results.astype({'Name': str,
                              'Pixel': object,
                              'model': str,
                              'cpHertz': np.float64,
                              'EHertz': np.float64,
                              'hysteresis': np.float64,
                              'AdjRsqHertz': np.float64,
                              'max_force': np.float64,
                              'max_adh': np.float64,
                              'ind_depth': np.float64,
                              'Height': np.float64,
                              'cpTing': np.float64,
                              'resnorm': np.float64,
                              'AdjRsq': np.float64,
                              'S': np.float64,
                              'contact_timeH': np.float64,
                              'EHertzBEC': np.float64,
                              'E0': np.float64,
                              'Einf': np.float64,
                              'alpha_tau': np.float64,
                              'contact_timeT': np.float64,
                              'Freq': np.float64,
                              'Estor': np.float64,
                              'Eloss': np.float64,
                              'E0BEC': np.float64,
                              'EinfBEC': np.float64,
                              'alpha_tauBEC': np.float64,
                              'E_adhesion': np.float64,
                              'adhesion': np.float64,
                              'comment': str})
    return Results


if __name__ == '__main__':  #
    Resultstest = make_Results(10)
