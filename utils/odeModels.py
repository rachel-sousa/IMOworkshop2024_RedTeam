# ====================================================================================
# ODE models
# ====================================================================================
import numpy as np
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import seaborn as sns
sns.set(style="white")
sys.path.append("./")
from odeModelClass import ODEModel

# ======================== House Keeping Funs ==========================================
def create_model(modelName,**kwargs):
    funList = {"GDRSModel":GDRSModel,
               "CellCycleModel":CellCycleModel}
    return funList[modelName](**kwargs)

# ======================= Models =======================================
class GDRSModel(ODEModel):
    ''' 
    GDRS Model from evolutionary tumor board
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GDRSModel"
        self.paramDic = {**self.paramDic,
            'r': 0.03,
            'epsilon': 0.01,
            'beta':0,
            'DMax':1,
            'V0': 1,
            'G0':0.1}
        self.stateVars = ['V', 'G']

    # The governing equations
    def ModelEqns(self, t, uVec):
        V, G, D = uVec
        D_hat = D/self.paramDic['DMax']
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['r'] * V - G * D * V 
        dudtVec[1] = - self.paramDic['epsilon'] * D * G + self.paramDic['beta'] * (1-D) * G
        dudtVec[2] = 0
        return (dudtVec)
    
    def RunCellCountToTumourSizeModel(self, popModelSolDf):
        return popModelSolDf['V'].values
    
# -----------------------------------------
class CellCycleModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CellCycleModel"
        self.paramDic = {**self.paramDic,
            'Emax_P': 1.9721, 'EC50_P': 0.15942, 'h_P1': 0.40941,
            'Imax_P': 1, 'IC50_P': 0.041212, 'h_P2': 0.10945,
            'Imax_A': 0.92714, 'IC50_A': 5.3318, 'h_A1': 5.1795,
            'Kmax_A': 2.8866, 'KC50_A': 0.14143, 'h_A2': 5.6103,
            'K_1': 0.050779, 'K_2': 0.17906, 'K_3': 0.39181,
            'K_4': 1.1443, 'K_5': 0.060188, 'K_6': 0.32358,
            'K_7': 2.1526, 'K_8': 0.0010523, 'p': 0.14826,
            'q': 0.45458, 'L': 13727.7261, 'tau': 95.9914,
            'G10': 0.01, 'S0': 0, 'SD0': 0, 'G20': 0, 'G2D0': 0,
            'Dead0': 0, 'A10': 0}
        self.stateVars = ['G1', 'S', 'SD', 'G2', 'G2D', 'Dead', 'A1']
    
    # Dose response curve
    def E_PARP1(self, PARPi):
        return self.paramDic['Emax_P'] * (PARPi**self.paramDic['h_P1']) / (self.paramDic['EC50_P']**self.paramDic['h_P1'] + PARPi**self.paramDic['h_P1'])

    def E_PARP2(self, PARPi):
        return self.paramDic['Imax_P'] * (PARPi**self.paramDic['h_P2']) / (self.paramDic['IC50_P']**self.paramDic['h_P2'] + PARPi**self.paramDic['h_P2'])

    def E_ATR1(self, ATRi):
        return self.paramDic['Imax_A'] * (ATRi**self.paramDic['h_A1']) / (self.paramDic['IC50_A']**self.paramDic['h_A1'] + ATRi**self.paramDic['h_A1'])

    def E_ATR2(self, ATRi):
        return self.paramDic['Kmax_A'] * (ATRi**self.paramDic['h_A2']) / (self.paramDic['KC50_A']**self.paramDic['h_A2'] + ATRi**self.paramDic['h_A2'])
    
    #Model
    def ModelEqns(self, t, uVec):
        print(uVec, "1")
        G1, S, SD, G2, G2D, Dead, A1, D1, D2 = uVec
        print(uVec, "2")
        dudtVec = np.zeros_like(uVec)
        print(dudtVec, "3")
        p = self.paramDic['p']
        q = self.paramDic['q']
        L = self.paramDic['L']
        tau = self.paramDic['tau']
        k_vals = [self.paramDic[f'K_{i}'] for i in range(1, 9)]
        
        # Differential equations
        dudtVec[0] = 2 * k_vals[4] * (1 - (G1 + S + SD + G2 + G2D + Dead) / L) * G2 - k_vals[0] * G1 #dG1_dt
        dudtVec[1] = k_vals[0] * (1 - p * (1 + self.E_PARP1(D1))) * G1 - k_vals[1] * S + k_vals[2] * (1 - self.E_ATR1(D2)) * SD #dS_dt
        dudtVec[2] = k_vals[0] * p * (1 + self.E_PARP1(D1)) * G1 - k_vals[2] * (1 - self.E_ATR1(D2)) * SD - k_vals[5] * SD - k_vals[3] * SD #dSD_dt
        dudtVec[3] = k_vals[1] * (1 - q) * S - k_vals[4] * (1 - (G1 + S + SD + G2 + G2D + Dead) / L) * G2 + k_vals[6] * (1 - self.E_PARP2(D1)) * G2D #dG2_dt
        dudtVec[4] = k_vals[1] * q * S + k_vals[5] * SD - k_vals[6] * (1 - self.E_PARP2(D1)) * G2D - k_vals[7] * G2D - A1 * G2D #dG2D_dt
        dudtVec[5] = k_vals[3] * SD + k_vals[7] * G2D + A1 * G2D #dDead_dt
        dudtVec[6] = (1 / tau) * (self.E_ATR2(D2) - A1) #dA1_dt
        dudtVec[7] = 0 #dD1_dt
        dudtVec[8] = 0 #dD2_dt
        return (dudtVec)
    
    def RunCellCountToTumourSizeModel(self, popModelSolDf):
        return np.sum(popModelSolDf[self.stateVars[:-1]], axis=1)
