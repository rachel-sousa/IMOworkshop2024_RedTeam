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
               "CellCycleModel":CellCycleModel,
               "ImmuneModel":ImmuneModel}
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
        #print(uVec, "1")
        G1, S, SD, G2, G2D, Dead, A1, D1, D2 = uVec
        #print(uVec, "2")
        dudtVec = np.zeros_like(uVec)
        #print(dudtVec, "3")
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

#--------------------
class ImmuneModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ImmuneModel"
        self.paramDic = {**self.paramDic,
            #across eq
            'ic50': 0.023, #micromol/L
            'd_1': 1.1e-7, #1/cell/day
            'delta_2': 0.5, #TODO
            'd_2': 1.1e-7, #1/cell/day
            'beta_3': 1,
            #dC/dt
            'r_C': 0.145, #1/day
            'beta_1': 0.9,
            'K': 1e10, #cell
            'delta_1': 0.5, #TODO
            'beta_2': 0.85,
            #dM_A/dt
            'alpha_A': 0.7e7, #cell/day
            'q_A': 1e10, #cell
            'd_A': 0.25, #1/day
            #dM_P/dt
            'alpha_P': 0.7e7, #cell/day
            'q_P': 1e10, #cell
            'beta_4': 0.8,
            'd_P': 0.25, #1/day
            #dT/dt
            'r_T': 0.1, #TODO maybe wrong cell/day
            'd_T': 2e-2, #1/day
            #initial
            'C0': 100,
            'M_A0': 10,
            'M_P0': 10,
            'T0': 10}
        self.stateVars = ['C', 'M_A', 'M_P', 'T']

    def ModelEqns(self, t, uVec):
        params = self.paramDic
        #uVec[0] = max(uVec[0], 0)
        C, M_A, M_P, T, D = uVec
        dudtVec = np.zeros_like(uVec)
        drug = D/(params['ic50']+D)
        omega = ((params["d_1"]*C*T)/(1+(params["delta_2"]*M_P))) + (params["d_2"]*C*M_A)*(1+(params['beta_1']*drug))
        dudtVec[0] = (params['r_C']*(1-(params['beta_1']*drug))*C*abs(1-(C/(params['K']+(params['delta_1']*M_P))))) - ((params['d_1']*C*T)/(1+(params['delta_2']*M_P))) - ((params['d_2']*C*M_A)*(1+(params['beta_1']*drug)))
        dudtVec[1] = (params['alpha_A']*(C/(params['q_A']+C))*(1+(params['beta_3']*drug))) - (params['d_A']*M_A)
        dudtVec[2] = (params['alpha_P']*(C/(params['q_P']+C))*(1-(params['beta_3']*drug))) - (params['d_P']*M_P)
        dudtVec[3] = (params['r_T']*T*M_A*omega)/(1+T) - (params['d_T']*T*(1-(params["beta_4"]*drug)))
        dudtVec[4] = 0
        #TODO temp debugging print statements
        # print("C", C, (params['r_C']*(1-(params['beta_1']*drug))*C*abs(1-(C/(params['K']+(params['delta_1']*M_P))))), ((params['d_1']*C*T)/(1+(params['delta_2']*M_P))), ((params['d_2']*C*M_A)*(1+(params['beta_1']*drug))))
        # print("M_A", M_A, (params['alpha_A']*(C/(params['q_A']+C))*(1+(params['beta_3']*drug))), (params['d_A']*M_A))
        # print("M_P", M_P, (params['alpha_P']*(C/(params['q_P']+C))*(1-(params['beta_3']*drug))), (params['d_P']*M_P))
        # print("T", T, (params['r_T']*T*M_A*omega), (params['d_T']*T*(1-(params["beta_4"]*drug))))
        # print(T, M_A, omega)
        # print()
        return (dudtVec)
