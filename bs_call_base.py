import numpy as np
import matplotlib as mpl
from scipy import special
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'Arial' # 'Times New Roman' 
from ipywidgets import widgets, interactive

def d1(S0, K, r, q, σ, T):
    return ( np.log(S0/K) + (r - q + σ**2 / 2.0) * T  )/(σ * np.sqrt(T) )
def d2(S0, K, r, q, σ, T):
    return ( np.log(S0/K) + (r - q - σ**2 / 2.0) * T  )/(σ * np.sqrt(T) )
def N(x):
    return special.erfc(-x/np.sqrt(2.0))/2.0
def bs_call_price(S0, K, r, q, σ, T):
    return S0 * np.exp(-q*T) * N(d1(S0, K, r, q, σ, T)) - K * np.exp(-r*T) * N(d2(S0, K, r, q, σ, T))
def payoff(S_T, K):
    return np.maximum(0, S_T - K)

