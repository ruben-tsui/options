from pylab import plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'
import warnings; warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import widgets, interactive, interact, interactive_output

# Vectorization with NumPy

def simulate(S0, u, d, p, T, N):
    '''
    S0 = 100  # initial price
    u  = 1.1  # "up" factor
    d  = 0.9 # "down" factor
    p  = 0.4  # probability of "up"
    T  = 30   # time-step size
    N  = 50000 # sample size (no. of simulations)    
    '''
    # Simulating I paths with M time steps
    S = np.zeros((T + 1, N))
    S[0] = S0
    for t in range(1, T + 1):
        z = np.random.rand(N)  # pseudorandom numbers
        S[t] = S[t - 1] * ( (z<p)*u + (z>p)*d )
          # vectorized operation per time step over all paths
    return S
    
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter #, MultipleLocator, FormatStrFormatter, AutoMinorLocator

def animate(S0, u, d, p, T, N, P=10):
    '''
    S: data
    NumSims: simulation size
    numPaths: no. of simulated paths shown
    '''
    S = simulate(S0, u, d, p, T, N)
    fig, mainplot = plt.subplots(figsize=(10, 5))
    mainplot.plot(S[:, :P])
    plt.grid(True)
    plt.xlabel('time step')
    plt.ylabel('price')
    divider = make_axes_locatable(mainplot)
    axHist = divider.append_axes("right", 2.5, pad=0.1, sharey=mainplot)
    axHist.hist(S[-1, :N], bins=15, orientation='horizontal', normed=True)
    axHist.yaxis.set_ticks_position("right")
    axHist.xaxis.set_major_formatter(FuncFormatter('{0:.1%}'.format))
    plt.grid(True)
    plt.xlabel('probability')
    plt.show()

    
S0=widgets.FloatSlider(min=100, max=500, step=100,  value=100,  description="$S_0$")
u=widgets.FloatSlider(min=1.0,  max=2.0, step=0.01, value=1.05, description="u")
d=widgets.FloatSlider(min=0.1,  max=1.0, step=0.1,  value=0.95, description="d")
p=widgets.FloatSlider(min=0.0,  max=1.0, step=0.1,  value=0.5,  description="up prob")
T=widgets.IntSlider(min=5,    max=100,   step=5,    value=25,   description='Time steps')
N=widgets.IntSlider(min=10000, max=100000, step=10000, value=5000, description='Sim. size')
P=widgets.IntSlider(min=10,   max=100,   step=10,   value=40,   description='Paths displayed')
ui1 = widgets.HBox([S0, u, p, d])
ui2 = widgets.HBox([T, N, P])
ui  = widgets.VBox([ui1, ui2])

out = interactive_output(animate, {'S0': S0, 'u': u, 'd': d, 'p': p, 'T': T, 'N': N, 'P': P})
display(ui, out)
