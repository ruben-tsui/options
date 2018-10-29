import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# For making attractive and informative statistical graph
import networkx as nx
from ipywidgets import widgets, interactive, interact, interactive_output
#mpl.patches.ArrowStyle = 'simple'

#%config InlineBackend.figure_format = 'svg'

#price0 = 100.0
def binomial_grid(n, p, up, price0):
    '''
    Purpose: Creates a binomial with n steps. Parameter p is the probability of "Up"
    price0: initial (beginning) price of security
    up_price: 
    down_price: 
    '''
    up_price = up
    down_price = 1.0/up

    G = nx.DiGraph()
    for i in range(0,n+1):
        for j in range(1,i+2):
            if i<n:
                G.add_edge((i,j),(i+1,j))
                G.add_edge((i,j),(i+1,j+1))
    posG = {} #dictionary with nodes position

    # draw nodes, edges (with arrows)
    for node in G.nodes():
        posG[node]=(node[0],n+2+node[0]-2*node[1])
#    nx.draw_networkx(G, pos=posG,
    nx.draw(G, pos=posG,
            edge_color='r', 
            node_color='blue', node_size=100, 
            arrows=True, arrowsize=8, arrowstyle='-', 
            alpha=0.1, width=0.5,
            with_labels=False
            #font_size=10, font_family="Times New Roman", font_color='k'
           )
    
    # draw one particular 'path'
    p = 0.5 # probability of "up"
    #np.random.seed(12345)

    trajectory = np.random.rand(n) < p
    trajectory = ['Up' if x < p else 'Down' for x in trajectory]
    print(trajectory)
    G2 = nx.DiGraph()  
    G2_prices = {(0,1):price0} # G2 stores nodes with prices - initialize with 1st node
    j=1
    for i in range(0,n):
        if trajectory[i] == 'Up':
            # j unchanged
            G2.add_edge((i,j), (i+1,j))
            G2_prices[(i+1,j)] = G2_prices[(i,j)] * up_price
        else:
            G2.add_edge((i,j), (i+1,j+1))
            G2_prices[(i+1,j+1)] = G2_prices[(i,j)] * down_price
            j += 1

    # draw labels
    G2_prices_labels = {}
    for node in G2_prices:
        G2_prices_labels[node] = "{:.2f}".format(G2_prices[node])
    pos_higher = {}
    y_off = 0.75  # offset on the y axis
    for k, v in posG.items():
        pos_higher[k] = (v[0], v[1]+y_off)
    labels = nx.draw_networkx_labels(G, pos=pos_higher,
                labels=G2_prices_labels, 
                font_size=8, font_family="DejaVu Sans", font_color='b'
                )

    nx.draw(G2, pos=posG,
            edge_color='r', 
            node_color='blue', node_size=100, 
            arrows=True, arrowsize=8, arrowstyle='-|>', 
            alpha=1, width=0.5,
            with_labels=False
           )


#################################################################
'''
step_size = widgets.IntSlider(min=2, max=15, step=1, value=5, description='No. of steps')
prob_up   = widgets.FloatSlider(min=0.1, max=1.0, step=0.05, readout_format='.2f', value=0.5, description='Prob. up')
button = widgets.Button(description="Refresh")
display(button)
def on_button_clicked(b):
    prob_up.value += np.random.rand()*0.0001
button.on_click(on_button_clicked)
interactive(binomial_grid, n = step_size, p = prob_up)
'''

# Define widgets
step_size = widgets.IntSlider(min=2, max=20, step=1, value=5, description='No. of steps')
prob_up   = widgets.FloatSlider(min=0.1, max=1.0, step=0.05, readout_format='.2f', value=0.5, description='Prob. up')

S0 = widgets.FloatSlider(min=50.0, max=500.0, step=10.0, readout_format='.1f', value=100.0, description='$S_0$')

u = widgets.FloatSlider(min=0.1, max=10.0, step=0.1, readout_format='.2f', value=1.25, description='$u$')
d = widgets.FloatSlider(min=0.1, max=10.0, step=0.1, readout_format='.2f', value=1.0/u.value, description='$d$')
def update_d(*args):
    d.value = 1.0/u.value
u.observe(update_d, 'value')


button = widgets.Button(description="Simulate")
ui1 = widgets.HBox([S0, u, d])
ui2 = widgets.HBox([step_size, prob_up, button])
ui  = widgets.VBox([ui1, ui2])
def on_button_clicked(b):
    prob_up.value += np.random.rand()*0.0001
button.on_click(on_button_clicked)
