import numpy as np
import matplotlib as mpl
#from scipy import special
from scipy.stats import norm
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'Arial' # 'Times New Roman' 
from ipywidgets import widgets, interactive

def d1(S0, K, r, q, σ, T):
    return ( np.log(S0/K) + (r - q + σ**2 / 2.0) * T  )/(σ * np.sqrt(T) )
def d2(S0, K, r, q, σ, T):
    return ( np.log(S0/K) + (r - q - σ**2 / 2.0) * T  )/(σ * np.sqrt(T) )
def N(x):
    return norm.cdf(x)

def bs_call_price(S0, K, r, q, σ, T):
    return S0 * np.exp(-q*T) * N(d1(S0, K, r, q, σ, T)) - K * np.exp(-r*T) * N(d2(S0, K, r, q, σ, T))
def payoff_call(S_T, K):
    return np.maximum(0, S_T - K)

def bs_put_price(S0, K, r, q, σ, T):
    return -S0 * np.exp(-q*T) * N(-d1(S0, K, r, q, σ, T)) + K * np.exp(-r*T) * N(-d2(S0, K, r, q, σ, T))
def payoff_put(S_T, K):
    return np.maximum(0, K - S_T)

### Delta
def delta(S0, K, r, q, σ, T, category='call'):
    if category=='call':
        return np.exp(-q*T) * N(  d1(S0, K, r, q, σ, T) )
    elif category=='put':
        return -np.exp(-q*T) * N( -d1(S0, K, r, q, σ, T) )
    else:
        return None
    
### Gamma
def gamma(S0, K, r, q, σ, T, category='call'):
    '''
    if category=='call':
        return np.exp(-q*T) * norm.pdf(  d1(x, K, r, q, σ, T) ) / ( S0*σ*np.sqrt(T) )
    elif category=='put':
        return np.exp(-q*T) * norm.pdf(  d1(x, K, r, q, σ, T) ) / ( S0*σ*np.sqrt(T) )
    else:
        return None
    '''
    return np.exp(-q*T) * norm.pdf(  d1(S0, K, r, q, σ, T) ) / ( S0*σ*np.sqrt(T) )

### Vega
def vega(S0, K, r, q, σ, T, category='call'):
    if category=='call':
        return ( S0*np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1(S0, K, r, q, σ, T)) )
    elif category=='put':
        return ( S0*np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1(S0, K, r, q, σ, T)) )
    else:
        return None
 
### Taylor polynomial
#### coefficients
def taylor_parabola(x0, K, r, q, σ, T, category='call'):
    '''
    Find formula (coefficients) of parabola f at x = x0
    f(x) = a x^2 + b x + d
    '''
    Δ = delta(x0, K, r, q, σ, T, category=category)
    Γ = gamma(x0, K, r, q, σ, T, category=category)
    if category=='call':
        price = bs_call_price(x0, K, r, q, σ, T)
    else:
        price = bs_put_price(x0, K, r, q, σ, T)
    a = 0.5 * Γ
    b = Δ - x0 * Γ
    d = price - a * x0*x0 - b * x0
    return (a, b, d) 

S0 = 100
r = 0.025; q = 0.005; T = 1.0
σ = 0.2; K = 100

def plot_func(σ, K, r, q):
    x = np.linspace(20, 180, 51)
    y = bs_call_price(x, K, r, q, σ, T)
    plt.figure()
    plt.plot(x, y, 'r', lw=2.5, label='present value')
    z = payoff_call(x, K)
    plt.plot(x, z, 'b-.', lw=1.5, label='payoff')
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel('index level $S_0$')
    plt.ylabel('price$')
    ####
    plt.show()
             
             
def plot_func2(σ, slope):

    gs = plt.GridSpec(3,4, wspace=0.3, hspace=0.3)
    fig = plt.figure(figsize=(9,12))

    x      = np.linspace(20, 180, 51)
    y_call = bs_call_price(x, K, r, q, σ, T)
    y_put  = bs_put_price(x, K, r, q, σ, T)
    z_call = payoff_call(x, K)
    z_put  = payoff_put(x, K)

    # call parabola
    a, b, d = taylor_parabola(slope, K, r, q, σ, T, category='call')
    print(f'a={a:.2f}, b={b:.2f}, d={d:.2f}')
    taylor2_call = a*x*x + b*x + d
    # put parabola
    a, b, d = taylor_parabola(slope, K, r, q, σ, T, category='put')
    taylor2_put  = a*x*x + b*x + d

    delta_call =  np.exp(-q*T) * N(  d1(x, K, r, q, σ, T) )
    delta_put  = -np.exp(-q*T) * N( -d1(x, K, r, q, σ, T) )
    gamma_call =  gamma(x, K, r, q, σ, T, category='call')
    gamma_put  =  gamma(x, K, r, q, σ, T, category='put')
    vega_put   =  vega(x, K, r, q, σ, T, category='put')
    
    s1 = fig.add_subplot(gs[0,:2]); s1.grid(True)
    s1.set_title('Call', fontsize=12, color='brown')
    s2 = fig.add_subplot(gs[0,2:], sharex=s1, sharey=s1); s2.grid(True)
    s2.set_title('Put', fontsize=12, color='brown')
    s3 = fig.add_subplot(gs[1,:2]); s3.grid(True)
    s3.set_title(r'$\Delta_C$', fontsize=12, color='blue')
    s4 = fig.add_subplot(gs[1,2:], sharey=s3); s4.grid(True)
    s4.set_title(r'$\Delta_P$', fontsize=12, color='blue')
    s5 = fig.add_subplot(gs[2,:2]); s5.grid(True)
    s5.set_title(r'$\Gamma$', fontsize=12, color='blue')
    s6 = fig.add_subplot(gs[2,2:]); s6.grid(True)
    s6.set_title(r'$Vega$', fontsize=12, color='blue')
    #s6 = fig.add_subplot(gs[2,2:])
    
    s1.plot(x, y_call, 'r', lw=2.5, label='present value')
    s1.axvline(x=slope, color="g", alpha=0.75, linewidth=2)
    s1.plot(x, z_call, 'b-.', lw=1.5, label='payoff')
    s1.plot(x, taylor2_call, 'purple', lw=1.5, label='taylor 2nd order')

    s2.plot(x, y_put, 'r', lw=2.5, label='present value')
    s2.axvline(x=slope, color="g", alpha=0.75, linewidth=2)
    s2.plot(x, z_put, 'b-.', lw=1.5, label='payoff')
    s2.plot(x, taylor2_put, 'purple', lw=1.5, label='taylor 2nd order')

    s3.plot(x, delta_call, 'r', lw=2.5, label='delta call')
    s3.axvline(x=slope, color="g", alpha=0.75, linewidth=2)

    s4.plot(x, delta_put, 'r', lw=2.5, label='delta put')
    s4.axvline(x=slope, color="g", alpha=0.75, linewidth=2)

    s5.plot(x, gamma_call, 'orange', lw=2.5, label='gamma call')
    s5.axvline(x=slope, color="g", alpha=0.75, linewidth=2)

    s6.plot(x, vega_put, 'purple', lw=2.5, label='gamma put')
    s6.axvline(x=slope, color="g", alpha=0.75, linewidth=2)
    
    #s3.barh(x2,y2,color='g')
    #s4.plot(x2,y2,'k')
    #s5 = fig.add_subplot(gs[2,1:])
    #s5.plot(x1,y1,'b^',x2,y2,'yo')
    
                              
                              