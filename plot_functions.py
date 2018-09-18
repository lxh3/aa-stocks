import numpy as np
import matplotlib.pyplot as plt


_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x+y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'

def rgb(triplet):
    """ Convert hex to rgb """
    return np.asarray([_HEXDEC[triplet[1:3]], _HEXDEC[triplet[3:5]], _HEXDEC[triplet[5:7]]])


def triplet(rgb, lettercase=LOWERCASE):
    """ Convert rgb to hex """
    return '#'+format(int(rgb[0])<<16 | int(rgb[1])<<8 | int(rgb[2]), '06'+lettercase).upper()


def hex2rbg_ary(array):
    """ Convert hex color array to rgb array """
    new_arry = []
    for color in array:
        new_arry.append(rgb(color))
    return np.asarray(new_arry)


def rgb2hex_ary(array):
    """ Convert rgb color array to hex array"""
    new_arry = []
    for color in array:
        new_arry.append(triplet(color))
    return np.asarray(new_arry)


def define_colors(cs,corner_fit,num_comp):
    """ Determine a weighted mix of colors """
    rgb_cs = hex2rbg_ary(cs)
    w = (corner_fit*num_comp[None,:])/(corner_fit*num_comp[None,:]).sum(axis=1)[:,None]
    cs = rgb2hex_ary(w.dot(rgb_cs))
    return cs


def plot_eve(u, k, numcos, sects, cols, ls='solid', maxylim=0):
    """ Function for plotting W, C, and V figures """
    prev = 0
    for i in np.arange(1,len(sects)+1):
        j = sum(numcos[:i])
        plt.bar(np.arange(prev, j), u[ prev:j, k],  width=1, color= np.asarray([cols[i-1]]*(j-prev)), ec='none' )
        prev = j
    plt.xlim(0,j);
    if maxylim: plt.ylim(-1, 1);
    return

        
def plot_flows(flow_data, ticks, ticker, order=[], plot_ts = 0, xticksTF=True):

    """ 
    Plot company evolutions in time 
    """    

    code = np.argwhere(ticks==ticker)[0][0]
    
    cs = np.asarray(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])

    mycolors = np.asarray(cs)

    yoff = 0.
    periods = np.shape(flow_data)[0]
    bases = np.shape(flow_data)[1]
    
    plt.figure(figsize=((10,6)))
    plt.subplot(211)
    for j in xrange(bases):
        plt.fill_between(np.arange(periods), flow_data[:,j,code] + yoff, yoff, 
            color= mycolors[j], linewidth=0, edgecolor='k')
        yoff = yoff + flow_data[:,j, code]
        
    #title(ticker)
    plt.xlim(0,periods-1)
    if xticksTF: plt.xticks(np.arange(5, periods, 15), ['95', '98', '01', '04', '07', '10', '13'] )
    else: plt.xticks([])
    plt.ylim(0,1)
    plt.yticks([])
    
    if plot_ts:
        plt.subplot(212)
        plt.plot(big[:,code], linewidth=0.3)
        plt.xticks(arange(100, 5003, 250), arange(1994, 2014, 1))
        plt.xlim(0,5003)

    return
