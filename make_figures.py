import itertools
import matplotlib
import gzip
import os

import pandas as pd
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

from scipy.linalg import svd
from StringIO import StringIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
   
from fit_functions import local_load, archanalysis_fixed_C, dolinearregression, center_normalize, center_normalize_removemm
from plot_functions import define_colors, plot_eve, plot_flows


def single_tetrahedron():

    """ 
    Make figure of single tetrahedron along 1st two singular vectors 
    colored by sector association 
    """

    # Load Fits
    filename = 'saves/8_factor_matrices.pkl.gz'
    E,W,C,SSE,varexpl = local_load(filename)

    # Calculate Singular Vectors
    A = np.dot(E, W)
    u,s,vt = svd(A.T, full_matrices=0)
    vtE = np.asarray(np.dot(vt, E))

    # Define colors and assign to data points     
    csu = np.array(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])
    csuarray = csu[np.argmax(W, 0)]

    # Initialize Plot
    i = 0
    j = 1
    matplotlib.rcParams['ytick.color']='#636363'
    matplotlib.rcParams['xtick.color']='#636363'
    matplotlib.rcParams['axes.linewidth']= 0
    output = StringIO()
    plt.ioff()

    frameon = False

    # Plot stock data and corner points
    xs, ys = u[:,i].T*s[i], u[:,j].T*s[j]
    plt.scatter(xs, ys, s=5, color=csuarray) 
    plt.scatter(vtE[i,:], vtE[j,:], s=10, color='k')

    # Plot tetrathedron edges
    xy = zip(vtE[i,:], vtE[j,:])
    for p in itertools.combinations(xy,2):
        plt.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], ls='-', color='#D9D9D9');

    # Save figure
    plt.savefig('figures/single_tetrahedron.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def tetrahedron_array():

    """ 
    Make figure of tetrahedra along 
    different combinations of singular vectors 
    colored by sector association 
    """

    # Load Fits
    filename = 'saves/8_factor_matrices.pkl.gz'
    E,W,C,SSE,varexpl = local_load(filename)

    # Calculate Singular Vectors
    A = np.dot(E, W)
    u,s,vt = svd(A.T, full_matrices=0)
    vtE = np.asarray(np.dot(vt, E))

    # Define colors and assign to data points     
    csu = np.array(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])
    csuarray = csu[np.argmax(W, 0)]

    # Plot
    fig, axs = plt.subplots(8,8,figsize=(20,20))

    for ((j,i),ax) in np.ndenumerate(axs):
        if i < j:
            xs, ys = u[:,i].T*s[i], u[:,j].T*s[j]
            ax.scatter(xs, ys, s=5, color=csuarray) 
            ax.scatter(vtE[i,:], vtE[j,:], s=10, color='k')
            ax.set_axis_off()

            xy = zip(vtE[i,:], vtE[j,:])
            for p in itertools.combinations(xy,2):
                ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], ls='--', color='#D9D9D9');
                
        else:
            plt.delaxes(ax)
               
    plt.subplots_adjust(hspace=0.05,wspace=0.05)
    plt.savefig('figures/tetrahedron_array.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def single_tetrahedron_st():
    
    """ 
    Make figure of single tetrahedron along 
    1st two singular vectors 
    colored by scottrade classification 
    """

    # Load Fits
    filename = 'saves/8_factor_matrices.pkl.gz'
    E,W,C,SSE,varexpl = local_load(filename)

    # Calculate Singular Vectors
    A = np.dot(E, W)
    u,s,vt = svd(A.T, full_matrices=0)
    vtE = np.asarray(np.dot(vt, E))

    #Define colors and labels

    cols = ['#FFF000', '#FDBF6F', '#A6CEE3', '#33A02C', '#FB9A99', '#B2DF8A', '#E31A1C', 
            '#6A3D9A', '#FF7F00', '#543005', '#CAB2D6', '#000000', '#1F78B4',  '#8C510A']
    sects = ['basic', 'capital', 'cyclical', 'energy',    'fin',    'health', 'noncyc',   
             'tech',   'telecom', 'utils', 'miscservices', 'realestate', 'retail', 'transport']
    numcos = [58, 61, 41, 42, 107, 53, 40, 93, 6, 57, 55, 31, 46, 15]

    sulis = [[v]*k for k, v in zip(numcos, cols)]
    csuarray = np.array([item for sublist in sulis for item in sublist])

    sulis_sects = [[v]*k for k, v in zip(numcos, sects)]
    sectsarray = np.array([item for sublist in sulis_sects for item in sublist])

    # Plot figure
    matplotlib.rcParams['ytick.color']='#636363'
    matplotlib.rcParams['xtick.color']='#636363'
    matplotlib.rcParams['axes.linewidth']= 0
    output = StringIO()
    plt.ioff()

    frameon = False

    i = 0
    j = 1
    xs, ys = u[:,i].T*s[i], u[:,j].T*s[j]
    plt.scatter(xs, ys, s=5, color=csuarray) 
    plt.savefig('figures/single_tetrahedron_st.png', bbox_inches='tight', pad_inches=0)
    plt.clf()

    # Make a legend
    recs = []
    for i in range(0,len(cols)):
        recs.append(mpatches.Rectangle([0,0],1,1,fc=cols[i]))
    plt.legend(recs,sects)
    plt.axis('off')
    plt.savefig('figures/single_tetrahedron_st_legend.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def tetrahedron_array_st():

    """ 
    Make figure of tetrahedra along 
    different combinations of singular vectors 
    colored by scottrade classification 
    """

    # Load Fits
    filename = 'saves/8_factor_matrices.pkl.gz'
    E,W,C,SSE,varexpl = local_load(filename)

    # Calculate Singular Vectors
    A = np.dot(E, W)
    u,s,vt = svd(A.T, full_matrices=0)
    vtE = np.asarray(np.dot(vt, E))

    # Define colors and labels
    cols = ['#FFF000', '#FDBF6F', '#A6CEE3', '#33A02C', '#FB9A99', '#B2DF8A', '#E31A1C',
            '#6A3D9A', '#FF7F00', '#543005', '#CAB2D6', '#000000', '#1F78B4',  '#8C510A']
    numcos = [58, 61, 41, 42, 107, 53, 40, 93, 6, 57, 55, 31, 46, 15]

    sulis = [[v]*k for k, v in zip(numcos, cols)]
    csuarray = np.array([item for sublist in sulis for item in sublist])

    # Plot figure
    fig, axs = plt.subplots(8,8,figsize=(20,20))

    for ((j,i),ax) in np.ndenumerate(axs):
        if i < j:
            xs, ys = u[:,i].T*s[i], u[:,j].T*s[j]
            ax.scatter(xs, ys, s=5, color=csuarray) 
            ax.scatter(vtE[i,:], vtE[j,:], s=10, color='k')
            ax.set_axis_off()

            xy = zip(vtE[i,:], vtE[j,:])
            for p in itertools.combinations(xy,2):
                ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], ls='--', color='#D9D9D9');
                
        else:
            plt.delaxes(ax)
               
    plt.subplots_adjust(hspace=0.05,wspace=0.05)
    plt.savefig('figures/tetrahedron_array_st.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def company_pies():

    """ Plot companies as pies colored by their exposure to each of the eight sectors"""
    
    # Define colors, define labels and load data
    cs= np.asarray(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])
    companies = np.asarray(['Berry Petroleum', 'Corning','IBM', 'PG&E','Plum Creek Timber','Wal-Mart']).reshape((2,3))
    company_tickers = np.asarray(['BRY','GLW','IBM','PCG','PCL','WMT']).reshape((2,3))
    E,W,C,SSE,varexpl = local_load('saves/8_factor_matrices.pkl.gz')
    X = pd.read_pickle('data/data.pkl')

    # Plot
    plt.figure(figsize=(9,6))
    plt.rcParams['patch.linewidth'] = 0  
    the_grid = gridspec.GridSpec(2,3)

    for i in xrange(2):
        for j in xrange(3):
            plt.subplot(the_grid[i,j], aspect=1)
            index = np.argwhere(X.columns.values==company_tickers[i,j])[0][0]
            plt.pie(W.T[index], colors=cs, shadow=True,explode=[0.1]*8)
            plt.title(companies[i,j],fontsize=16)

    plt.savefig('figures/company_pies.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def sector_pies():

    """ 
    Plot sectors as pies colored by their
    relation to a sector decomposition with 
    a different number of archetypes
    """

    # Load in fit data
    E_list = []
    weights = []
    for i in xrange(2,10):
        E,W,C,SSE,varexpl = local_load('saves/'+str(i)+'_factor_matrices.pkl.gz')
        E_list.append(E)
        if i<9:
            weights.append(W)
        if i==8:
            C8 = C

    E2 = E_list[0]
    E3 = E_list[1]
    E8 = E_list[-2]

    # Determine relationship between decompositions with varying numbers of sectors
    E,W,SSE,varexpl = archanalysis_fixed_C(E2, E8, noc=8, i_0=0);
    W28 = W
    E,W,SSE,varexpl = archanalysis_fixed_C(E3, E8, noc=8, i_0=0);
    W38 = W
    E,W,SSE,varexpl = archanalysis_fixed_C(E8, E2, noc=2, i_0=0);
    W82 = W
    E,W,SSE,varexpl = archanalysis_fixed_C(E8, E3, noc=3, i_0=0);
    W83 = W

    # Determine colors 
    w_list = []
    for j in xrange(len(E_list)-2):
        XC,S,SSE,varexpl = archanalysis_fixed_C(E_list[j], E_list[j+1], noc=j+3, i_0=0)
        W = S
        w = (W/W.sum(axis=0)[None,:]).T #normalized
        w_list.append(w)
    
    num_comps = []
    for i in xrange(len(weights)):
        temp_arry = []
        for j in xrange(i+2):
            temp_arry.append(len(np.where(np.argmax(weights[i],axis=0)==j)[0]))
        num_comps.append(np.asarray(temp_arry))

    cs_list = []
    cs= np.asarray(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])
    cs_list.append(cs)
    for i in xrange(1,7):
        cs = define_colors(cs,w_list[-i],num_comps[-i])
        cs_list.append(cs)
    cs_list = cs_list[::-1]

    # Plot pies

    noc=2
    plt.figure(figsize=(6,3))
    the_grid = gridspec.GridSpec(1,noc)
    sectors2 = ['c-assets','c-goods']

    for j in xrange(noc):
        plt.subplot(the_grid[0,j], aspect=1)
        plt.pie(W28.T[j], colors=cs_list[-1], shadow=True)
        plt.title(sectors2[j],fontsize=20)

    plt.savefig('figures/W28.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    noc=3
    plt.figure(figsize=(9,3))
    the_grid = gridspec.GridSpec(1,noc)
    sectors3 = ['c-staples','c-assets','c-tech']

    for j in xrange(noc):
        plt.subplot(the_grid[0,j], aspect=1)
        plt.pie(W38.T[j], colors=cs_list[-1], shadow=True)
        plt.title(sectors3[j],fontsize=20)
    
    plt.savefig('figures/W38.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    sectors = np.asarray(['c-industrial','c-energy','c-real estate','c-cyclical','c-non-cyclical','c-tech','c-utility','c-financial'])

    noc=8
    plt.figure(figsize=(24,3))
    the_grid = gridspec.GridSpec(1,noc)

    for j in xrange(noc):
        plt.subplot(the_grid[0,j], aspect=1)
        plt.pie(W82.T[j], colors=cs_list[0], shadow=True)
        plt.title(sectors[j],fontsize=20)

    plt.savefig('figures/W82.png', bbox_inches='tight', pad_inches=0)    
    plt.close()

    noc=8
    plt.figure(figsize=(24,3))
    the_grid = gridspec.GridSpec(1,noc)

    for j in xrange(noc):
        plt.subplot(the_grid[0,j], aspect=1)
        plt.pie(W83.T[j], colors=cs_list[1], shadow=True)
        plt.title(sectors[j],fontsize=20)

    plt.savefig('figures/W83.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def cumulative_log_returns():

    """ Create figure showing the cumulative log returns of each sector """

    E,W,C,SSE,varexpl = local_load('saves/8_factor_matrices.pkl.gz')
    e = np.exp(E)
    x = np.zeros([5001,8])
    x[1:,:] = E
    x[0,:] = [1.]*8
    p = np.cumsum(x, axis=0)

    indxs= (p /(250.0**.5))
    order=[3, 1, 7, 0, 4, 2, 5, 6]

    indxs=indxs[:, order]
    cs = np.asarray(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])
    cs = cs[order]

    fig, axs = plt.subplots(2,4, figsize=(9.6,6))  

    for i in xrange(8):

        plt.rcParams['ytick.color']='#636363'
        plt.rcParams['xtick.color']='#636363'
        plt.rcParams['axes.linewidth']= 0
        output = StringIO()
        plt.ioff()
        frameon = False

        ax = plt.subplot(2,4,i+1)
        plt.plot(indxs[:,i], color=cs[i])

        ax.tick_params(axis='y', direction='out')
        ax.yaxis.tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.xaxis.tick_bottom()

        plt.xlim(0,5001)
        if i !=0 and i!=4: 
            plt.yticks([-2,0,2,4],[])
        else: plt.yticks([-2,0,2,4],[-2,0,2,4])
        
        if i<4:
            plt.xticks([343,1853, 3359, 4869],[])
        else: plt.xticks([343, 1853, 3359, 4869], ['95', '01', '07', '13'])

    
    plt.subplots_adjust(hspace=0.15,wspace=0.35)
    plt.savefig('figures/cumulative_log_returns.png',bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def unweighted_price_index():

    """ Create figure showing the unweighted price index of each sector """

    E,W,C,SSE,varexpl = local_load('saves/8_factor_matrices.pkl.gz')

    X_raw = np.load('data/raw_data.npy')
    indxs = np.dot(X_raw, C)
    indxs = indxs/indxs[0,:]

    cs = np.asarray(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])

    order=[3, 1, 7, 0, 4, 2, 5, 6]

    indxs = indxs[:, order]
    cs = cs[order]

    fig, axs = plt.subplots(1,8, figsize=(12,2.5))  

    for i in xrange(8):

        plt.rcParams['ytick.color']='#636363'
        plt.rcParams['xtick.color']='#636363'
        plt.rcParams['axes.linewidth']= 0
        output = StringIO()
        plt.ioff()
        frameon = False

        ax = plt.subplot(1,8,i+1)
        plt.plot(indxs[:,i], color=cs[i])

        ax.tick_params(axis='y', direction='out')
        ax.yaxis.tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.xaxis.tick_bottom()

        plt.ylim(0,10)
        plt.xlim(0,5001)
        if i !=0: 
            plt.yticks([0,5,10],[])
        else: plt.yticks([0,5,10],[0,5,10])
        
        if i<0:
            plt.xticks([343,1853, 3359, 4869],[])
        else: 
            plt.xticks([343, 1853, 3359, 4869], ['95', '01', '07', '13'])
    
    plt.subplots_adjust(hspace=0.15,wspace=0.35)
    plt.savefig('figures/unweighted_price_index.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def normalized_log_returns():

    """ Create figure showing the normalized log returns of each sector """

    E,W,C,SSE,varexpl = local_load('saves/8_factor_matrices.pkl.gz')

    indxs = E
    cs = np.asarray(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])

    order=[3, 1, 7, 0, 4, 2, 5, 6]

    indxs=indxs[:, order]
    cs= cs[order]

    fig, axs = plt.subplots(1,8, figsize=(12,2.5))  

    for i in xrange(8):

        plt.rcParams['ytick.color']='#636363'
        plt.rcParams['xtick.color']='#636363'
        plt.rcParams['axes.linewidth']= 0
        output = StringIO()
        plt.ioff()
        frameon = False

        ax = plt.subplot(1,8,i+1)
        plt.plot(indxs[:,i], color=cs[i])

        ax.tick_params(axis='y', direction='out')
        ax.yaxis.tick_left()
        ax.tick_params(axis='x', direction='out')
        ax.xaxis.tick_bottom()

        plt.ylim(-6,6)
        plt.xlim(0,5001)
        if i !=0 : 
            plt.yticks([-5,0,5],[])
        else: 
            plt.yticks([-5,0,5],[-5,0,5])
        
        if i<0:
            plt.xticks([343,1853, 3359, 4869],[])
        else: 
            plt.xticks([343,1853, 3359, 4869],[]) 
    
    plt.subplots_adjust(hspace=0.15,wspace=0.35)
    plt.savefig('figures/normalized_log_returns.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return 

    
def W():

    """ 
    Weight distribution in canonical sectors. Each of the eight subplots shows the
    constituent participation weights of all 705 companies in a canonical sector (rows of W_fs). 
    Stocks are colored by their scottrade classification.
    """

    E,W,C,SSE,varexpl = local_load('saves/8_factor_matrices.pkl.gz')

    cols = ['#984ea3', '#6a51a3', '#df65b0', '#e41a1c', '#fff000', '#b3de69', '#4daf4a', 
            '#377eb8', '#006d2c', '#ff7f00', '#ae017e', '#a65628', '#f781bf',  '#3f007d']
    sects = ['basic', 'capital', 'cyclical', 'energy',    'fin',    'health', 'noncyc',   
             'tech',   'telecom', 'utils', 'miscservices', 'realestate', 'retail', 'transport']
    numcos = [58, 61, 41, 42, 107, 53, 40, 93, 6, 57, 55, 31, 46, 15]

    plt.rcParams['ytick.color']='#636363'
    plt.rcParams['xtick.color']='#636363'
    plt.rcParams['axes.linewidth']= 0

    order=[3, 1, 7, 0, 4, 2, 5, 6]

    fig = plt.figure(figsize=(12,12))
    fig.frameon = False

    for i in xrange(0,np.shape(W)[0],1):
        plt.subplot(5,2,i+1)
        plt.xlim(0,705)
        plt.yticks([])
        plt.ylim(0,1)
        if i>5:
            plt.xticks(xrange(0,705,100), []) 
        else: 
            plt.xticks(xrange(0,705,100), [])
        plot_eve(W.T, order[i], numcos, sects, cols)

    plt.subplots_adjust(hspace=0.2,wspace=0.1)
    plt.savefig('figures/W.png',bbox_inches='tight',pad_inches=0)
    plt.close()

    return


def V():

    """
    Creates Figure D1 in associated paper. See paper for description.
    """

    cols = ['#984ea3', '#6a51a3', '#df65b0', '#e41a1c', '#fff000', '#b3de69', '#4daf4a',
            '#377eb8', '#006d2c', '#ff7f00', '#ae017e', '#a65628', '#f781bf',  '#3f007d']
    sects = ['basic', 'capital', 'cyclical', 'energy',    'fin',    'health', 'noncyc',
             'tech',   'telecom', 'utils', 'miscservices', 'realestate', 'retail', 'transport']
    numcos = [58, 61, 41, 42, 107, 53, 40, 93, 6, 57, 55, 31, 46, 15]

    dataframe = pd.read_pickle('data/data.pkl')
    new_X = dataframe.values[:, :]
    Y = new_X-new_X.mean(axis=0)
    Z = Y/Y.std(axis=0)
    u0, s0, vT0 = svd(Z.T, full_matrices=0)

    plt.rcParams['ytick.color']='#636363'
    plt.rcParams['xtick.color']='#636363'
    plt.rcParams['axes.linewidth']= 0

    fig = plt.figure(figsize=(12,12))
    fig.frameon = False

    for i in xrange(0,8,1):
        plt.subplot(5,2,i+1)
        plt.xlim(0,705)
        if np.mod(i+1,2)==0:
            plt.yticks([-0.3,0,0.1],[])
        else:
            plt.yticks([-0.3,0,0.1], [])       
        plt.ylim(-0.30,0.15)
        if i>5:
            plt.xticks(xrange(0,705,100), [] ) 
        else: 
            plt.xticks(xrange(0,705,100), [])
        plot_eve(u0,i,numcos,sects,cols)

    plt.subplots_adjust(hspace=0.2,wspace=0.1)
    plt.savefig('figures/V.png',bbox_inches='tight',pad_inches=0)
    plt.close()

    return


def C():

    """
    Creates Figure B1 in associated paper. See paper for description.
    """

    E,W,C,SSE,varexpl = local_load('saves/8_factor_matrices.pkl.gz')

    cols = ['#984ea3', '#6a51a3', '#df65b0', '#e41a1c', '#fff000', '#b3de69', '#4daf4a', 
            '#377eb8', '#006d2c', '#ff7f00', '#ae017e', '#a65628', '#f781bf', '#3f007d']
    sects = ['basic', 'capital', 'cyclical', 'energy',    'fin',    'health', 'noncyc',   
             'tech',   'telecom', 'utils', 'miscservices', 'realestate', 'retail', 'transport']
    numcos = [58, 61, 41, 42, 107, 53, 40, 93, 6, 57, 55, 31, 46, 15]

    plt.rcParams['ytick.color']='#636363'
    plt.rcParams['xtick.color']='#636363'
    plt.rcParams['axes.linewidth']= 0

    order=[3, 1, 7, 0, 4, 2, 5, 6]

    fig = plt.figure(figsize=(12,12))
    fig.frameon = False

    for i in xrange(0,np.shape(W)[0],1):
        plt.subplot(5,2,i+1)
        plt.xlim(0,705)
        if i>5:
            plt.xticks(xrange(0,701,100), ['0', '100','200','300','400','500','600','700'])
        else: 
            plt.xticks(xrange(0,701,100), [])
        plot_eve(C, order[i], numcos, sects, cols)

    plt.subplots_adjust(hspace=0.2,wspace=0.1)
    plt.savefig('figures/C.png',bbox_inches='tight',pad_inches=0)
    plt.close()

    return


def flows():

    """ Plots evolution of company sector associated in time """
    
    flows = local_load("saves/flows.pkl.gz")
    dataframe = pd.read_pickle('data/data.pkl')
    companies = ['Berry Petroleum', 'Corning','IBM', 'PG&E','Plum Creek Timber','Wal-Mart']
    company_tickers = ['BRY','GLW','IBM','PCG','PCL','WMT']
    for i in xrange(len(companies)):
        ticks = dataframe.columns
        tick = company_tickers[i]
        plot_flows(flows,ticks,tick, plot_ts=0)
        plt.savefig('figures/'+companies[i]+'_flow.png',bbox_inches='tight',pad_inches=0)
        plt.close()

    return


def noisy_flows():

    """ Plots 'evolution' of noise applied to a static company decomposition """
 
    flows = local_load("saves/noisy_flows.pkl.gz")
    dataframe = pd.read_pickle('data/data.pkl')
    companies = ['Berry Petroleum', 'Corning','IBM', 'PG&E','Plum Creek Timber','Wal-Mart']
    company_tickers = ['BRY','GLW','IBM','PCG','PCL','WMT']
    for i in xrange(len(companies)):
        ticks = dataframe.columns
        tick = company_tickers[i]
        plot_flows(flows,ticks,tick, plot_ts=0)
        plt.savefig('figures/'+companies[i]+'_noisy_flow.png',bbox_inches='tight',pad_inches=0)
        plt.close()

    return


def singular_values():

    """ Plot singular value distribution of stock data """

    X = pd.read_pickle('data/data.pkl')

    plt.rcParams['ytick.color']='#636363'
    plt.rcParams['xtick.color']='#636363'
    plt.rcParams['axes.linewidth']= 0
    plt.rcParams['patch.linewidth'] = 1
    output=StringIO()
    plt.ioff()
    frameon = False

    G = X.values-X.values.mean(axis=0)
    G = G/G.std(axis=0)
    U, S, VT = svd(G.T, full_matrices=0)
    plt.hist(S, np.arange(0, 300, 2), normed=1, histtype='stepfilled', color='#80B1D3');

    (t,n)=np.shape(G)
    ur, sr, vrt = svd(np.random.randn(n,t), full_matrices=0)
    plt.hist(sr, xrange(0, 300, 2), histtype='step',normed=1,  color='#E7298A');
    plt.savefig('figures/singular_values.png',bbox_inches='tight',pad_inches=0)
    plt.close()

    return


def fama_french():

    """ 
    Produces projection plots for our three dimensional decomposition
    and that of Fama and French colored by market cap values
    """

    dataframe = pd.read_pickle('data/data.pkl')  # log returns 
    data, MM = center_normalize_removemm(dataframe)

    S3 = local_load('saves/3_factor_matrices_mm_removed.pkl.gz')[1]

    #Load the fama and french data into pandas with the right parsers
    ff_df = pd.io.parsers.read_table('data/FF_Factors.txt',skiprows=4,delimiter='\s+', parse_dates=True)
    ff_df = ff_df/100

    #restrict fama and french to the dates of interest
    mytimes = dataframe.index
    small_ff_df = ff_df.loc[mytimes]

    #Loads the data for each of the stocks above in a list
    with gzip.open('data/market_cap_list.pkl.gz','rb') as f:
        market_cap_list = pickle.load(f)

    ##clean up the data
    stock_names=[]
    market_caps=[]
    mc_stocks=[]
    for i, cap in enumerate(market_cap_list):
        if cap != 'N/A':
            if "B" in cap:
                market_caps.append(float(cap.replace("B", ""))*10**9)
            if "M" in cap:
                market_caps.append(float(cap.replace("M", ""))*10**6)
            stock_names.append(dataframe.columns[i])
            mc_stocks.append(S3.T[i])
    stock_names=np.asarray(stock_names)
    market_caps=np.asarray(market_caps)
    mc_stocks=np.asarray(mc_stocks)

    ##divide into large cap and small cap
    big_mc_stocks=[]
    big_stock_names=[]
    med_mc_stocks=[]
    med_stock_names=[]
    sm_mc_stocks=[]
    sm_stock_names=[]
    bigmed_cutoff=10*10**9
    smmed_cutoff=2*10**9
    for i in xrange(len(market_caps)):
        if market_caps[i]>=bigmed_cutoff:
            big_mc_stocks.append(mc_stocks[i])
            big_stock_names.append(stock_names[i])
        elif market_caps[i]<=smmed_cutoff:
            sm_mc_stocks.append(mc_stocks[i])
            sm_stock_names.append(stock_names[i])
        else:
            med_mc_stocks.append(mc_stocks[i])
            med_stock_names.append(stock_names[i])
    big_mc_stocks=np.asarray(big_mc_stocks)
    med_mc_stocks=np.asarray(med_mc_stocks)
    sm_mc_stocks=np.asarray(sm_mc_stocks)

    size=3
    fig,axs = plt.subplots(size,size,figsize=(20,20), subplot_kw={"frameon":False}, sharex=True, sharey=True)
    for i in range(size):
        for j in range(size):
            if j>i:
                k = list(set([0,1,2]).difference(set([i])).difference(set([j])))[0]
                c = np.r_[ ["#e41a1c"]*len(big_mc_stocks[:,i]), ["#377eb8"]*len(med_mc_stocks[:,i]), ["#4daf4a"]*len(sm_mc_stocks[:,i])]
                xs = np.r_[ big_mc_stocks[:,i], med_mc_stocks[:,i], sm_mc_stocks[:,i] ]
                ys = np.r_[ big_mc_stocks[:,j], med_mc_stocks[:,j], sm_mc_stocks[:,j] ]
                zs = np.r_[ big_mc_stocks[:,k], med_mc_stocks[:,k], sm_mc_stocks[:,k] ]
                pks = zs.argsort()
            
                axs[i,j].scatter(xs[pks], ys[pks], c=c[pks], linewidth=0, alpha=0.8)
                axs[i,j].axis("equal");
            else:
                axs[i,j].axis('off');
    plt.savefig('figures/fama_french1.png',bbox_inches='tight',pad_inches=0)

    big_stocks=[]
    for name in big_stock_names:
        big_stocks.append(dataframe[name])
    med_stocks=[]
    for name in med_stock_names:
        med_stocks.append(dataframe[name])
    sm_stocks=[]
    for name in sm_stock_names:
        sm_stocks.append(dataframe[name])
    big_stocks=pd.DataFrame(big_stocks)
    med_stocks=pd.DataFrame(med_stocks)
    sm_stocks=pd.DataFrame(sm_stocks)

    out = dolinearregression(big_stocks.T, small_ff_df[['FFMkt-RF','FFSMB','FFHML']],small_ff_df,MM,sub=True)
    big_w=out[0]
    out = dolinearregression(med_stocks.T, small_ff_df[['FFMkt-RF','FFSMB','FFHML']],small_ff_df,MM,sub=True)
    med_w=out[0]
    out = dolinearregression(sm_stocks.T, small_ff_df[['FFMkt-RF','FFSMB','FFHML']],small_ff_df,MM,sub=True)
    sm_w=out[0]

    size=3
    fig,axs = plt.subplots(size,size,figsize=(20,20), subplot_kw= {"frameon":False} ,sharex=True, sharey=True)
    for i in range(size):
        for j in range(size):
            if j>i:
                #k = # other index
                #sortpk = big_w[k,:].argsort()
            
                k = list(set([0,1,2]).difference(set([i])).difference(set([j])))[0]
                c = np.r_[ ["#e41a1c"]*len(big_w[i,:]), ["#377eb8"]*len(med_w[i,:]), ["#4daf4a"]*len(sm_w[i,:])]
                xs = np.r_[ big_w[i,:], med_w[i,:], sm_w[i,:] ]
                ys = np.r_[ big_w[j,:], med_w[j,:], sm_w[j,:] ]
                zs = np.r_[ big_w[k,:], med_w[k,:], sm_w[k,:] ]
                pks = zs.argsort()
            
                axs[i,j].scatter(xs[pks], ys[pks], c=c[pks], linewidth=0, alpha=0.8)            
            
                axs[i,j].axis("equal");
            else:
                axs[i,j].axis('off');

    plt.savefig('figures/fama_french2.png',bbox_inches='tight',pad_inches=0)
    plt.close()

    return

def make_figures():

    """ Runs all functions in this file to produce the figures in the associated paper."""

    cwd = os.getcwd()
    folder = cwd+'/figures/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    single_tetrahedron()

    tetrahedron_array()

    single_tetrahedron_st()

    tetrahedron_array_st()

    company_pies()

    sector_pies()

    cumulative_log_returns()

    unweighted_price_index()

    normalized_log_returns()

    W()

    V()

    C()

    flows()

    noisy_flows()

    singular_values()

    fama_french()

    return 
