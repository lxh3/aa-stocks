import os
import gzip
import numpy as np
import pandas as pd
from scipy.linalg import svd, lstsq
from plot_functions import define_colors

try:
    import cPickle as pickle
except:
    import pickle

from archetypal_analysis import archanalysis, archanalysis_fixed_C


def center_normalize(dataframe):

    """
    Takes stock return data stored in pandas dataframe and 
    converts it into a centered, normalized, numpy array
    """

    Y = dataframe-dataframe.mean(axis=0) #Center
    Z = Y/Y.std(axis=0) #Normalize
    u0, s0, vT0 = svd(Z.T, full_matrices=0)
    data = np.dot(u0[:,1:], s0[1:][:,None]*vT0[1:,:]).T # Filter market

    return data


def center_normalize_removemm(dataframe):

    """
    Takes stock return data stored in pandas dataframe and
    converts it into a centered, normalized, numpy array 
    with the market mode removed
    """

    Y = dataframe-dataframe.mean(axis=0)
    Z = Y/Y.std(axis=0)
    Z2 = (Z.T-Z.mean(axis=1)).T
    MM = Z.mean(axis=1) #Market Mode

    u0, s0, vT0 = svd(Z2.T, full_matrices=0)
    data = np.dot(u0[:,1:], s0[1:][:,None]*vT0[1:,:]).T
    return data, MM


def local_load(fn):

    """ load with gzip and pickle"""

    with gzip.open(fn) as f:
        data = pickle.load(f)
    return data


def run_aa(dataframe, minarch=2, maxarch=9, initialization_index=128, maxiter=500, conv_crit=1e-11, ff=False, verbose=False):
    
    """
    Runs archetypal analysis algorithm and saves the associated matrices

    Input:
    data - numpy array of centered, normalized returns
    minarch/maxarch - min/max number of archetypes desired  
    initialization_index - random seed for FurthestSum in archetypal_analysis 
    maxiter - maximum iterations allowed 
    conv_crit - the convergence criteria
    ff - whether output will be compared with that of fama and french (ie remove market mode)

    Output (saved):
    E - basis vectors representing the n corners (archetypes) of the hyper-tetrahedron 
    W - participation weights matrix
    C - convex hull of the dataset
    SSE - sum of squares error
    varexpl - percent variation explained by the model

    """

    cwd = os.getcwd()
    folder = cwd+'/saves/'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    delta=0

    if not ff:
        data = center_normalize(dataframe)

        for i in xrange(minarch, maxarch+1):

            if verbose:
                print 'Performing archetypal analysis with '+str(i)+' factors'
            E,W,C,SSE,varexpl = archanalysis(data, i, initialization_index, delta, conv_crit, maxiter)
            filename = 'saves/'+str(i)+'_factor_matrices.pkl.gz'
            with gzip.open(filename,'wb') as f:
                pickle.dump([E,W,C,SSE,varexpl],f,-1)

        MM= None

    else:
        data, MM = center_normalize_removemm(dataframe)

        if verbose:
            print 'Performing 3 factor archetypal analysis on data with the market mode removed'

        E,W,C,SSE,varexpl = archanalysis(data, 3, initialization_index, delta, conv_crit, maxiter)
        filename = 'saves/3_factor_matrices_mm_removed.pkl.gz'
        with gzip.open(filename,'wb') as f:
            pickle.dump([E,W,C,SSE,varexpl],f,-1)

    return


def gaus(mu, t1, t2, sdev):

    """ Gaussian function """ 

    return np.exp( -((np.arange(t1,t2)-mu)/sdev)**2./2. ) 


def run_aa_flows(dataframe,noise=False,verbose=False):

    """
    Runs archetypal analysis algorithm on gaussian weighted stock data 
    where the gaussian moves through time

    Input:
    data - numpy array of centered, normalized returns
    noisy - flag to add noise for a robustness check

    Output (saved):
    flows - company evolution in time wrt fixed sectors

    """
 
    num_archetypes = 8
    filename = 'saves/'+str(num_archetypes)+'_factor_matrices.pkl.gz' 
    XC,S,C,SSE,varexpl = local_load(filename)
    
    if noise:

        if verbose:
            print "Computing 'company' evolution in time for gaussian noise"        
        X = pd.read_pickle('data/data.pkl')

        sectors = np.asarray(['Industrial','Energy','Real Estate','Cyclical','Non-Cyclical','Tech','Utility','Finance'])
        W = pd.DataFrame(S,index=sectors,columns=X.columns.values)
        company_tickers = ['BRY','GLW','IBM','PCG','PCL','WMT']
        company_weights = W[company_tickers]
        
        fake_data = XC.dot(company_weights.values)+np.random.normal(size=(X.values.shape[0],company_weights.values.shape[1]))
        X[company_tickers] = fake_data
        
        data = center_normalize(X)
        
    else:

        if verbose: 
            print "Computing company evolution in time"

        data = center_normalize(dataframe)

    C0 = C.copy()
    S0 = S.copy()
    T,N = dataframe.shape
    jump = 50. 
    t_end = T
    sdev = 250. # Gaussian window s.d.

    delta_1=0.0
    initialization_index = 128 # random seed for FurthestSum

    flows = np.zeros((np.shape(np.arange(0, t_end+1, jump))[0], num_archetypes, N))

    for i, mu in enumerate(np.arange(0, t_end+1, jump)):

        #print "WINDOW # ", i, ". Peak at ", mu
   
        B = data * gaus(mu, 0, t_end, sdev)[:,None]
    
        [XC, S, C, sse, varexp] = archanalysis(B, noc=num_archetypes, i_0=initialization_index, delta=delta_1,
                                               conv_crit=1e-7, C=C0, S=S0, updateC=False, updateS=True)
    
        if i>0: 
            ordr = np.argmax(np.dot(flows[i-1], S.T), axis=1)
            #print "Ordering of S ", ordr
            assert all(np.arange(num_archetypes)==np.unique(ordr)), "Rows of S cannot be permuted!"   
            flows[i] = S[ordr,:]
        else: flows[i] = S
    
    if noise:
        savefn = 'saves/noisy_flows.pkl.gz'
    else:
        savefn = 'saves/flows.pkl.gz'
        
    with gzip.open(savefn,'wb') as f:
        pickle.dump(flows,f,-1)
        
    return


def str_template_head(num, name, hex_color):

    """ Template for head of sankey json file """

    return '{"node":'+str(num)+',"name":"'+str(name)+'", "color":"'+str(hex_color)+'"},'


def str_template_body(i,j,w):

    """ Template for body of sankey json file """

    return '{"source":'+str(i)+',"target":'+str(j)+',"value":'+str(w)+'},'


def create_sankey_json():

    """ Creates json necessary to produce the sankey diagram using D3.js """
    
    X = pd.read_pickle('data/data.pkl')
    XC_list = []
    weights = []
    for i in xrange(2,10):
        XC,S,C,SSE,varexpl = local_load('saves/'+str(i)+'_factor_matrices.pkl.gz')
        XC_list.append(XC)
        weights.append(S)
        
    corner_fits = []
    for i in xrange(1,len(XC_list)):
        XC,S,SSE,varexpl = archanalysis_fixed_C(XC_list[i],XC_list[i-1], noc=i+1, i_0=0)
        corner_fits.append(S)
    
    #### GET NODE WEIGHTS ####
    
    node_weights = []
    w = weights[-1].sum(axis = 1)/(np.sum(weights[-1],axis=1)).sum()
    node_weights.append(w)
    
    for c in reversed(corner_fits):
        nw = w.dot(c.T)
        node_weights.append(nw)
        w = nw
        
    node_weights = node_weights[::-1] 
    
    #### DEFINE COLORS ####
    
    num_comps = []
    for i in xrange(len(weights)):
        temp_arry = []
        for j in xrange(i+2):
            temp_arry.append(len(np.where(np.argmax(weights[i],axis=0)==j)[0]))
        num_comps.append(np.asarray(temp_arry))

    cs_list = []
    cs = np.asarray(['#000000','#000000','#000000','#000000','#000000','#000000','#000000','#000000','#000000'])
    cs_list.append(cs)
    cs = np.asarray(['#984EA3', '#E41A1C', '#A65628', '#F781BF', '#4DAF4A', '#377EB8', '#FF7F00', '#FFF000'])
    cs_list.append(cs)
        
    for i in xrange(-2,-8,-1):
        cs = define_colors(cs,corner_fits[i],num_comps[i])
        cs_list.append(cs)
    cs_list = cs_list[::-1] 
    
    #### DEFINE LABELS ####
    
    sectors = np.asarray(['c-industrial','c-energy','c-real estate','c-cyclical',
                          'c-non-cyclical','c-tech','c-utility','c-financial'])
    sectors2 = np.asarray(['c-assets','c-goods'])
    names = []
    for i in xrange(8):
        temp_names=[]
        for j in xrange(i+2):
            if i == 0:
                temp_names.append(sectors2[j])
            elif i ==6:
                temp_names.append(sectors[j])
            else:
                temp_names.append('')
        names.append(np.asarray(temp_names))
     
    #### WRITE JSON FILE ####
    
    nums = [9,8,7,6,5,4,3,2]

    #Header
    head = ''
    num=0
    for i in xrange(len(cs_list)-1,-1,-1):
        c = cs_list[i]
        n = names[i]
        for j in xrange(len(c)):
            head += str_template_head(num,n[j],c[j])
            num+=1
    head = head[:-1]

    #Body
    chunk = ''
    count=0
    for i in xrange(len(corner_fits)-1,-1,-1):
        w = corner_fits[i].T*node_weights[i+1][:,None]
        for j in xrange(w.shape[0]):
            for k in xrange(w.shape[1]):
                l = j+int(sum(nums[0:count]))
                m = k+int(sum(nums[0:count+1]))
                chunk += str_template_body(l,m,w[j,k])
        count += 1
    chunk = chunk[:-1]

    #Full String
    full_str = '{"nodes":['
    full_str += head
    full_str +='],"links":['
    full_str += chunk
    full_str += ']}'

    #Save
    with open("saves/sankey.json", "w") as text_file:
        text_file.write(full_str)
  
    return


def dolinearregression(y, X, small_ff_df, MM, sub=True, const=True, market=True):

    """ Do the linear regression on two data frames"""

    # Try to remove the safe returns
    X=X.copy()
    if sub:
        y = y.sub(small_ff_df.FFRF,axis='index')
    yarr = y.values
    if market:
        X['market']=MM.values
    if const:
        X['constant'] = 1
    Xarr = X.values

    # Perfrom the linear regression
    out = lstsq(Xarr,yarr)
    pred = Xarr.dot(out[0])
    
    if len(y.shape)==2:
        Ns = y.shape[1]
    else:
        Ns=1
    Nt = y.shape[0]
    p = Xarr.shape[1] - 1
    
    sserr = (( yarr - pred)**2).sum(0)
    sstot = ((yarr-yarr.mean(0))**2).sum(0)
    dfe = y.shape[0]-1
    dft = y.shape[0] - Xarr.shape[1] - 1 - 1
    
    r2rawarr = 1. - (sserr)/(sstot)
    r2raw = 1. - (sserr.sum())/(sstot.sum())
    r2adjarr = 1 - ( 1. - r2rawarr)* (Nt-1)/( Nt - p - 1)
    r2adj = 1 - ( 1. - r2raw)* (Nt-1)/( Nt - p - 1)
    r2ours=(1./Ns)*(1. - sserr/sstot).sum()
    #print r2raw, r2adj, r2ours
    
    return out
