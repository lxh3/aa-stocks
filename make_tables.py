import pandas as pd
import numpy as np

from collections import Counter

from fit_functions import local_load


def ninth_sector_comps():
    
    """ Create list of companies composing the 9th sector """

    # Load dataframe, 8 sector and 9 sector fits
    X = pd.read_pickle('data/data.pkl')
    XC8,S8,C8,SSE8,varexpl8 = local_load('saves/8_factor_matrices.pkl.gz')
    XC9,S9,C9,SSE9,varexpl9 = local_load('saves/9_factor_matrices.pkl.gz')

    # Divide companies by strongest sector affiliation for the 8 factor model
    divided_comps_8 = []
    for i in xrange(8):
        index = np.where(np.argmax(S8,axis=0)==i)
        comps = X.columns.values[index]
        divided_comps_8.append(comps)
    
    # Divide companies by strongest sector affiliation for the 9 factor model
    divided_comps_9 = []
    for i in xrange(9):
        index = np.where(np.argmax(S9,axis=0)==i)
        comps = X.columns.values[index]
        divided_comps_9.append(comps)

    # Determine overlap between 8 and 9 sector decompositions
    count = np.zeros((9,8))
    for i in xrange(9):
        for j in xrange(8):
            C = Counter(divided_comps_8[j]) & Counter(divided_comps_9[i])
            count[i,j]=(len(list(C.elements())))

    # Visually identified 9th sector and it's constituent companies from count
    ninth_sector_companies = divided_comps_9[4]
   
    return ninth_sector_companies


def top_two_sector_comps():

    """ 
    Creates list of top 20 companies composing each of the sectors in the
    two sector decomposition with the percentage they contribute to the sector
    """

    X = pd.read_pickle('data/data.pkl')

    # Add the sector and ticker information
    sects = ['basic', 'capital', 'cyclical', 'energy', 'fin', 'health', 'noncyc',
             'tech', 'telecom', 'utils', 'miscservices', 'realestate', 'retail', 'transport']
    numcos = [58, 61, 41, 42, 107, 53, 40, 93, 6, 57, 55, 31, 46, 15]
    sulis_sects = [[v]*k for k, v in zip(numcos, sects)]
    sectsarray= np.array([item for sublist in sulis_sects for item in sublist])
    tickers = X.columns.values.tolist()
    sectors = sectsarray.tolist()
    tuples = list(zip(*[tickers,sectors]))
    index = pd.MultiIndex.from_tuples(tuples, names=['tickers', 'sectors'])
    Xst = pd.read_pickle('data/data.pkl')
    Xst.columns = index

    #Load two factor matrices
    XC2,S2,C2,SSE2,varexpl2 = local_load('saves/2_factor_matrices.pkl.gz')

    # Determine top 20 companies in each of the two sectors
    # and the percentage of the sector attributable to them
    Corner1 = Xst.columns[C2[:,0].argsort()[-20:][::-1]]
    Percent1 = np.round(C2[C2[:,0].argsort()[-20:][::-1]][:,0]*100,2)
    Corner2 = Xst.columns[C2[:,1].argsort()[-20:][::-1]]
    Percent2 = np.round(C2[C2[:,1].argsort()[-20:][::-1]][:,1]*100,2)
    
    return Corner1, Percent1, Corner2, Percent2


def top_three_sector_comps():

    """ 
    Creates list of top 20 companies composing each of the sectors in the
    three sector decomposition with the percentage they contribute to the sector
    """

    X = pd.read_pickle('data/data.pkl')
    
    # Add the sector and ticker information
    sects = ['basic', 'capital', 'cyclical', 'energy', 'fin', 'health', 'noncyc',
             'tech', 'telecom', 'utils', 'miscservices', 'realestate', 'retail', 'transport']
    numcos = [58, 61, 41, 42, 107, 53, 40, 93, 6, 57, 55, 31, 46, 15]
    sulis_sects = [[v]*k for k, v in zip(numcos, sects)]
    sectsarray= np.array([item for sublist in sulis_sects for item in sublist])
    tickers = X.columns.values.tolist()
    sectors = sectsarray.tolist()
    tuples = list(zip(*[tickers,sectors]))
    index = pd.MultiIndex.from_tuples(tuples, names=['tickers', 'sectors'])
    Xst = pd.read_pickle('data/data.pkl')
    Xst.columns=index

    # Load three factor matrices
    XC3,S3,C3,SSE3,varexpl3 = local_load('saves/3_factor_matrices.pkl.gz')

    # Determine top 20 companies in each of the three sectors
    # and the percentage of the sector attributable to them
    Corner1 = Xst.columns[C3[:,0].argsort()[-20:][::-1]]
    Percent1 = np.round(C3[C3[:,0].argsort()[-20:][::-1]][:,0]*100,2)
    Corner2 = Xst.columns[C3[:,1].argsort()[-20:][::-1]]
    Percent2 = np.round(C3[C3[:,1].argsort()[-20:][::-1]][:,1]*100,2)
    Corner3 = Xst.columns[C3[:,2].argsort()[-20:][::-1]]
    Percent3 = np.round(C3[C3[:,2].argsort()[-20:][::-1]][:,2]*100,2)

    return Corner1, Percent1, Corner2, Percent2, Corner3, Percent3


def make_table_data():
    
    """ Runs each of the functions in this file and writes the data to a txt file """

    with open('figures/table_data.txt','w') as tf:   
    
        ninth_sector_companies = ninth_sector_comps()
        tf.write('The tickers of the companies constituting the 9th sector (Table G1) are as follows: \n\n')
        for company in ninth_sector_companies:
            tf.write(str(company)+' ')
        tf.write('\n\n\n')

        Corner1, Percent1, Corner2, Percent2 = top_two_sector_comps()
        tf.write('The latex input corresponding to Table G2 is as follows: \n\n')
        for i in xrange(len(Corner1)):
            tf.write(Corner1[i][0]+' & '+Corner1[i][1]+' & '+str(Percent1[i])+'\% & '+Corner2[i][0]+' & '
                     +Corner2[i][1]+' & '+str(Percent2[i])+'\% \\ \n')
        tf.write('\n\n\n')

        Corner1, Percent1, Corner2, Percent2, Corner3, Percent3 = top_three_sector_comps()
        tf.write('The latex input corresponding to Table G3 is as follows: \n\n')
        for i in xrange(len(Corner1)):
            tf.write(Corner1[i][0]+' & '+Corner1[i][1]+' & '+str(Percent1[i])+'\% &'+Corner2[i][0]+' & '
                     +Corner2[i][1]+' & '+str(Percent2[i])+'\% &'+Corner3[i][0]+' & '+Corner3[i][1]+' &'
                     +str(Percent1[i])+'\% \\ \n')
        tf.write('\n\n\n')
    
    return
