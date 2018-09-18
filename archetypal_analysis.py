"""
This script is a python implementation of the MATLAB code accompanying:

Morup, M. and Hansen, L.K., Archetypal analysis for machine learning and data mining.
Neurocomputing,2012,80,54-63

written by Ricky Chachra and updated by Lorien Hayden
"""

import numpy as np

def FurthestSum(K, noc, i, exclude=[]):

    """
    % FurthestSum algorithm to efficiently generate initial seeds/archetypes
    %
    % Usage
    %   i=FurthestSum(K,noc,i,exclude)
    %
    % Input:
    %   K           Either a data matrix or a kernel matrix
    %   noc         number of candidate archetypes to extract
    %   i           inital observation used for to generate the FurthestSum
    %   exclude     Entries in K that can not be used as candidates
    %
    % Output:
    %   i           The extracted candidate archetypes
    """

    (I,J)=np.shape(K)
    index=np.arange(J)
    index[exclude]=-1
    index[i]=-1
    ind_t=i
    i= [i]
    sum_dist=np.zeros((1,J))
    Kt=K    
    Kt2=np.sum(Kt**2., axis=0)

    for k in np.arange(noc+10):

        if k>noc-2: # Remove initial seed        
           Kq=np.dot(Kt[:,i[0]].T, Kt) 
           sum_dist=sum_dist - np.sqrt(np.abs(Kt2 - 2*Kq + Kt2[i[0]]))                                     
           index[i[0]]= i[0]  
           del i[0] 

        t=index[index >= 0]
        Kq=np.dot(Kt[:,ind_t].T , Kt)
        sum_dist = sum_dist + np.sqrt(np.abs(Kt2-2*Kq+Kt2[ind_t])) 
        val = np.max(sum_dist[0, t])
        ind = np.argmax(sum_dist[0, t])  
        ind_t=t[ind] 
        i.append(t[ind]) 
        index[t[ind]] = -1

    return np.sort(i)



def archanalysis(X, noc, i_0, delta=0, conv_crit=1e-11, maxiter=500, C=None, S=None, I=None, U=None, updateC=True, updateS=True):

    """
    % Principal Convex Hull Analysis (PCHA) / Archetypal Analysis
    %   [XC,S,C,SSE,varexpl]=PCHA(X,noc,W,I,U,delta,varargin)
    %
    %   Solves the following PCH/AA problem
    %   \|X(:,U)-X(:,I)CS\|_F^2 s.t. |s_j|_1=1, 1+delta<=|c_j|_1<=1+delta,
    %   S>=0 and C>=0
    % X             data array (Missing entries set to zero or NaN)
    % noc           number of components
    % I             Entries of X to use for dictionary in C (default: I=1:size(X,2))
    % U             Entries of X to model in S              (default: U=1:size(X,2))
    % opts.         Struct containing:
    %       C            initial solution (optional) (see also output)
    %       S            initial solution (optional) (see also output)
    %       maxiter      maximum number of iterations (default: 500 iterations)
    %       conv_crit    The convergence criteria (default: 10^-6 relative change in SSE)
    % i_0           initial candidate for FurthestSum
    % XC            I x noc feature matrix (i.e. XC=X(:,I)*C forming the archetypes) 
    % S             noc x length(U) matrix, S>=0 |S_j|_1=1
    % C             length(I) x noc matrix, C>=0 1-delta<=|C_j|_1<=1+delta
    % SSE           Sum of Squares Error
    % varexpl       Percent variation explained by the model
    """
    if U is None: U=np.arange(np.shape(X)[1])
    if I is None: I=np.arange(np.shape(X)[1])
    
    SST=np.sum(X[:,U] * X[:,U]) # scalar
    
    # Initialize C 
    if C is None:
        #print("Status: Calling FurthestSum")
        i=FurthestSum(X[:,I],noc,i_0)
        #print("Status: Done with FurthestSum", i)
        C=np.zeros((len(I), noc)) 
        C[i,np.arange(noc)]=np.ones(len(i))  
    XC=np.dot(X[:,I],C) 

    muS, muC, mualpha=1.,1.,1.
    
    # Initialize S 
    if S is None:   
        XCtX = np.dot(XC.T , X[:,U])  
        CtXtXC = np.dot(XC.T, XC)     
        S=-np.log(np.random.rand(noc,len(U)))
        assert ~(np.isnan(S).any()), "PCHA: S has random nan values at initialization"
        assert (np.sum(S, axis=0) !=0).all(), "PCHA: S.sum(0) has 0 values at initialization"
        S=S / np.sum(S, axis=0)  
        SSt=np.dot(S,S.T) 
        SSE=SST-2.* np.sum(XCtX * S) + np.sum(CtXtXC *SSt) 
        #print("Status: Calling Supdate to initialize")
        [S,SSE,muS,SSt]=Supdate(S,XCtX,CtXtXC,muS,SST,SSE,25)  
        #print("Status: Supdate done. S is initialized")
        
    else:
        CtXtXC = np.dot(XC.T, XC)     
        XSt=np.dot(X[:,U], S.T) 
        SSt=np.dot(S, S.T)            
        SSE=SST-2.* np.sum(XC *XSt) + np.sum(CtXtXC *SSt)                 
    
    # Set PCHA parameters
    itr=0 
    dSSE=np.inf 
    varexpl=(SST-SSE)/SST
    
    # Display algorithm profile
    #print( '%12s | %12s | %12s | %12s | %12s | %12s | %12s | %12s'%('Iteration','Expl. var.','Cost func.','Delta SSEf.','muC','mualpha','muS',' Time(s)   ') )
    dline = '-------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+'
       
    while np.abs(dSSE)>=conv_crit*np.abs(SSE) and itr<maxiter and varexpl<0.9999:
        itr += 1
        SSE_old=SSE;
            
        # C (and alpha) update
        XSt=np.dot(X[:,U],S.T)
        
        if updateC: [C,SSE,muC,mualpha,CtXtXC,XC]=Cupdate(X[:,I],XSt,XC,SSt,C,delta,muC,mualpha,SST,SSE,1)     
        
        # S update    
        XCtX=np.dot(XC.T , X[:,U])    
        if updateS: [S,SSE,muS,SSt]=Supdate(S,XCtX,CtXtXC,muS,SST,SSE,10)   
               
        # Evaluate and display iteration
        dSSE=SSE_old-SSE;
        if np.mod(itr,10)==0:  
            #pause(0.000001);
            varexpl=(SST-SSE)/SST;
            #print( '%12.0f | %12.4f | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e \n'%(itr,varexpl,SSE,dSSE/abs(SSE),muC,mualpha,muS) )

    # display final iteration
    varexpl=(SST-SSE)/SST;
    #print(dline)
    #print( '%12.0f | %12.4f | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e \n'%(itr,varexpl,SSE,dSSE/abs(SSE),muC,mualpha,muS) )
    
    return [XC,S,C,SSE,varexpl]     

def Supdate(S,XCtX,CtXtXC,muS,SST,SSE,niter):
    
    (noc,J)=np.shape(S)

    for k in np.arange(niter):
        SSE_old=SSE  
        g=(np.dot(CtXtXC, S)-XCtX) /(float(SST)/float(J)) 
        g=g -  np.sum(g*S, axis=0)[None,:]
        stop=True 
        Sold=S 
            
        while stop:
            S=Sold-g*muS 
            S[S<0] = 0. 
            assert (np.sum(S, axis=0) !=0).all(), "Supdate dividing by 0"
            assert ~(np.isnan(S).any()), "Supdate: S has nan values"
            S=S / np.sum(S, axis=0)[None,:]
            SSt=np.dot(S, S.T)      
            SSE=SST-2.* np.sum(XCtX *S)+ np.sum(CtXtXC *SSt)              
            if SSE<=SSE_old*(1.+1e-9):
                muS=muS*1.2 
                stop=False 
            else: 
                muS=muS/2.
    return [S, SSE, muS, SSt]

def Cupdate(X,XSt,XC,SSt,C,delta,muC,mualpha,SST,SSE,niter=1):
                                       
    (J,noc)=np.shape(C)
    if delta != 0:
        alphaC=np.sum(C, axis=0)  
        C=np.dot(C,np.diag(alphaC**-1.))     

    XtXSt=np.dot(X.T, XSt)
    
    for k in np.arange(niter):
        
        # Update C        
        SSE_old=SSE         
        g = (np.dot(X.T,np.dot(XC,SSt))-XtXSt)/float(SST)         
        if delta != 0: g=np.dot(g, np.diag(alphaC)) 
            
        g=g - np.sum(g *C, axis=0)[None,:]         
        stop2=True
        Cold=C
        while stop2:
            C=Cold-muC*g 
            C[C<0]=0. 
            
            nC=np.sum(C, axis=0)+2.2e-16 
            C= np.dot(C,np.diag(nC**-1))  
            if delta != 0:
                Ct= np.dot(C,np.diag(alphaC))  
            else: Ct=C 
                
            XC=np.dot(X,Ct) 
            CtXtXC=np.dot(XC.T, XC) 
            SSE=SST-2.* np.sum(XC *XSt) +   np.sum(CtXtXC *SSt)  
            if SSE<=SSE_old*(1.+1e-9):
                muC=muC*1.2 
                stop2=False 
            else: muC=muC/2.
        
        # Update alphaC        
        SSE_old=SSE 
        if delta !=0:                                                           
            g=(np.diag(np.dot(CtXtXC,SSt)).T /alphaC - np.sum(C*XtXSt, axis=0 ))/(SST*float(J))                        
            stop=True 
            alphaCold=alphaC 
            while stop:
                alphaC=alphaCold-mualpha*g 
                alphaC[alphaC<1.-delta]=1.-delta 
                alphaC[alphaC>1.+delta]=1.+delta                             
                XCt=np.dot(XC, np.diag(alphaC /alphaCold)) 
                CtXtXC = np.dot(XCt.T,  XCt)    
                SSE=SST- 2.* np.sum(XCt *XSt) + np.sum(CtXtXC *SSt) 

                if SSE <= SSE_old*(1.+1e-9):  
                    mualpha=mualpha*1.2 
                    stop=False 
                    XC=XCt 
                else: mualpha=mualpha/2.
                    
    if delta !=0: C=np.dot(C,np.diag(alphaC)) 
    
    return [C,SSE,muC,mualpha,CtXtXC,XC]

def archanalysis_fixed_C(X, XC, noc, i_0, delta=0, conv_crit=1e-11, maxiter=500, S=None, I=None, U=None, updateS=True):
    """
    Archetypal Analysis performed with fixed features XC 
    
    used to find relations between decompositions with differing numbers of archetypes 
    in order to construct a Sankey diagram
    """
    if U is None: U=np.arange(np.shape(X)[1])
    if I is None: I=np.arange(np.shape(X)[1])
    
    SST=np.sum(X[:,U] * X[:,U]) # scalar
    
    muS, muC, mualpha=1.,1.,1.
    
    # Initilize S 
    if S is None:   
        XCtX = np.dot(XC.T , X[:,U])  
        CtXtXC = np.dot(XC.T, XC)     
        S=-np.log(np.random.rand(noc,len(U)))
        assert ~(np.isnan(S).any()), "PCHA: S has random nan values at initialization"
        assert (np.sum(S, axis=0) !=0).all(), "PCHA: S.sum(0) has 0 values at initialization"
        S=S / np.sum(S, axis=0)  
        SSt=np.dot(S,S.T) 
        SSE=SST-2.* np.sum(XCtX *S)+ np.sum(CtXtXC *SSt) 
        #print "Status: Calling Supdate to initialize" 
        [S,SSE,muS,SSt]=Supdate(S,XCtX,CtXtXC,muS,SST,SSE,25)   
        #print "Status: Supdate done. S is initialized"
        
    else:
        CtXtXC = np.dot(XC.T, XC)     
        XSt=np.dot(X[:,U], S.T) 
        SSt=np.dot(S, S.T)            
        SSE=SST-2.* np.sum(XC *XSt) + np.sum(CtXtXC *SSt)                 
    
    # Set PCHA parameters
    itr=0 
    dSSE=np.inf 
    varexpl=(SST-SSE)/SST
    
    # Display algorithm profile
    #print '%12s | %12s | %12s | %12s | %12s | %12s | %12s | %12s'%('Iteration','Expl. var.','Cost func.','Delta SSEf.','muC','mualpha','muS',' Time(s)   ')
    dline = '-------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+'
       
    while np.abs(dSSE)>=conv_crit*np.abs(SSE) and itr<maxiter and varexpl<0.9999:
        itr += 1
        SSE_old=SSE;
            
        # C (and alpha) update
        XSt=np.dot(X[:,U],S.T)
        
        # S update    
        XCtX=np.dot(XC.T , X[:,U])    
        if updateS: [S,SSE,muS,SSt]=Supdate(S,XCtX,CtXtXC,muS,SST,SSE,10)   
               
        # Evaluate and display iteration
        dSSE=SSE_old-SSE;
        if np.mod(itr,10)==0:  
            #pause(0.000001);
            varexpl=(SST-SSE)/SST;
            #print '%12.0f | %12.4f | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e \n'%(itr,varexpl,SSE,dSSE/abs(SSE),muC,mualpha,muS) 

    # display final iteration
    varexpl=(SST-SSE)/SST;
    #print dline
    #print '%12.0f | %12.4f | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e \n'%(itr,varexpl,SSE,dSSE/abs(SSE),muC,mualpha,muS) 
   
    return [XC,S,SSE,varexpl]     

