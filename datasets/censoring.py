import numpy as np 
import pandas as pd  
import random 
import math

############################################## Method for time censoring data ##################################################

def time_censoring_data(df):
    """_summary_

    Args:
        survival dataframe (pandas dataframe): complete follow-up data + censored data

    Returns:
        TC_dta (pandas dataframe): time-censored follow-up data
    """
    np.random.seed(1234)
    
    column = "status" if 'status' in df.columns else "event"
    uncensored_data=df[df[column]==1]
    censored_data = df[df[column]==0]
    
    TC_data=uncensored_data
    N=TC_data.shape[0]
    
    Q50=np.median(TC_data['time'])
    Q99=np.percentile(TC_data['time'], 99)
    
    k=(np.log(np.log(100)-np.log(2)))/(np.log(Q99)-np.log(Q50))
    b=Q50/(np.log(2)**(1/k))
    p=np.random.uniform(0,1,N)
    
    time=np.array(TC_data.loc[:,'time'])
    status=np.array(TC_data.loc[:,column])
    t=np.zeros(N)
    
    for i in range(N):
        t[i]=b*(np.log(1/p[i]))**(1/k)
        if t[i]<time[i]:
            time[i]=t[i]
            status[i]=0
    
    TC_data.loc[:,'time']=time
    TC_data.loc[:,column]=status
    
    new_TC_data=pd.concat([TC_data,censored_data],axis=0).reset_index(drop=True) 
    return new_TC_data 

###############################################################################################################################

def interim_censoring_data(df):
    np.random.seed(1234)
    column = "status" if 'status' in df.columns else "event"
    
    uncensored_data=df[df[column]==1]
    censored_data = df[df[column]==0]
    
    IC_data=uncensored_data
    N=IC_data.shape[0]
    Q50=np.median(IC_data['time'])
    Q99=np.percentile(IC_data['time'], 99)
    k=(np.log(np.log(100)-np.log(2)))/(np.log(Q99)-np.log(Q50))
    b=Q50/(np.log(2)**(1/k))
    p1=np.random.uniform(0,1,N)
    p2=np.random.uniform(0,1,N)
    time=np.array(IC_data.loc[:,'time'])
    status=np.array(IC_data.loc[:,column])
    t1=np.zeros(N)
    t2=np.zeros(N)
    for i in range(N):
        t1[i]=b*(np.log(1/p1[i]))**(1/k)
    for i in range(N):
        t2[i]=b*(np.log(1/p2[i]))**(1/k)
    for i in range(N):    
        if np.absolute(t1[i]-t2[i])<time[i]:
            time[i]=np.absolute(t1[i]-t2[i])
            status[i]=0
    IC_data.loc[:,'time']=time
    IC_data.loc[:,column]=status
    
    new_IC_data=pd.concat([IC_data,censored_data],axis=0).reset_index(drop=True) 
    return new_IC_data

##############################################################################################################################

def case_censored_data(df,p):
    np.random.seed(1234)
    column = "status" if 'status' in df.columns else "event"
    
    uncensored_data=df[df[column]==1]
    censored_data = df[df[column]==0]
    
    CC_data=uncensored_data  ## All uncensored data
    N=CC_data.shape[0]  ## Shape of uncensored data
    
    sampled_set = random.sample(range(N), math.floor(p*N)) ## Randomly select p% of observation from total number of observations N
    CC_censored_data=CC_data[CC_data.index.isin(sampled_set)] ## Extract the randomly selected observations from the uncesnored data for creating censored cohort
    CC_uncensored_data=CC_data[~CC_data.index.isin(sampled_set)] ## Uncensored group who are not in censored cohort
    CC_censored_data.loc[:,'time']=(CC_censored_data.loc[:,'time']-np.random.uniform(0,0.5,1)[0]*CC_censored_data.loc[:,'time']).astype('int64') ## Observed time for censored group is reduced by a random amount (generated from uniform distribution)
    
    CC_censored_data.loc[:,column]=0 ## Change the label from 1 to 0 for censored group
    CC_full_data=pd.concat([CC_uncensored_data,CC_censored_data]) ## Concatenate censored and uncensored data
    CC_full_data=CC_full_data.sample(frac=1) ## Randomly suffle the dataset 
  
    new_CC_data=pd.concat([CC_full_data,censored_data],axis=0).reset_index(drop=True) 
    return new_CC_data
