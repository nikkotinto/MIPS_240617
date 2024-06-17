#---------------------------------------------------------------------------
# FUNCTIONS FOR DATA MANIPULATION, 
#---------------------------------------------------------------------------
#CONSIDER USING df.assign() instead!

#from cv2 import dft
import pandas as pd
import numpy as np
import config as cfg
import sys
from pathlib import Path
from IPython.display import clear_output  
import pingouin as pg
import os
import time
import matplotlib.pyplot as plt
import math as m
from pathlib import Path
from numpy import False_
import menu_func as mfc

# EXTRA MODULES
from scipy import ndimage
#import fl
#import cv2
#from pyevtk.hl import imageToVTK, gridToVTK, pointsToVTK, polyLinesToVTK
#from nd2reader import ND2Reader
import json
#import pyevtk
from scipy import stats
from IPython.display import display





if 'inh_order' not in globals():
    inh_order = cfg.MIPS_order

#---------------------------------------------------------------------------
# FUNCTIONS DEFINING VARIABLES, CONSIDER USING df.assign() instead!
#---------------------------------------------------------------------------


def order_var(df, var = 'inh', order = 'cfg'):

    if order == 'cfg':
        df[var] = pd.Categorical(df[var], cfg.var_order[var])
    else: 
        df[var] = pd.Categorical(df[var], order)
    #df['quadrant3'] = pd.Categorical(df['quadrant3'], cfg.var_order['quadrant3'])
    return df

def order_cols(df):
    cols_ = df.columns.tolist()
    ordered = []
    unordered = []
    for col in cols_: 
        if col in cfg.var_order:
            ordered.append(col)
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.set_categories(inh_order if col == 'inh' else cfg.var_order[col])
        else:
            unordered.append(col)
    print(f'ordered: {ordered}')
    print(f'unordered: {unordered}')
    return df
            


def topo_bins_var(df, bin_steps = [4, 4, 2], widths = [250, 250, 70], vars = ['x_s','ys','zs']):
    for var, bin_step, width in zip(vars, bin_steps, widths): 
        start_bin = -width #+ bin_step
        start_name = start_bin #+ bin_step/2
        end_name = width - bin_step#/2
        xy_bins = np.arange(start_bin, width, bin_step)
        xy_names = np.arange(start_name,end_name,bin_step).tolist()
        if var in ['x_s', 'ys', 'zs']:
            var_name = var[0] + '_bin'
        else:
            var_name = var + '_bin'
        df[var_name] = pd.cut(df[var], xy_bins, labels=xy_names,include_lowest=True).astype('Int64')
    #df['y_bin'] = pd.cut(df['ys'], xy_bins, labels=xy_names,include_lowest=True).astype('Int64')
    #df['z_bin'] = pd.cut(df['zs'], xy_bins, labels=xy_names,include_lowest=True).astype('Int64')
    return df

def led_bins_var(dfg):#Creates bin variables tled & zled  for heatmaps etc. 
    zbins=[-2,4]+np.arange(8,72,4).tolist()#np.arange(-8,72,4)
    zgroup_names=np.round(np.linspace(0,68,17),0)#np.round(np.linspace(-6,74,19),0)
    #tbins=np.arange(0,196,2)
    tbins=np.linspace(0,194,98).tolist()#np.linspace(0,192,97)
    tgroup_names=np.round(np.linspace(0,600,97),0)
    #tgroup_names=np.arange(0,196,2)
    if 'zs' in dfg.columns:  
        dfg['zled'] = pd.cut(dfg['zs'], zbins, labels=zgroup_names,include_lowest=True).astype('Int64')
    if 'sec' in dfg.columns:
        dfg['tled'] = pd.cut(dfg['frame'], tbins, labels=tgroup_names,include_lowest=True).astype('Float64')#.astype('Int64')    
    return dfg

def x_abs_bin_var(df, bin_step = 4, width = 250, var = 'x_s'):
    start_bin = 0 #+ bin_step
    start_name = start_bin #+ bin_step/2
    end_name = width#/2
    x_bins = np.arange(start_bin, width + bin_step, bin_step)
    x_names = np.arange(start_name,end_name,bin_step).tolist()
    if var in ['x_s', 'ys', 'zs']:
        var_name = var[0] + '_bin_abs'
    else:
        var_name = var + '_bin_abs'
    df[var_name] = pd.cut(df[var].abs(), x_bins, labels=x_names,include_lowest=True).astype('Int64')
    return df

def dist_c_var(df):# Creates variables dist_c & dist_cz that give distance from center
    df['dist_c']=((df.loc[:,'x_s'])**2+(df.loc[:,'ys'])**2)**0.5
    df['dist_cz']=((df.loc[:,'x_s'])**2+(df.loc[:,'ys'])**2+(df.loc[:,'zs'])**2)**0.5
    return df

def dist_i_var(df):
    if 'dist_c' not in df.columns:
        df = dist_c_var(df)
    
    df['dist_i'] = df['dist_c'] - 37.5
    
    
    #df.loc[df.dist_c > 37.5, 'dist_i'] = df['dist_c'] - 37.5
    
    df['dist_iz'] = df['zs']
    df.loc[df.dist_i > 0, 'dist_iz'] = ((df['dist_i'])**2 + (df['zs'])**2)**0.5
    
    
    #df.loc[df.dist_i < 0, 'dist_i']
    
    #df['dist_iz'] = df['dist_i'] + df['zs']
    
    return df

def standardized_var(df, vars = ['dist_iz'], group_vars = ['path']):
    dfg = df.groupby(group_vars)[vars].transform(lambda x: (x - x.mean()) / x.std())
    var_names = [var + '_std' for var in vars]
    
    print(var_names)
    dfg = dfg.rename(columns = {old: new for old, new in zip(vars, var_names)})
    df = pd.concat([df, dfg], axis = 1)
    
    return df
    
    


def dist_bins_var(df):
    dist_bins = list(range(0,122,4))+[250]
    dist_names = dist_bins[1:]
    df['dist_bins'] = pd.cut(df['dist_c'], dist_bins, labels=dist_names).astype('Int64')
    return df

def isovol_bin_var(df,vols_in_injury=5): # Function that calculates isovolumes with four decimals to avoid duplicate bin names
    if 'dist_cz' in df.columns:
        inj_zone_vol=(2/3)*m.pi*(37.5**3)
        vol_step=inj_zone_vol/vols_in_injury#inj_zone_vol/10
        volumes_=np.arange(0,vol_step*101,vol_step)#np.arange(0,vol_step*201,vol_step)
        radii=((3*volumes_/(2*m.pi)))**(1/3)
        radii[-1]=250
        df['iso_vol']=pd.cut(df['dist_cz'],radii,labels=np.round(radii[1:],4)).astype('float64')
        #vol_step=20000
        #n_=np.arange(0,210)
        #radii=((3*vol_step*n_/(2*m.pi)))**(1/3)
        #radii[-1]=250.0
        #df['iso_vol']=pd.cut(df['dist_cz'],radii,labels=radii[1:])
    return df

### HEJ!!

def isovol_bin_var1(df,vols_in_injury=5, volumes = False):# Function that calculates isovolumes with one decimal NB might produce duplicate bin names!
    if 'dist_cz' in df.columns:
        inj_zone_vol=(2/3)*m.pi*(37.5**3)
        vol_step=inj_zone_vol/vols_in_injury #CHANGED FROM 3
        volumes_=np.arange(0,vol_step*101,vol_step)
        radii=((3*volumes_/(2*m.pi)))**(1/3)
        #del radii[-1]
        radii[-1]=250
        if volumes:
            df['iso_volumes']=pd.cut(df['dist_cz'],radii,labels=volumes_[1:]).astype('float64').round(1)
        else:
            df['iso_vol']=pd.cut(df['dist_cz'],radii,labels=radii[1:]).astype('float64').round(1)
     
    return df

def isoA_var(df,nA_in_injury=10, areas = False, var = 'dist_c'):
    if var in df.columns:
        inj_zone_A=m.pi*(37.5**2)#m.pi*(37.5**2)/2
        A_step=inj_zone_A/nA_in_injury #CHANGED FROM 3
        A_=np.arange(0,inj_zone_A*20,A_step)#A_step*(nA_in_injury*10+1)
        radii=((A_/(m.pi)))**(1/2)#((2*A_/(m.pi)))**(1/2)
        radii[-1]=250
        
        if areas: 
            iso_areas = pd.DataFrame({'iso_A': radii[1:], 'iso_A_area': A_[1:]})
            display(iso_areas.head(nA_in_injury))
            df['iso_Area']=pd.cut(df[var],radii,labels=A_[1:]).astype('float64').round(1)
        else:
            df['iso_A']=pd.cut(df[var],radii,labels=radii[1:]).astype('float64').round(1)
        #df['iso_A']=pd.cut(df['dist_c'],radii,labels=radii[1:]).astype('float64').round(0).astype('int64')#.round(1)
        print('Unit area:', A_step)
    else:
        print('Calculate dist_c variable first!')
    return df

def a_bin_var(dfii, max_area = 50000, area_step = 1000):
    #dfii = df.copy()
    if 'area' not in dfii.columns:
        try:
            dfii['area'] = (dfii['dist_c'].pow(2)).mul(np.pi).copy()
        except:
            print('dataframe does not contain the right columns')

    #100000# = dfi1.area.max() +1

    dfii = dfii[dfii.area < max_area]
    #range = pd.interval_range(start = 0 , end = max_area, freq = 2000 )
    names = np.arange(0,max_area,area_step)
    edges = np.arange(0,max_area+area_step,area_step)
    dfii.loc[:, 'a_bin'] = pd.cut(dfii['area'], bins = edges, right = True, labels = names).astype('float64')
    return dfii

def rho_var(df, abs = False, bin = True, bin_width = 4):
    df['rho'] = np.arctan2(df['ys'], df['x_s'].abs())*180/np.pi # Angle = 0 at point (0,1)
    #df['rho'] = np.arctan(df['ys'],df['x_s'].abs(), )*180/np.pi
    #df['rho'] = np.arctan2(df['ys'],df['x_s'].abs(), )*180/np.pi
    if abs:
        df['rho'] = df['rho'].abs()
        
    #df['rho'] = df['rho'] -90
    
    if bin:
       # edges = np.arange(0,184,4)
       # labels = np.arange(2, 182, 4)
        min = -90
        max = 90
        edges = np.arange(min,max + bin_width,bin_width)
        labels = np.arange(min + bin_width/2, max + bin_width/2, bin_width)
        df.loc[:,'rho_bin'] = pd.cut(df['rho'], edges, labels= labels, include_lowest= True).astype('float64').round(1)
        
    return df

def azimuth_var(df, abs = False, bin = True, bin_width = 5):
    df['azi'] = np.arctan2(df['ys'], df['x_s'].abs())*180/np.pi # Angle = 0 at point (0,1)
    #df['rho'] = np.arctan(df['ys'],df['x_s'].abs(), )*180/np.pi
    #df['rho'] = np.arctan2(df['ys'],df['x_s'].abs(), )*180/np.pi
    if abs:
        df['azi'] = df['azi'].abs()
        
    #df['rho'] = df['rho'] -90
    
    if bin:
       # edges = np.arange(0,184,4)
       # labels = np.arange(2, 182, 4)
        min = -90
        max = 90
        edges = np.arange(min,max + bin_width,bin_width)
        #labels = np.arange(min + bin_width/2, max + bin_width/2, bin_width)
        labels = np.linspace(min, max, len(edges)-1)#np.arange(min , max , bin_width)
        df.loc[:,'azi_bin'] = pd.cut(df['azi'], edges, labels= labels, include_lowest= True).astype('float64').round(1)
        
    return df
    

def time_var(df):# Creates time variables sec, minute and phase from frame
    df['sec']=df['frame']*3.1
    return df

def tracktime_var(df):
    df['track_time']=df['nrtracks']*3.1
    return df

def res_time_var(df):
    res_tracks = df['nrtracks'] - df['tracknr'] ## res_time = variable that shows how many tracks that are left before platelet dissociation 
    df['res_time'] = res_tracks * 3.1
    return df


def fframe_var(df, var = 'frame'):
    var_name = 'f'+var
    first_frame = df.sort_values(by = ['path','particle','tracknr']).groupby(['path','particle'])[[var]].first().rename(columns = {var:var_name})#.reset_index()#.rename(var_name)#
    df = df.merge(first_frame, on = ['path','particle'])
    #df[var_name] = df.groupby(['path','particle',var]).nth(0)[var]#.reset
    
    
    return df

def lframe_var(df, var = 'frame'):
    var_name = 'l'+var
    last_frame = df.sort_values(by = ['path','particle','tracknr']).groupby(['path','particle'])[[var]].last().rename(columns = {var:var_name})
    df = df.merge(last_frame, on = ['path','particle'])
    
    return df

def fsec_var(df, sec_per_bin = 30):
    if 'fsec' not in df.columns:
        df = fframe_var(df, var = 'sec')
    
    edges = np.arange(0,600 + sec_per_bin,sec_per_bin)#np.arange(0,205,10)#np.arange(0,196,3)
    #labels = list(np.linspace(0,570,20).astype('int'))#np.round(np.arange(0,600,9.3),0).astype('int')

    df.loc[:,'fsec_bin'] = pd.cut(df['fsec'], edges, labels= edges[:-1], include_lowest= True)
    return df

def bin_time_var(df, bin_size = 10, var = 'sec'):
    var_name = f'{bin_size}_{var}'
    df[var_name] = pd.cut(df[var], np.arange(0,(600+bin_size),bin_size), labels = np.arange(0,600,bin_size),right = True, include_lowest= True).astype('Int32').round(0)
    df[var_name] = pd.Categorical(df[var_name], ordered = True)
    return df

def bintime_from_frame(df, bin_size = 3, var = 'frame'):
    var_name = 'bin_sec'#f'{bin_size}_{var}'
    df[var_name] = pd.cut(df['frame'], np.arange(0,(193+bin_size),bin_size), labels = np.arange(0,600,bin_size*3.1),right = True, include_lowest= True).astype('float64').round(0).astype('Int64')
    return df

def minute_var(df):
    if 'sec' not in df.columns:
        df=time_var(df)
    #df.loc[:,'minute'] = pd.cut(df['sec'], 10, labels=np.arange(1,11,1))
    edges = np.arange(0,660,60)
    t = np.arange(0,11,1)
    #t_i = [str(t1)+'-'+ str(t2) for t1,t2 in zip(t[:-1],t[1:])]
    df.loc[:,'minute'] = pd.cut(df['sec'], edges, labels=t[1:])
    return df

def hsec_var(df):
    t = np.arange(0,700,100)
    t_i = [str(t1)+'-'+ str(t2) for t1,t2 in zip(t[:-1],t[1:])]
    df.loc[:,'hsec'] = pd.cut(df['sec'], t, labels=t_i, include_lowest= True)
    
    return df

def tri_phase(df, var = 'sec'):
    t = [0, 100, 300, 600]#np.arange(0,700,100)
    t_i = [str(t1)+'-'+ str(t2) for t1,t2 in zip(t[:-1],t[1:])]
    df.loc[:,f'tri_{var}'] = pd.cut(df[var], t, labels=t_i, include_lowest= True)
    
    return df

def bi_phase(df, var = 'sec'):
    t = [0, 300, 600]#np.arange(0,700,100)
    t_i = [str(t1)+'-'+ str(t2) for t1,t2 in zip(t[:-1],t[1:])]
    df.loc[:,f'bi_{var}'] = pd.Categorical(pd.cut(df[var], t, labels=t_i, include_lowest= True), t_i, ordered = True)
    
    return df


def phase_var(df):
    df.loc[:,'phase'] = pd.qcut(df['frame'], 3, labels=['Early','Mid','Late'])
    return df

def phasebreak_var(df):
    df['phase']=pd.Categorical(np.where(df.sec<phase_break,'Growth','Consolidation'), cfg.var_order['phase'], ordered = True)
    #df['quadrant3'] = pd.Categorical(df['quadrant3'], cfg.var_order['quadrant3'], ordered = True)
    return df


def injury_zone_var(df):# Creates variable that divide plts into thos inside and outside injury zone
    if 'dist_cz' in df.columns.tolist():
        df['injury_zone']=df.dist_cz<38
    else:
        df=dist_c_var(df)
        df['injury_zone']=df.dist_cz<38
    return df

def height_var(df):
    t =np.arange(0,80,20)
    t_i = [str(t1)+'-'+ str(t2) for t1,t2 in zip(t[:-1],t[1:])]
    df['height'] = pd.cut(df['zs'], 3, labels=t_i, right = True, include_lowest=True)
    
    return df
    
    #if 'zf' in df.columns:
    #    df.loc[:,'height']=pd.cut(df.zf, 3, labels=["bottom", "middle", "top"])
    #else:
    #    df.loc[:,'height']=pd.qcut(df.zs, 3, labels=["bottom", "middle", "top"])

    return df

def size_var(df):

    df_size = df.groupby(['inh','path'], observed = True).size(#df[df.nrtracks>1].groupby(['inh','path']).size(
        ).groupby(level=0).transform(lambda x: pd.qcut(x,2,labels=['small','large'])).rename('size').reset_index()
    df = df.merge(df_size, on = ['inh','path'])
    return(df)

def size_var2(df):
    df_size = df.groupby(['inh','path','frame'], observed = True).size(#df[df.nrtracks>1].groupby(['inh','path']).size(
        ).groupby(level=[0,1]).max().groupby(level=0).transform(lambda x: pd.qcut(x,2,labels=['small','large'])).rename('size').reset_index()
    df = df.merge(df_size, on = ['inh','path'])
    return(df)


def size_quant_var(df):
# size_perc = variable that divides experiments into size quantiles
    df_size = df.groupby(['inh','path'], observed = True).size().groupby(level=0).transform(lambda x: pd.qcut(x,4, labels = [0, 0.25, 0.5, 0.75], )).astype('float64').rename('size_quant')#duplicates= 'drop'
    df = df.merge(df_size, on = ['inh', 'path'])
    df_size = df_size.reset_index()
    #df_size.head()
    df_size[df_size.size_quant > 0.49].groupby(['inh']).size_quant.value_counts()
    return(df)
    
def z_pos_var(df):
    df.loc[:,'z_pos']=pd.cut(df.zf, 2, labels=["bottom", "top"])
    return df

def zz_var(df, decimals = 0):
    if 'zf' in df.columns:
        df['zz']=np.round(df.loc[:,'zf'],decimals= decimals)#pc['zz']=np.round(pc.loc[:,'zs'],decimals=0)
    else:
        df['zz']=np.round(df.loc[:,'zs'],decimals= decimals)
    df = df.astype({'zz': int})
    return df

def position_var(df):
    df['position']='outside'
    df.loc[(df.dist_c<38),'position']='head'
    df.loc[(df.dist_c>38)&(df.ys<0)&(abs(df.x_s)<38),'position']='tail'
    return df

def quadrant_var(df, mode = 'xy'):
    
    if mode == 'xy':
        df['quadrant']='lateral'
        df.loc[(df['ys']>df['x_s'])&(df['ys']>-df['x_s']),'quadrant']='anterior'
        df.loc[(df['ys']<df['x_s'])&(df['ys']<-df['x_s']),'quadrant']='posterior'
        
    else:
        df['quadrant']='lateral'
        if 'rho' in df.columns:
            df.loc[(df['rho'] < -45),'quadrant']='posterior'
            df.loc[(df['rho'] > 45),'quadrant']='anterior'
        elif 'rho_bin' in df.columns:
            df.loc[(df['rho_bin'] < -45),'quadrant']='posterior'
            df.loc[(df['rho'] > 45),'quadrant']='anterior'
        else:
            print('Cannot calculate quadrant var!') 
            
    return df

def quadrant1_var(df):
    if 'injury_zone' not in df.columns:
        df=injury_zone_var(df)
    df['quadrant1']='lateral'
    df.loc[(df['ys']>df['x_s'])&(df['ys']>-df['x_s']),'quadrant1']='anterior'
    df.loc[(df['ys']<df['x_s'])&(df['ys']<-df['x_s']),'quadrant1']='posterior'
    df.loc[df.injury_zone,'quadrant1']='core'
    return df

def quadrant2_var(df):
    if 'inside_injury' not in df.columns:
        df=inside_injury_var(df)
    df['quadrant2']='lateral'
    df.loc[(df['ys']>df['x_s'])&(df['ys']>-df['x_s']),'quadrant2']='anterior'
    df.loc[(df['ys']<df['x_s'])&(df['ys']<-df['x_s']),'quadrant2']='posterior'
    df.loc[df.inside_injury,'quadrant2']='center'
    return df

def quadrant3_var(df, mode = 'xy'):
    
    df['quadrant3']=None    
    df['quadrant3'] = pd.Categorical(df['quadrant3'], cfg.var_order['quadrant3'], ordered = True)
    if mode == 'xy': #DET ÄR FEL HÄR!!!!!!!!!!!
        df.loc[(df['ys']>0),'quadrant3']='al'#'AL'
        df.loc[(df['ys']>df['x_s']) & (df['ys']>-df['x_s']),'quadrant3']='aa'#'A'
        
        df.loc[(df['ys']<0),'quadrant3']='pl'#'PL'
        df.loc[(df['ys']<df['x_s']) & (df['ys']<-df['x_s']),'quadrant3']='pp'#'P'
        
        #df.loc[(df['ys']<df['x_s']),'quadrant3']='P'
        #df.loc[(df['ys']<0),'quadrant3']='P'
        #df.loc[(df['ys']>df['x_s'])&(df['ys']>-df['x_s']),'quadrant3']='A'
        #df.loc[(df['ys']<df['x_s'])&(df['ys']<-df['x_s']),'quadrant3']='P'
    elif 'rho' in df.columns:
        df.loc[(df['rho'] < 0),'quadrant3']= 'pl'#'PL'
        df.loc[(df['rho'] > 0),'quadrant3']= 'al'#AL'
        df.loc[(df['rho'] < -45),'quadrant3']= 'pp'#'P'
        df.loc[(df['rho'] > 45),'quadrant3']= 'aa'#'A'
    elif 'rho_bin' in df.columns:
        df.loc[(df['rho_bin'] < 0),'quadrant3']= 'pl'#'PL'
        df.loc[(df['rho_bin'] > 0),'quadrant3']= 'al'#'AL'
        df.loc[(df['rho_bin'] < -45),'quadrant3']='pp'#'P'
        df.loc[(df['rho_bin'] > 45),'quadrant3']= 'aa'#'A'
    
    else:
        print('Cannot calculate quadrant3 var!') 
    return df

def bi_quad_var(df, mode = 'quadrant'):
    
    df['bi_quad'] = 'a&l'
    df['bi_quad'] = pd.Categorical(df['bi_quad'], ['a&l', 'pp'], ordered = True)
    if mode == 'quadrant': 
        df.loc[(df['quadrant'] == 'posterior'),'bi_quad']='pp'
    
    elif mode == 'quadrant3':
        df.loc[(df['quadrant3'] == 'pp'),'bi_quad']='pp'
        #df.loc[(df['quadrant3'] == 'pl'),'bi_quad']='posterior'
        
    else:
        print('Cannot calculate bi_quad var!')
    return df
        

def azi_quadrant_var(df, quad = 'quadrant3'):
    quads = df[quad]
    
    if quad == 'quadrant3':
        
        angles = np.arange(np.pi*1/2, -np.pi*3/4, -np.pi*1/4)
        start_angles = [ang for ang in angles[:-1]]
        stop_angles = [ang for ang in angles[1:]]
        
        #conditions  = [ quads == 'A', quads == 'AL', quads == 'PL', quads == 'P']
        conditions  = [ quads == 'aa', quads == 'al', quads == 'pl', quads == 'pp']
        choices_start  = start_angles
        choices_stop  = stop_angles
        df['quad_start'] = np.select(conditions, choices_start, default=np.nan)
        df['quad_stop'] = np.select(conditions, choices_stop, default=np.nan)
    
    if quad == 'regio':
        
        angles = np.arange(np.pi*1/2, -np.pi*3/4, -np.pi*1/4)
        start_angles = [ang for ang in angles[:-1]]
        stop_angles = [ang for ang in angles[1:]]
        
        #conditions  = [ quads.str.contains('A_'), quads.str.contains('AL_'), quads.str.contains('PL_'), quads.str.contains('P_')]
        conditions  = [ quads.str.contains('aa_'), quads.str.contains('al_'), quads.str.contains('pl_'), quads.str.contains('pp_')]
        choices_start  = start_angles
        choices_stop  = stop_angles
        df['quad_start'] = np.select(conditions, choices_start, default=np.nan)
        df['quad_stop'] = np.select(conditions, choices_stop, default=np.nan)
        
        
    return df

def region_var(df, 
               #partition_var = 'quadrant3',
               rho_var = 'dist_c',
               dist_var = 'iso_A',
               ):
    
    
    
    if dist_var == 'iso_A':
        n_Inj_step = 1#len(cfg.var_order[partition_var])


        inj_A = np.pi*(37.5**2)
        A_step = n_Inj_step*inj_A
        A_=np.arange(0,inj_A*12,A_step)
        #radii=list(((A_/(np.pi)))**(1/2))#((2*A_/(m.pi)))**(1/2)
        radii=list(((A_/(np.pi)))**(1/2))
        display(radii)
        edges = radii.copy()
        #edges.insert(0, -1)
        #edges.append(1000)
        names = [f'_{n:.0f}' for n, i in enumerate(radii[:-1])]
    else: 
        radii = np.arange(0,110,10)
        display(radii)
        edges = radii.copy()
        names = [f'_{i:.0f}' for n, i in enumerate(radii[:-1])]
        
    
    df.loc[:, 'region'] = pd.cut(df[rho_var], bins = edges, right = True, labels = names)#.astype('float64')
    
    return df



def regio_var(df,
              reg_var1 = 'quadrant3',
              reg_var2 = 'region',
              ): 

    n_vars = 0
    for var in reg_var1, reg_var2:
        if var in df.columns:
            n_vars += 1
        else:
            if var == 'quadrant3': 
                df = quadrant3_var(df)
                n_vars += 1
                
            elif var == 'region': 
                df = region_var(df)
                n_vars += 1
        
            else: print(f'var {var} not in df.columns')
    if n_vars == 2:
        df['regio'] = df[reg_var1].astype(str) + df[reg_var2].astype(str)
        #df.loc[df.dist_c < 37.5, 'regio'] = 'ii'
        if len(df.iloc[0]['region']) == 2:
            df['regio'] = pd.Categorical(df['regio'], cfg.var_order['regio'], ordered = True)
        else:
            df['regio'] = pd.Categorical(df['regio'], cfg.var_order['regio2'], ordered = True)
            
    return df

def inside_injury_var(df): 
    if 'position' in df.columns:
        df['inside_injury']=df.position.isin(['head'])
    else:
        if not 'dist_c' in df.columns:
            df=dist_c_var(df)
        df['inside_injury']=df.dist_c<37.5
    df['inside_injury'] = pd.Categorical(df['inside_injury'], cfg.var_order['inside_injury'])

    return df

def diff_var(df, var = 'dv'):
    #dfi = df.sort_values(by=['inh','exp_id','particle','frame'])
    dfi = df.sort_values(by=['frame'])
    diff = dfi.groupby(['inh','exp_id','particle'])[var].diff()
    df[f'{var}_diff'] = diff
    return df
    
def deceleration_var(df, var = 'dv'):
    
    if f'{var}_diff' not in df.columns:
        df = diff_var(df, var = var)
    decel = -df[f'{var}_diff']
    df[f'deceleration'] = decel
    
    return df
    
def sliding_var(df, var = 'dvy_s'):
    df['sliding'] = np.where(df[var]<0, -df[var], np.nan)

def cont_var(df):
     df['cont']=((-df['x_s'])*df['dvx'] + (-df['ys'])*df['dvy'] + (-df['zf'])*df['dvz'] 
     )/(((df['x_s'])**2 + (df['ys'])**2 + (df['zf'])**2)**0.5)
     return df

def cont_xy_var(df):
     df['cont_xy']=((-df['x_s'])*df['dvx'] + (-df['ys'])*df['dvy']
     )/((df['x_s'])**2 + (df['ys'])**2)**0.5
     df['cont_x']=((-df['x_s'])*df['dvx'])/((df['x_s'])**2)**0.5
     df['cont_y']=((-df['ys'])*df['dvy'])/((df['ys'])**2)**0.5
     return df


def cum_cont_var(df):
    vars=[]
    cont_vars=['cont','dvz','cont_xy','cont_x','cont_y']
    cont_s_vars=['cont_s','dvz_s','cont_xy_s','cont_x_s','cont_y_s']
    for var,var_s in zip(cont_vars,cont_s_vars):
        if var in df.columns:
            vars.append(var)
        elif var_s in df.columns:
            vars.append(var_s)
    vars_cum=[var+'_cum' for var in vars]

    df=df.set_index(['path','particle','tracknr']).sort_index().reset_index()
    cumsum=df.groupby(['path','particle'])[vars].cumsum().rename(columns={var:var_cum for var,var_cum in zip(vars,vars_cum)})
    #cumsum=df.groupby(['path','particle'])[vars].cumsum().rename(columns={'dvz':'dvz_cum','cont':'cont_cum','cont_xy':'cont_xy_cum'})
    df=pd.concat([df,cumsum],axis=1)
    return df


def mov_class_var(df):#New definition 191209
    for exp_id in pd.unique(df.exp_id):
        dfi=df[df.exp_id==exp_id].copy()
    try:
        still=pd.unique(dfi[((dfi.displ_tot/dfi.nrtracks)<0.1)&(dfi.displ_tot<4)].particle)
        loose=pd.unique(dfi[(dfi.displ_tot>5)&((dfi.cont_tot/dfi.displ_tot)<0.2)].particle)
        contractile=pd.unique(dfi[((dfi.cont_tot/dfi.displ_tot)>0.5)&(dfi.displ_tot>1)].particle)
    except TypeError:
        print(exp_id,dfi.displ_tot.dtypes,dfi.nrtracks.dtypes,(dfi.displ_tot/dfi.nrtracks))
    df.loc[(df.exp_id==exp_id)&(df['particle'].isin(still)),'mov_class']="still"
    df.loc[(df.exp_id==exp_id)&(df['particle'].isin(loose)),'mov_class']="loose"
    df.loc[(df.exp_id==exp_id)&(df['particle'].isin(contractile)),'mov_class']="contractile"
    return df

def movclass_timebin_var(dfg,tr_thr):
    #dfg=dfg.reset_index()
    #still=pd.unique(dfg[(dfg.dv<3)&(dfg.cont<1)].particle)
    #loose=pd.unique(dfg[(dfg.dv>5)&(dfg.dvy<-2)&(dfg.cont<0)].particle)
    #contractile=pd.unique(dfg[((dfg.cont/dfg.dv)>0.5)&(dfg.dv>5)].particle)
    
    dfg.loc[:,'mov_phase']='none'
    dfg.loc[((dfg.dvxy_tot<3))&(dfg.t_nrtracks>tr_thr)&(dfg.zs<6),'mov_phase']="still"#&(dfg.cont_tot<1)
    dfg.loc[(dfg.dvxy_tot>6)&(dfg.dvy_tot<-4)&(dfg.cont_tot<0)&(dfg.t_nrtracks>tr_thr),'mov_phase']="loose"
    dfg.loc[((dfg.cont_tot/dfg.dvxy_tot)>0.5)&(dfg.dvxy_tot>5)&(dfg.t_nrtracks>tr_thr),'mov_phase']="contractile"
    return dfg.reset_index()

def movesum_var(pc1):
    pc1=pc1.reset_index().set_index(['path','particle'])

    #Grupperar data pa path och partikel for att kunna analysera olika partiklar for sig
    grouped=pc1.groupby(['path','particle'])
    #Adderar de olika variablerna och beraknar summan av alla observationer
    summed=grouped.sum()
    #cont_tot summerar den totala kontraktionen for en partikel
    pc1['cont_tot']=summed.cont
    # displ_tot summerar den totala forflyttningen for en partikel
    pc1['displ_tot']=abs(summed.dvx)+abs(summed.dvy)
    # dvz_tot summerar den totala forflyttningen i z-led for en partikel
    pc1['dvz_tot']=summed.dvz
    # dvz_tot summerar den totala forflyttningen i y-led for en partikel
    pc1['dvy_tot']=summed.dvy
    pc1 = pc1.reset_index()#.set_index('pid')
    return pc1

def movesum_timebin_var(df):
    tr_thr=5
    df=df.set_index('frame').sort_index().reset_index().set_index(['inh','exp_id','particle','time_bin']).sort_index()
    dfg=df.groupby(['inh','exp_id','particle','time_bin'])#[['pid']].count().reset_index()
    dfg_rank=dfg.rank()
    df['t_tracknr']=dfg_rank.frame
    dfg_count=dfg.count()
    df['t_nrtracks']=dfg_count.frame
    dfg_sum=dfg.sum()
    df['cont_tot']=dfg_sum.cont
    df['dvxy_tot']=abs(dfg_sum.dvx)+abs(dfg_sum.dvy)
    df['dvz_tot']=dfg_sum.dvz
    df['dvy_tot']=dfg_sum.dvy
    #df=movclass_timebin_var(df,tr_thr).reset_index()
    return df

def movement_var(df):
    df['movement']='none'
    df.loc[(df.dv<0.1) & (df.tracked),'movement']='immobile' #pc.loc[(pc.dv)<0.3,'movement']='still'
    df.loc[(abs(df.dvy/df.dv)>0.5) & (df.dvy<0),'movement']='drifting' #pc.loc[(pc.dv>0.3)&(pc.cont_p<0.5),'movement']='drifting'
    df.loc[(df.dv>0.3) & (df.cont_p>0.5) ,'movement']='contracting'
    df.loc[(df.stab>3),'movement']='unstable'
    return df

def rolling_blackman_var(df,var):
    dfi=df.set_index(['inh','exp_id','particle','frame']).sort_index().reset_index().set_index(['inh','exp_id','particle'])    
    dfg=dfi.groupby(['inh','exp_id','particle'])[var].rolling(window=5,win_type='blackman',min_periods=1,center=True).mean()
    #grp_ls_=[]
    #for index,gr in dfg:
    #    roll=gr.rolling(window=5,win_type='blackman',min_periods=1,center=True).mean()
    #    grp_ls_.append(roll)
    #var_sr=pd.concat(grp_ls_,axis=0)
    dfi[f'{var}_roll']=dfg
    dfi=dfi.reset_index().set_index('pid').sort_index().reset_index()
    return dfi
    
def rolling_mean_var(df,var):
    dfi=df.set_index(['inh','exp_id','particle','frame']).sort_index().reset_index().set_index(['inh','exp_id','particle'])    
    dfg=dfi.groupby(['inh','exp_id','particle'])[var]
    grp_ls_=[]
    for index,gr in dfg:
        roll=gr.rolling(window=5,min_periods=1,center=True).mean()
        grp_ls_.append(roll)
    var_sr=pd.concat(grp_ls_,axis=0)
    dfi[f'{var}_roll']=var_sr
    dfi=dfi.reset_index().set_index('pid').sort_index().reset_index()
    return dfi

def fibrin_var(df):
    dfi=df.set_index(['inh','exp_id','particle','frame']).sort_index().reset_index().set_index(['inh','exp_id','particle'])    
    dfg=dfi.groupby(['inh','exp_id','particle'])['c1_mean']
    grp_ls_=[]
    for index,gr in dfg:
        roll=gr.rolling(window=9,min_periods=1,center=False).mean()
        grp_ls_.append(roll)
    var_sr=pd.concat(grp_ls_,axis=0)
    dfi['fib_roll']=var_sr
    dfi['fib_diff']=dfi['c1_mean']-dfi['fib_roll']
    dfi['relfibdiff']=dfi['fib_diff']/dfi['fib_roll']
    dfi=dfi.reset_index().set_index('pid').sort_index().reset_index()
    return dfi

def rolling_mov_var(df):
    movements=['dv','dvx','dvy','dvz','cont']
    dfi=df.set_index(['inh','exp_id','particle','frame']).sort_index().reset_index().set_index(['inh','exp_id','particle'])    
    dfg=dfi.groupby(['inh','exp_id','particle'])
    grp_ls_=[]
    for index,gr in dfg:
        roll=gr[movements].rolling(window=5,win_type='blackman',min_periods=1,center=True).mean()
        grp_ls_.append(roll)
    var_sr=pd.concat(grp_ls_,axis=0)
    for mov in movements:
        dfi[f'{mov}_roll']=var_sr[mov]
    dfi=dfi.reset_index().set_index('pid').sort_index().reset_index()
    return dfi


## Scaling

def scale_var(df,var1):
    var2=var1+'_s'
    df.loc[:,var2]=df[var1].copy()*1000/3.1
    print(var2)
    return df

def scale_vars(df):
    vars=[var for var in cfg.scale_vars if var in df.columns]
    for var in vars:
        df=scale_var(df,var)
    return df

def discretize_vars(df, vars = ['fsec', 'ca_corr','c1_mean'], bin_nr = 20):
    for var in vars:
        max = df[var].max()
        min = df[var].min()
        
def area_vars(dfip, var = 'dv', lim1 = 5, lim2 = 150):
    #dfip = dfip.copy()
    print('Input variable:', var)
    lim_ = sorted([lim1, lim2])

    if var in cfg.rankCatVars.keys():
        run = True
        new_var = cfg.rankCatVars[var]
        print('Output variable:', new_var)
        if new_var in cfg.posCatVars_:
            print('pos sign')
            sign = 'pos'
        elif new_var in cfg.negCatVars_:
            print('neg sign')
            sign = 'neg'
        else:
            print('no sign')
            run = False
    else:
        print('var not in catVarDic dictionary')
        run = False 
    
    if run:
        dfip = make_cat_var(dfip, var, new_var, lim_, sign)
        return new_var, dfip.dropna()
    else:
        print('cannot caculate new variable')
            
def make_cat_var(dfi, old_var, new_var, lims, sign = 'pos'):
    print(lims)
    old_vals = dfi[old_var]
    
    if len(lims) == 2:
        choices = list(reversed(cfg.catRanks_))
        #choices     = [ "high", 'medium', 'low' ]
        conditions  = [ old_vals >= lims[1], (old_vals < lims[1]) & (old_vals > lims[0]), old_vals <= lims[0]]
        if sign == 'pos':
            new_vals = np.select(conditions, choices, default=np.nan)
        else: 
            new_vals = np.select(conditions, list(reversed(choices)), default=np.nan)
    elif len(lims) == 1:
        choices     = list(reversed(cfg.catRanks_[0::2]))
        conditions  = [ old_vals >= lims[0], old_vals < lims[0]]
        if sign == 'pos':
            new_vals = np.select(conditions, choices, default=np.nan)
        else: 
            new_vals = np.select(conditions), list(reversed(choices), default=np.nan)
    else:
        print(f'You can only have one or two limits')
    
    dfi[new_var] = new_vals
        
    return(dfi)

    

def binning_labels_var(dfg,binned_var,bins):
    #bin_labels=
    dfg[binned_var]=pd.cut(dfg[binned_var],bins,precision=0)
    bin_var=f'{binned_var}_binlabel'
    bin_labels=[]
    for bin in dfg[binned_var].sort_values().unique():
        bin_label=str(np.round(np.mean((bin.right+bin.left)/2),0))
        bin_labels.append(bin_label)
        #print (bin,bin_label)
        dfg.loc[dfg[binned_var]==bin,bin_var]=bin_label
    bin_order = sorted(bin_labels, key=lambda x: float(x))
    return dfg,bin_var,bin_order

def qbinning_labels_var(dfg,binned_var,bins):
    dfg[binned_var]=pd.qcut(dfg[binned_var],bins,precision=0)
    bin_var=f'{binned_var}_binlabel'
    bin_labels=[]
    for bin in dfg[binned_var].sort_values().unique():
        bin_label=str(int(np.round(np.mean((bin.right+bin.left)/2),0)))
        bin_labels.append(bin_label)
        #print (bin,bin_label)
        dfg.loc[dfg[binned_var]==bin,bin_var]=bin_label
    bin_order = sorted(bin_labels, key=lambda x: float(x))
    return dfg,bin_var,bin_order

def qbinning_quant(dfg,binned_var,bins):
    quint=np.arange(1,bins+1,1)
    labels=[]
    for n in quint:
        labels.append(str(n*10))#labels.append(str(n)+'th')
    dfg[binned_var]=pd.qcut(dfg[binned_var],bins,labels=labels,precision=0)
    return dfg,binned_var,labels

def new_exp_ids(pc):
    pc["exp_id"] = pc.groupby(["path"]).ngroup()
    for inh in pc.inh.unique():
        pc.loc[pc.inh==inh,'inh_exp_id']=pc[pc.inh==inh].groupby('path').ngroup()#grouper.group_info[0]
    return pc

def depth_N(df_s,fill_dist=10,track_thr=1, zfloor = 4):#fill_dist=10

        
    x_='x_s'
    y_='ys'
    z_='zs' 
    xyz = [x_, y_, z_]
    
    # Find max and min coordinates of thrombi
    pos_max = np.round(df_s[xyz].abs().max(),0).astype('int')
    pos_min = np.round(df_s[xyz].min(),0).astype('int')#np.round(pos.min().abs(),0).astype('int')
    print(pos_min)
    
    xsize=int(pos_max[x_] - pos_min[x_] + 1)
    ysize=int(pos_max[y_] - pos_min[y_] + 1)
    zsize=int(pos_max[z_] - pos_min[z_] + 5)
    print('Dimensions:',xsize,ysize,zsize)
     # Recalculate coordinates to remove negative values
    for coord in xyz:
        df_s.loc[:,coord] = df_s[coord] - pos_min[coord]

    #Group df by frames
    depth_grp_=[]

    frames = [180, 190]#[10, 20, 50, 100, 150]#[151,152,153]#
    nrows = len(frames)
    ncols = 2

    #---------------------------------------------------------------------------
    # 230422 NEW LINES INSERTING LISTS TO OBTAIN DATA FROM SURFACE RECONSTRUCTION 
    #paths_ = []
    #zero_depth = []
    #pos_depth = []
    #depth = []
    c0_ = []#len(pcad)
    #cneg_ = []
    nonzero_ = []#len(pcad > 0)
    more1_ = []
    more2_ = []
    more10_ = []#len(pcad > 10)
    frame_ = []
    path_ = []
    #---------------------------------------------------------------------------
    
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*6,2*nrows), constrained_layout=True)
    path =  df_s.path.unique()[0]
    print('Path: ',path)
    print('Frame number:')
    for i, dfi in df_s.groupby('frame'):#df_s[df_s.frame.isin(frames)].groupby('frame'):
        print('Frame:',i)
        pos_all = dfi.copy()[xyz]
        pos = dfi[dfi.nrtracks>track_thr].copy()[xyz]#dfi[dfi.nrtracks>track_thr].copy()[xyz]#dfi[xyz]#
        # Make array with coordinates for all platelets (intgers)
        pos_all=np.round(pos_all,0).astype('int')
        pos=np.round(pos,0).astype('int')
        
        pc_pos_all = pos_all.values.T.tolist()
        pc_pos=pos.values.T.tolist()
        # CONSTRUCT NUMPY ZERO ARRAY OF SAME SIZE AS THROMBI
        

        pcc=np.zeros((xsize,ysize,zsize))
        

        # FILL ALL ELEMENTS OCCUPIED BY PLATELETS WITH ONES        
        pcc[tuple(pc_pos)]=1

        # FILL THE VOLUME UNDER THE PLATELETS WITH ONES 
        
        x_max = int(pos.loc[pos[z_]<zfloor+6,x_].quantile(0.9))#max()
        x_min = int(pos.loc[pos[z_]<zfloor+6,x_].quantile(0.1))#min()
        y_max = int(pos.loc[pos[z_]<zfloor+6,y_].quantile(0.9))#.#max()
        y_min = int(pos.loc[pos[z_]<zfloor+6,y_].quantile(0.1))#min()
        pcc[x_min:x_max,y_min:y_max,:zfloor]=1

        # FIRST EUCLIDIAN DISTANCE TRANSFORM ON EMPTY ELEMENTS, CALCULATES HOW FAR OUTSIDE OF THE THROMBUS AN ELEMENT IS  
        pcd=ndimage.morphology.distance_transform_edt(pcc==0)
        # BOOLEAN ARRAY SETS ALL ELEMENTS THAT ARE CLOSER THAN FILL DIST TO TRUE  
        pca=pcd<fill_dist

        # FILLS ALL HOLES IN ARRAY TO ENCLOSE THROMBUS
        pca=ndimage.binary_fill_holes(pca)
        pca_i = ~pca# == False
        #plt.figure(figsize=(10,3))
        #plt.imshow((pca[pos_min[x_],:,:]).T, cmap='jet', vmin=0, vmax=2)
        #plt.colorbar()
        #plt.show()

        # NEW DISTANCE TRANSFORM CALCULATES DISTANCE TO SURFACE 
        pcad=ndimage.morphology.distance_transform_edt(pca)
            #print(abs(pos_min[x_]),abs(pos_min[y_]))
        pcadi=ndimage.morphology.distance_transform_edt(pca_i) #+ fill_di
        
        # SUBTRACTS THE "MASK" (FILL DISTANCE) FROM DEPTH 
        pdepth_p = pcad[tuple(pc_pos_all)] -fill_dist
        pdepth_i = pcadi[tuple(pc_pos_all)]
        pdepth = pdepth_p - pdepth_i
        
            
        depth_grp=pd.DataFrame({'depth' : (pdepth)})
        
        depth_grp[['path', 'particle', 'frame', 'pid']]=dfi.reset_index()[['path', 'particle', 'frame', 'pid']]
        depth_grp_.append(depth_grp)
        #print(np.round((abs(pos_min[x_]) + abs(pos_max[x_]))/2),0)
        
        #---------------------------------------------------------------------------
        # 230422 NEW LINES INSERTING LISTS TO OBTAIN DATA FROM SURFACE RECONSTRUCTION 

        c0_.append(np.count_nonzero(pcad == 0))
        #cneg_.append(np.count_nonzero(pcadi  0))
        nonzero_.append(np.count_nonzero(pcad > 0))
        more1_.append(np.count_nonzero(pcad > 1))
        more2_.append(np.count_nonzero(pcad > 2))
        more10_.append(np.count_nonzero(pcad > 10))
        frame_.append(i)
        path_.append(path)
        
        #unique, counts = np.unique(np.round(pcad,0), return_counts=True)
        #count_vals = dict(zip(unique, counts))
        #c_depth = pd.DataFrame({'path':df_s.path.iloc[0], 'depth_val': unique,'counts': counts})
        
                #---------------------------------------------------------------------------
        
        
        #---------------------------------------------------------------------------
        # Plot function
        if i in frames:
            planes = {
                    'xz' : (pcad[np.round((abs(pos_min[x_])+ abs(pos_max[x_]))/2,0).astype('int'),:,:]).T,#(pcad[0,:,:]).T,
                    'yz' : (pcad[:,np.round((abs(pos_min[y_])+ abs(pos_max[y_]))/2,0).astype('int'),:]).T,
                    }
            #ncols = len(planes.keys())
            for col, plane in enumerate(planes):
                cax = axs[frames.index(i),col].imshow(planes[plane], cmap='jet', vmin=0, vmax=20, origin='lower')
            #axs.set_title(plane)
                #plt.title(plane)
                #axs.invert_yaxis()
                

                axs[frames.index(i),col].text(0.05, 0.05, f'Plane:{plane}\nFrame:{i}', horizontalalignment='left',verticalalignment='top',color = 'white',fontsize = 14)
                
                if i == frames[-1]:
                    cbar = plt.colorbar(cax)
            fig.suptitle(path)
                #axs[frames.index(i),col].annotate(plane, xy=(0.8, 0.95), color='white',  xycoords='axes fraction',horizontalalignment='left', verticalalignment='top',fontsize=20)#xytext=(0.8, 0.95), textcoords='axes fraction',
        #-------------------    --------------------------------------------------------
    #plt.show()
    #plt.colorbar()
    
    c_depth = pd.DataFrame({'path':path_, 'frame':frame_, 'count_0': c0_,'c_more1': more1_, 'c_more2': more2_, 'c_not0':nonzero_, 'c_more10': more10_})
    df_depth=pd.concat(depth_grp_, axis=0)
    df_depth = df_depth.merge(c_depth, on = ['path','frame'])
    return df_depth#, c_depth
        
def average_neighbour_distance(pc):
    t_grp=pc.set_index('pid').groupby(['path', 'frame']).apply(_nearest_neighbours_average)
    pc = pd.concat([pc.set_index('pid'), t_grp.set_index('pid')], axis=1).reset_index()
    return pc
        
def _nearest_neighbours_average(pc):
    from scipy import spatial
    nb_count=3
    nba_list=[5,10,15]
    key_dist={}
    key_idx={}
    #print(len(pc))
    p1i=pc.reset_index().pid
    if len(pc)>np.array(nba_list).max():
        dmap=spatial.distance.squareform(spatial.distance.pdist(pc[['x_s','ys','zs']].values))
        dmap_sorted=np.sort(dmap, axis=0)
        #dmap_idx_sorted=np.argsort(dmap, axis=0)
        for i in nba_list:
            nb_dist=(dmap_sorted[1:(i+1),:]).mean(axis=0)
            #nb_idx=dmap_idx_sorted[i+1,:]
            key_dist['nba_d_' + str(i)]=nb_dist
            #key_idx['nb_i_' + str(i)]=pd.Series(nb_idx).map(p1i).values.astype('int').tolist()
        #key_idx.update(key_dist)
    else:
        a = np.empty((len(pc)))
        a[:] = np.nan
        key_dist[('nba_d_' + str(nba_list[0]))]=a
    df=pd.DataFrame(key_dist)
    df=pd.concat([p1i, df], axis=1)
    return df
        


def path_split_list(ls1):
    # Use path to create variables describing experimental conditions
    
    
    fpos = {'date':0,'mouse':1,'inj':2,'inh':3,'exp':4}
    fstart = {'date':0,'mouse':5,'inj':3,'inh':0,'exp':3}
    df_dic = {'path':ls1}

    for key,pos in fpos.items():
        
        #columns.append(key)
        vals = []
        for path in ls1:
            if pos < (path.count('_') + 1):
                vals.append(path.split('_')[pos][fstart[key]:])
            else:
                vals.append(np.nan)
        df_dic.update({key:vals})
      
    mouse_inj = []      
    for path in ls1:
        mouse_inj.append(path.split('_')[fpos['mouse']][fstart['mouse']:] 
                         + '_' + path.split('_')[fpos['inj']][fstart['inj']:])
    df_dic.update({'mouse_inj':mouse_inj})
    dft = pd.DataFrame(data = df_dic)
    return dft

def pathsplit_list_minj(ls1):
    # Use path to create variables describing experimental conditions
    
    
    fpos = {'date':np.nan,'mouse':np.nan,'inj':np.nan,'inh':np.nan,'exp':np.nan}
    fstart = {'date':0,'mouse':5,'inj':3,'inh':0,'exp':3}
    df_dic = {}
    path1 = ls1[0]
    path_ls = path1.split('_')
    for pos, str in enumerate(path_ls):
        if 'IVMTR' in str:
            fpos.update({'mouse':pos})
        if 'inj' in str:
            fpos.update({'inj':pos})
    
    fpos1 = {}       
    fpos1.update({key:val for key,val in fpos.items() if val})
    
    for key,pos in fpos1.items():
        #columns.append(key)
        vals = []
        for path in ls1:
            #if pos < (path.count('_') + 1):
            #    vals.append(path.split('_')[pos][fstart[key]:])
            #else:
             #   vals.append(np.nan)
            vals.append(path.split('_')[pos][fstart[key]:])
        df_dic.update({key:vals})
    
    if 'mouse' in fpos1.keys() and 'inj' in fpos1.keys():
        mouse_inj = []      
        for path in ls1:
            mouse_inj.append(path.split('_')[fpos['mouse']][fstart['mouse']:] 
                            + '_' + path.split('_')[fpos['inj']][fstart['inj']:])
        df_dic.update({'mouse_inj':mouse_inj})
    dft = pd.DataFrame(data = df_dic)
    return dft

def inh_names_2lines(df):
    longnames_list=['Bivalirudin','Cangrelor','CMFDA','Control',
                 'MIPS','Saline','SQ','Vehicle MIPS',
                 'Vehicle SQ','PAR4+/-','PAR4-/+','PAR4-/- + biva',
                 'PAR4-/-','ASA + Vehicle','ASA','Salgav + Vehicle',
                 'Salgav'
                ]
    inh_dic={'vehicle MIPS':'vehicle\nMIPS', 'salgavDMSO':'salgav\nDMSO', 'vehicle sq':'vehicle\nsq', 'par4--biva':'par4--\nbiva'}

#---------------------------------------------------------------------------
# FUNCTIONS FOR MANIPULATING OTHER DATASTRUCTURES
#---------------------------------------------------------------------------
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#Function for selecting unique elements in list while preserving order
#-------------------------------------------------------------
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

#---------------------------------------------------------------------------
# BUILD DATA STRUCTURES WITH/WITHOUT MENUS 
#---------------------------------------------------------------------------
# Build dataframes for analysis
#---------------------------------------------------------------------------
def build_df_lists(col_list,treatments):#Builds dataframe from lists of variables and list (inh_order) with treatments
    global inh_order,inh_list
    inh_order=treatments
    inh_list=[cfg.longtoshort_dic[inh] for inh in inh_order]
    df_=[]
    paths = Path(cfg.df_paths.path[0]).glob('**/*')
    paths = [x for x in paths if x.is_file()]
    list1=[path for inh in inh_list for path in paths if inh in path.name]
    path_list=list(set(list1))
    for n_df,fi in enumerate(path_list):
        dfi=pd.read_parquet(fi)#dfi=pd.read_pickle(fi)
        dfi=dfi[dfi.frame<194]# Remove rows with data from frames > 193
        dfi_col=[]#'path', 'inh','particle'
        absent_cols=[]
        if 'all_vars' in col_list:
            df_.append(dfi)
        else:
            for col in col_list:
                if col in dfi.columns:
                    dfi_col.append(col)
                else:
                    absent_cols.append(col)      
            df_.append(dfi.loc[:,dfi_col])
        if absent_cols:
            print(f'columns absent in {fi}:{absent_cols}')
    pc=pd.concat(df_, ignore_index=True)
    for inh in pc.inh.unique():
        pc.loc[pc.inh==inh,'inh']=cfg.shorttolong_dic[inh]#.casefold()
    if 'minute' in pc.columns:
        pc.loc[:,'minute'] = pd.cut(pc['time'], 10, labels=np.arange(1,11,1))
    pc=new_exp_ids(pc)
    #pc=pc.drop(['level_0','index'],axis=1).reset_index()
    #print(pc.columns)
    #if 'level_0' in pc.columns:
    #    pc=pc.drop(['level_0'],axis=1)
    #if 'index' in pc.columns:
    #    pc=pc.drop(['index'],axis=1)
   # pc=pc.reset_index()
    if 'pid' in pc.columns:
        pc=pc.drop(columns=['pid'])
    pc.index.name = 'pid'
    pc=pc.reset_index()
    pc=pc.rename(columns={'time':'sec'})
    if 'Demo Injuries' in treatments:
        inh_list=cfg.all_demo_
        inh_order=[cfg.shorttolong_dic[inh] for inh in inh_list]
  #  #clear_output(wait=False)
    print(f'Treatments = {pc.inh.unique()}',flush=True) #Paths included = {pc.path.unique()}\n
    print(f'RESULTING DATAFRAME\nNo of columns: {pc.shape[1]} \nNo of rows: {pc.shape[0]}',
          f'\nMemory use: {np.round(sys.getsizeof(pc)/1000000,1)} Mb',flush=True)
    return pc




def change_inh_order(inh_list):
    global inh_order
    inh_order=inh_list

def reorder_inh_menu():#Function for reordering treatments in list
    global inh_list,inh_order
    names1=[]
    nr_treatments=len(inh_list)
    names=[cfg.shorttolong_dic[inh] for inh in inh_list]
    for nr in range(nr_treatments):
        for n,name in enumerate(names):
            print(n,name)
        print('  ',flush=True)
        choice = int(input('Pick the treatment that will be next in order'))
        names1.append(names[choice])
        names.remove(names[choice])
    inh_order=names1
    inh_list=[cfg.longtoshort_dic[inh] for inh in names1]     


    
def show_expseries():#Function showing the available series for analysis
    print(f'Experimental cohorts available for selection:\n{73 * "-"}\n')
    n=0
    for serie_name,serie in zip(cfg.expseries_listnames,cfg.treatments_):
        #print(f'{n}:{serie_name}\t')
        list_names=[]
        for inh in serie:
            list_names.append(cfg.shorttolong_dic[inh])
        print(f'{n}:{serie_name}\n{list_names}')
        print('\n',flush=True)
        n+=1



def df_outliers_menu(df_outliers,df_inliers):
        df_outliers=df_outliers.loc[:,['measure','inh','path','value']]
        df_inliers=df_inliers.reset_index().loc[:,['measure','inh','path','value']]
        print(df_outliers.to_markdown(),'\n')
        col_vars=input(f'Choose which experiments you want to save as outliers \nEnter your choice as integers separated by spaces\n{73 * "-"}\n')
        flist = [int(x) for x in col_vars.split()]
        df_choice=df_outliers.iloc[flist]
        
        choice = input('Do you want to add additional experiments in the outliers file? (y/n)')
        if choice =='y':
            print(df_inliers.to_markdown(),'\n')
            print('Choose which experiments you want to save as outliers',flush=True)
            col_vars=input(f'Enter your choice as integers separated by spaces\n{73 * "-"}\n')
            flist = [int(x) for x in col_vars.split()]
            df_choice1=df_inliers.iloc[flist]
            print(df_choice1.to_markdown())
            df_choice=pd.concat([df_choice,df_choice1])
        #print(df_choice.to_markdown())
        choice = input('Do you want to save your results in the outliers file? (y/n)')
        if choice =='y':
            df_choice.to_csv('df_outliers.csv')

def xtravars_menu():
    print('One of the following variables can be used to analyse different regions of the thrombus separately:')
    for c, value in enumerate(cfg.thr_reg_vars,0):
        print(c, value) 
    print(' ',flush=True)
    var_nr=int(input("Enter your choice:"))
    var=cfg.thr_reg_vars[var_nr]
    return var

def add_xtravars(df,xtra_vars):
    if isinstance(xtra_vars, dict):
        xtra_var_ls=xtra_vars.values()
    if 'phase' in xtra_var_ls:
        df=phase_var(df)
    if 'injury_zone' in xtra_var_ls:
        df=injury_zone_var(df)
    if 'minute' in xtra_var_ls:
        df=minute_var(df)
    if 'position' in xtra_var_ls:
        df=position_var(df)
    if 'height' in xtra_var_ls:
        df=height_var(df)
    if 'z_pos' in xtra_var_ls:
        df=z_pos_var(df)
    if 'inside_injury' in xtra_var_ls:
        df=inside_injury_var(df)
    return df

def add_xtravar(df,xtra_var):
    #if xtra_var == 'inside_injury':
    #    df=inside_injury_var(df)
    if xtra_var == 'injury_zone':
        df=injury_zone_var(df)
    elif xtra_var == 'minute':
        df=minute_var(df)
    #elif xtra_var == 'position':
    #    df=position_var(df)
    elif xtra_var == 'height':
        df=height_var(df)
    elif xtra_var == 'z_pos':
        df=z_pos_var(df)
    elif xtra_var == 'phase':
        df=phase_var(df)
    elif xtra_var == 'quadrant':
        df=quadrant_var(df)
    elif xtra_var == 'quadrant1':
        print('quadrant1 added')
        df=quadrant1_var(df)
    elif xtra_var == 'quadrant2':
        print('quadrant2 added')
        df=quadrant2_var(df)
    return df

# Calculate max for control line (e.g. for plotting)
#---------------------------------------------------------------------------
def break_max(df):
    dfg=rolling_count(df)
    dfg=dfg.groupby(['inh','sec'])[['roll','diff']].mean().groupby(level=[0]).rolling(window=6,min_periods=1,center=True).mean().droplevel([0]).reset_index()
    dfgi=dfg[dfg.inh==inh_order[0]]
    dfgi.loc[:,'max'] = dfgi.roll[(dfgi.roll.shift(1) < dfgi.roll) & (dfgi.roll.shift(-1) < dfgi.roll)]
    dfg_max=dfgi.loc[dfgi['max']>0,:]
    phase_break=dfg_max.iloc[0,1]
    return phase_break

def growth_phase(df):
    global phase_break
    dfg=rolling_count(df)
    dfgi=dfg.groupby(['inh','sec'])['roll'].mean().groupby(level=[0]).rolling(window=6,min_periods=1,center=True).mean().droplevel([0]).reset_index()

    
    dfgi=dfgi[dfgi.inh==inh_order[0]].copy()
    #print(dfgi)
    dfgi.loc[:,'max'] = dfgi.loc[(dfgi.roll.shift(1) < dfgi.roll) & (dfgi.roll.shift(-1) < dfgi.roll),'roll']
    plt.figure(figsize=(1.5,1.5))
    plt.scatter(dfgi.sec, dfgi['max'], c='g')
    plt.plot(dfgi.sec,dfgi.roll)
    dfg_max=dfgi.loc[dfgi['max']>0,:]
    phase_break=dfg_max.iloc[0,1]
    print('Phase break:',phase_break)
    df['phase']=np.where(df.sec<phase_break,'Growth','Consolidation')
    return df#phase_break


def growth_phase2(dfi):
    #global phase_break
    dfg = rolling_count(dfi)
    dfgi=dfg.groupby(['inh','sec'])['roll'].mean().groupby(level=[0]).rolling(window=6,min_periods=1,center=True).mean().droplevel([0]).reset_index()
    treatments = dfi.inh.unique().tolist()

    for key,inhs in cfg.treatment_orders.items():
        t1 = [treatment for treatment in treatments if treatment in inhs]
        if len(t1)>0:
            print(t1)
            #if any(treat in inhs for treat in treatments):
            dfg1= dfgi[dfgi.inh==inhs[0]].copy()
        #print(dfgi)
            dfg1.loc[:,'max'] = dfg1.loc[(dfg1.roll.shift(1) < dfg1.roll) & (dfg1.roll.shift(-1) < dfg1.roll),'roll']
            plt.figure(figsize=(3,3))
            plt.scatter(dfg1.sec, dfg1['max'], c='g')
            plt.plot(dfg1.sec,dfg1.roll)
            dfg_max=dfg1.loc[dfg1['max']>0,:]
            phase_break=dfg_max.iloc[0,1]
            print('Phase break:',phase_break)
            dfi.loc[dfi.inh.isin(inhs),'phase']=np.where(dfi.loc[dfi.inh.isin(inhs),'sec']<phase_break,'Growth','Consolidation')
    return dfi#phase_break



def obs_weight_var(dfi1, coeff = False, grouper = ['inh', 'tri_sec',], var1 = 'path', var2 = 'sec'):
    #Calculating observation weights for use e.g. in ecdf plots 'path' and 'frames'
    weights1 = dfi1.groupby(grouper, observed= True)[var1].nunique().rdiv(1).rename('weight')
    if len(grouper) > 1:
        weights2 = dfi1.groupby(grouper, observed= True)[var2].nunique().rdiv(1).rename('weight1')
        weights = weights1.mul(weights2).rename('weight')
    else:
        weights = weights1
        
    if coeff: 
        weights = weights.mul(coeff)
    dfi1 = dfi1.merge(weights, on = grouper)
    return dfi1

# Create grouped dataframes (e.g. for plotting)
#---------------------------------------------------------------------------
def rolling_count(df,hue_var=False,x_var='sec', window = 8): #,ctrl=False##Grouped df with rolling counts, difference in 'diff' column 
    
    treatment_var=['inh']
    exp_var=['path']

    group_id = treatment_var 
    #levels=[0,1]
    #timemeanvars=treatment_var + x_var
    if hue_var:
        for hue in hue_var:
            if hue not in ['inh','path']:
                group_id.append(hue)
        #group_id += hue_var #+ exp_var#'inh_exp_id'
        #levels=[0,1,2]
    exp_id=group_id + exp_var 
    group_time= group_id + [x_var]

    g_levels=list(range(len(exp_id)))
    levels2=g_levels[:-1]
    exp_time=exp_id+[x_var]
    print(exp_time)
    #if x_var!=['sec']:
    #    grouping1=grouping1+['sec']
    df_gr=df.groupby(exp_time, observed = True).size()
    
    df_roll=df_gr.groupby(level=g_levels).rolling(window=window, min_periods=1,center=False).mean().droplevel(g_levels)#,win_type='bartlett',min_periods=1,center=True
    
    df_diff=df_roll.groupby(level=g_levels).diff()
    df_grouped=df_gr.to_frame().rename(columns={0:'count'})
    df_grouped['roll']=df_roll#.reset_index()['ts']
    df_grouped['diff']=df_diff#.reset_index()['ts']
    df_grouped=df_grouped.reset_index()
        #df_grouped['inh_per']=df_inh_day['roll']
    
    return df_grouped#,df_inh

def rolling_mean(df,hue_var=False,x_var='sec', y_var = 'ca_corr', window = 8): #,ctrl=False##Grouped df with rolling counts, difference in 'diff' column 
    
    treatment_var=['inh']
    exp_var=['path']

    group_id = treatment_var 
    #levels=[0,1]
    #timemeanvars=treatment_var + x_var
    if hue_var:
        for hue in hue_var:
            if hue not in ['inh','path']:
                group_id.append(hue)
        #group_id += hue_var #+ exp_var#'inh_exp_id'
        #levels=[0,1,2]
    exp_id=group_id + exp_var 
    group_time= group_id + [x_var]

    g_levels=list(range(len(exp_id)))
    levels2=g_levels[:-1]
    exp_time=exp_id+[x_var]
    print(exp_time)
    #if x_var!=['sec']:
    #    grouping1=grouping1+['sec']
    df_gr=df.groupby(exp_time, observed = True).mean()[y_var]#.rename(columns={'pid':'plts'})#
    #df_grouped=df_grouped.reset_index()
    #Second grouping without time 
    #dfg=df_grouped.groupby(grouping_vars)[[y_var]].rolling(window=8,min_periods=1,center=True).mean().reset_index()
    df_roll=df_gr.groupby(level = g_levels).rolling(window=window,min_periods=1,center=False).mean().droplevel(g_levels).to_frame().rename(columns = {0: y_var}).reset_index()
    

    return df_roll#,df_inh

def per_inh(dfi,hue_var=False,x_var=['sec'],y_var='roll',ctrl=inh_order[0]): ##Grouped df with rolling counts, difference in 'diff' column 
    treatment_var=['inh']
    exp_var=['path']
    group_id = treatment_var 
    exp_id = group_id + exp_var
    
    if hue_var:
        group_id += hue_var #+ exp_var#'inh_exp_id'
        exp_id += hue_var
        dfg=rolling_count(dfi,hue_var=hue_var)
    else:
        dfg=rolling_count(dfi)

    group_time = group_id + x_var
    exp_time = exp_id + x_var 
    dfg=dfg.set_index(exp_time)[[y_var]]#drop = True
    dfs=dfg[y_var]
    #dfg
    grp_ctrl=dfs.groupby(level=group_time).mean().loc[ctrl]#.set_index(exp_time)[y_var]#.droplevel(group_time)#.reset_index(drop=True)
    #dfg=dfg.set_index
    perc_inh=(((grp_ctrl-dfs)/grp_ctrl)*100).reorder_levels(exp_time)#.set_index(exp_time)
    inh_ratio=(dfs/grp_ctrl).reorder_levels(exp_time)
    dfg['per_inh']=perc_inh
    dfg['inh_ratio']=inh_ratio
    

    return dfg#,perc_inh#,dfg

    
def per_reg_inh(dfi,hue_var=['quadrant2'],x_var=['sec'],y_var='roll',ctrl=inh_order[0]): ##Grouped df with rolling counts, difference in 'diff' column 
    #print(group_id, '\n\n\n\n',exp_id, '\n\n\n\n', exp_time)
    #print(dfi.head())
    treatment_var=['inh']
    exp_var=['path']
    

    dfg1=per_inh(dfi,x_var=x_var,y_var=y_var,ctrl=ctrl)
    dfg1['quadrant2']='all'
    dfg1_mean=dfg1.groupby(level=treatment_var+x_var).mean()

    dfg2 = per_inh(dfi, hue_var=hue_var, x_var= x_var, y_var=y_var, ctrl=ctrl)
    dfg2=dfg2.reset_index(hue_var)


    new_index=treatment_var+x_var+exp_var+hue_var
    print(new_index)
    dfg2 = pd.concat([dfg2,dfg1],axis=0).reset_index().set_index(new_index)
    #dfg2 = pd.concat([dfg2,dfg1],axis=0).reset_index().set_index(['inh','sec','path','quadrant2'])
    rel_inh=(1-(dfg2['inh_ratio']/dfg1_mean['inh_ratio']))*100#.reset_index()
    dfg2['rel_inh']=rel_inh#['inh_ratio']
    dfg2=dfg2.reset_index()
    return dfg2


def rolling_perc2(df1, df2, grouping_var=False,x_var=['sec']):
    df1=rolling_count(df1,hue_var=grouping_var,x_var=x_var)
    df2=rolling_count(df2,hue_var=grouping_var,x_var=x_var)
    df_per=df1.roll/df2.roll
    #df1['roll']=df_per
    return df_per#['roll']


def rolling_count_old(df,grouping_var,x_var): ##Grouped df with rolling counts, difference in 'diff' column 
    dv_=[]
    dg_=[]
    grouping_vars=['inh','path']
    if grouping_var:
        grouping_vars=[grouping_var]+grouping_vars#'inh_exp_id'
    grouping1=grouping_vars+[x_var]
    print(grouping1)
    #if x_var!=['sec']:
    #    grouping1=grouping1+['sec']
    df_grouped=df.groupby(grouping1)['pid'].size().reset_index()
    #df_grouped=df.groupby(grouping1).count()[['pid']]#.rename(columns={'pid':'plts'})#
    #df_grouped=df_grouped.reset_index()
    df_roll=df_grouped.groupby(grouping_vars)['pid'].rolling(window=8, min_periods=1,center=True).mean().reset_index()
    #df_roll=df_grouped.groupby(grouping_vars)['pid'].rolling(window=6,win_type='bartlett',min_periods=1,center=True).mean().reset_index()#.droplevel(level=0)
    df_diff=df_roll.groupby(grouping_vars)['pid'].diff()
    #dfg=df_grouped.groupby(level=[0,1,2])['pid'].
    df_grouped['roll']=df_roll['pid']*40
    df_grouped['diff']=df_diff*40
    #for i,gr in dfg:
    #
    #    df2=gr[['pid']].rolling(window=6,win_type='bartlett',min_periods=3,center=True).mean()#.\
    #    dv_.append(df2)
    #    df_gr=df2.diff()
    #    dg_.append(df_gr)
    #dv=pd.concat(dv_, axis=0)
    #df_grouped['roll']=dv*40
    #d_gr=pd.concat(dg_, axis=0)
    #df_grouped['diff']=d_gr*40
    if x_var==['minute']:
        grouping2=grouping_vars+[x_var]
        df_grouped=df_grouped.groupby(grouping2).mean()[['roll']].reset_index()
    return df_grouped

#def rolling_mean(df,grouping_var,y_var,x_var): #Rolling mean values for variable y_var 
#   dv_=[]
#    #x_var=['sec']
#    if grouping_var=='inh':
#        grouping_vars=[grouping_var]+['inh_exp_id']
#    else:
#        grouping_vars=[grouping_var]+['inh','inh_exp_id']
#    grouping1=grouping_vars+ [x_var]
#    if x_var!='sec':
#        grouping1=grouping1+['sec']
#        print(grouping1)
#    # First grouping with time
#    df_grouped=df.groupby(grouping1).mean()[[y_var]]#.rename(columns={'pid':'plts'})#
#    df_grouped=df_grouped.reset_index()
    #Second grouping without time 
    #dfg=df_grouped.groupby(grouping_vars)[[y_var]].rolling(window=8,min_periods=1,center=True).mean().reset_index()
#    dfg=df_grouped.groupby(grouping_vars)[[y_var]].rolling(window=6,win_type='bartlett',min_periods=1,center=True).mean().reset_index()
#    
#    df_grouped['roll']=dfg[y_var]
#    return df_grouped

def rolling_means(df,grouping_var,y_vars,x_var): #Rolling mean values for variables y_vars 
    dv_=[]
    #x_var=['sec']
    if grouping_var=='inh':
        grouping_vars=[grouping_var]+['inh_exp_id']
    else:
        grouping_vars=[grouping_var]+['inh','inh_exp_id']
    grouping1=grouping_vars+[x_var]
    if x_var!='sec':
        grouping1=grouping1+['sec']
        print(grouping1)
    #df_grouped=df.set_index(['inh','inh_exp_id','sec']).sort_index().reset_index()#.set_index(['inh','exp_id'])  
    # First grouping with time
    df_grouped=df.groupby(grouping1)[y_vars].mean()#.rename(columns={'pid':'plts'})#
    df_grouped=df_grouped.reset_index()
    print(df_grouped.describe())
    #Second grouping without time 
    dfg=df_grouped.groupby(grouping_vars)[y_vars].rolling(window=6,win_type='bartlett',min_periods=1,center=False).mean().reset_index()
    print(dfg.describe())
    df_grouped[y_vars]=dfg[y_vars]
    #Rolling
    #for i,gr in dfg:
    
    #    df2=gr[[y_var]].rolling(window=8,win_type='blackman',min_periods=3,center=True).mean()#.\'bartlett'
    #    dv_.append(df2)
    #dv=pd.concat(dv_, axis=0)
    #df_grouped['roll']=dfg[y_var]
    return df_grouped

def rolling_perc(df1,df2,grouping_var='inside_injury',x_var='sec',window=6): #Rolling percentiles for df2/df1
    #dv_=[]
    #dg_=[]
    grouping_vars=[grouping_var]+['inh','inh_exp_id']
    grouping1=grouping_vars+[x_var]
    if x_var!='sec':
        grouping1=grouping1+['sec']
    dfg1=df1.groupby(grouping1)['pid'].size()#['pid']#.reset_index()
    dfg2=df2.groupby(grouping1)['pid'].size()#.count()['pid']
    dfg=dfg2/dfg1
    df_grouped=dfg.reset_index() 
    df_roll=df_grouped.groupby(grouping_vars)['pid'].rolling(window=window,win_type='bartlett',min_periods=1,center=True).mean().reset_index()
    df_diff=df_roll.groupby(grouping_vars)['pid'].diff()
    #for i,gr in dfg:
    #    df2=gr[['pid']].rolling(window=8,win_type='blackman',min_periods=3,center=True).mean()
    #    dv_.append(df2)
    #    df_gr=df2.diff()
    #    dg_.append(df_gr)
    #dv=pd.concat(dv_, axis=0)
    df_grouped['roll']=df_roll['pid']*100
    df_grouped['diff']=df_diff*100
    #df_grouped['roll']=dv*100
    #d_gr=pd.concat(dg_, axis=0)
    #df_grouped['diff']=d_gr*100
    if x_var==['minute']:
        grouping2=grouping_vars+[x_var]
        df_grouped=df_grouped.groupby(grouping2).mean()[['roll']].reset_index()
    return df_grouped

def rolling_bartlett(gr,y_var):
    gr[f'{y_var}_roll']=gr[y_var].rolling(window=6,win_type='bartlett',min_periods=3,center=True).mean()
    gr['count_roll']=gr['count'].rolling(window=4,win_type='bartlett',min_periods=3,center=True).mean()
    return gr

def rolling_mean_zled(df,y_var):
    dfg=df.groupby(['inside_injury','inh','zled','sec']).mean().reset_index()#Lägg till inh_exp_id senare!
    df_count=df.groupby(['inside_injury','inh','zled','sec']).count()['pid'].reset_index()#Lägg till inh_exp_id senare!
    dfg['count']=df_count['pid']
    dfg=dfg.groupby(['inside_injury','inh','zled']).filter(lambda x:x['count'].sum() >1000)
    dfg=dfg.groupby(['inside_injury','inh','zled']).apply(rolling_bartlett)
    dfg=dfg[['inside_injury','inh','zled','sec',y_var,f'{y_var}_roll','count']]
    dfg=dfg.reset_index()
    return dfg

def rolling_timecount(df,grouping_var,x_var):
    #grouping_var=[y_var]
    #x_var=[x_var]
    df_grouped=rolling_count(df,grouping_var,x_var)
    return df_grouped

def heatmap_filter(dfg1,mean_vars, smooth = 'gauss'): # FÖRSÖK ATT KÖRA GAUSSIAN ISTÄLLET FÖR ROLLING, EJ LYCKATS!
    #print('filter:')
        #dfg1 = dfg1.pivot(index = hue_var, columns =x_var)
        #dfg1 = dfg1.dropna()
        for p in mean_vars:
           for inh in dfg1.inh.unique():
                dfgp1 = dfg1[dfg1.inh == inh].pivot(index = 'hue_var', columns ='x_var', values=p)

                dfg_n = dfgp1.to_numpy()
        #print(dfg_n)
        if smooth == 'uniform':
            dfg_n=ndimage.uniform_filter(dfg_n, size=2)
        elif smooth == 'gauss':
            dfg_n=ndimage.gaussian_filter(dfg_n, sigma=2)#sigma=3)
        elif smooth == 'gauss_s1':
            dfg_n=ndimage.gaussian_filter(dfg_n, sigma=1)
        elif smooth == 'gauss1D':
            dfg_n=ndimage.gaussian_filter1d(dfg_n, sigma=2)
        dfg1.loc[:] = dfg_n
        dfg1 = dfg1.unstack()
        #print('OK')
        #dfg1=dfg1.fillna(0)
        return dfg1


def df_first(dfi):

    pg = dfi.sort_values(by = ['path','particle','tracknr']).groupby(['path','particle'])#.nth(0)#.rename('fframe')

    first_vars = ['nrtracks','track_time', 'size', 'inh', 
                "x_s", "ys", "zs", 
                'frame', 'sec', 
                'ca_corr', 'c0_mean', 'nba_d_5', 
                'elong', 'quadrant', 'dist_c', 'inside_injury', 'hsec', 'tri_sec', 'phase'] 
    renamed_vars = ["x_s", "ys", "zs", 'frame', 'sec', 'ca_corr', 'c0_mean', 'nba_d_5', 'elong', 'quadrant', 'dist_c', 'inside_injury', 'hsec', 'tri_sec', 'phase'] 

    new_names = [var + f'_1' for var in renamed_vars]

    df1 = pg[first_vars].first().rename(columns = {var:var_name for var, var_name in zip(renamed_vars, new_names)})
    
    return df1


def define_area_df(dfii, method = 'cyl', x = 'a_bin',
                   mean_vars = ['stab', 'track_time', 'res_time', 'dist_c'], 
                   hue = 'inh', col = 'tri_sec', row = 'quadrant',
                   melt = False, 
                   agg_var = 'path',
                   cond = False, 
                   square_dims = [4, 4, 2],
                   **agg_dic):
    
    rho_bin_width = 5
    rho_bin_nr = 180/rho_bin_width
    area_step = 1000
    
    def quadrant_bin_var(df):
        df['quadrant']='lateral'
        df.loc[(df['y_bin']>df['x_bin'])&(df['y_bin']>-df['x_bin']),'quadrant']='anterior'
        df.loc[(df['y_bin']<df['x_bin'])&(df['y_bin']<-df['x_bin']),'quadrant']='posterior'
        return df
    
    agg_dic.update({'mean_vars':mean_vars, 
               'hue':hue, 'col':col, 'row': row, 
               'melt':melt, 
               'cond': cond, 
               'x':x,
               'agg_var':agg_var})
    
    if method == 'cyl':
        
        if 'a_bin' not in dfii.columns:
            dfii = a_bin_var(dfii)
            
        area_df, kws = agg_plot(dfii, **agg_dic)
    
    elif method == 'pie': 
        print('Method: pie')
        if 'a_bin' not in dfii.columns:
            dfii = a_bin_var(dfii, area_step= area_step)
        
        dfii = rho_var(dfii, bin_width = rho_bin_width) # with bin width 15 area size is 167 with 5 it is 56  
        agg_dic.update({'row':'rho_bin'})
        area_df, kws = agg_plot(dfii, **agg_dic)
        area_df = quadrant_var(area_df, mode = 'rho')
        
        bin_size = area_step/rho_bin_nr 
        
    elif method == 'pie_vol': 
        print('Method: pie_vol')
        
        if 'a_bin' not in dfii.columns:
            dfii = a_bin_var(dfii)
        
        dfii = rho_var(dfii, bin_width = rho_bin_width) # with bin width 15 area size is 167 with 5 it is 56  
        agg_dic.update({'y':'rho_bin'})
        
        dfii = zz_var(dfii, decimals= -1)
        agg_dic.update({'plot_break_var':'zz'})
        
        bin_size = area_step*10/rho_bin_nr #Area bin * height * nr of angle bins
        
        area_df, kws = agg_plot(dfii, **agg_dic)
        area_df = quadrant_var(area_df, mode = 'rho')
        
        
    
    elif method == 'square':
        dfii = topo_bins_var(dfii, bin_steps = square_dims)
        agg_dic1 = {}
        for key, value in agg_dic.items(): 
            if value == 'quadrant':
                #quadrant = True
                agg_dic1.update({key:False})
            else:  
                agg_dic1.update({key:value})
        agg_dic1.update(dict(x = 'x_bin', y = 'y_bin', ))
        
        area_df, kws = agg_plot(dfii, **agg_dic1)
        area_df = quadrant_bin_var(area_df)
        area_df = a_bin_var(area_df)
        
        bin_size = square_dims[0]*square_dims[1]*square_dims[2]
        #area_df, kws = agg_plot(dfii, cond = False, x = 'a_bin', **agg_dic)
    
    if 'dist_c' in area_df.columns:
        area_df['dist_c'] = area_df['dist_c'].round(0)
    
    if 'path' in area_df.columns: 
        area_df = obs_weight_var(area_df, coeff= bin_size, grouper = ['inh'])
    else: 
        area_df['weight'] = bin_size
    
    return area_df, bin_size, kws

def area_df_dic(df_i, 
            method = 'pie_vol',
            mean_vars = ['stab', 'track_time', 'dist_c', 'dvy', 'dv', 'ca_corr', 'nba_d_5', 'elong', 'flatness', 'rho'],
            roll = False, 
            **kws):
    
    agg_dic = dict(method = method, mean_vars = mean_vars, roll = roll, agg_var = False, )
    agg_dic.update(**kws)
    
    
    dfa, bin_size, kws = define_area_df(df_i, **agg_dic)
    dfa_tracked, bin_size, kws = define_area_df(df_i[df_i.nrtracks > 1], **agg_dic)
    
    agg_dic.update({'agg_var': 'path'})
    
    dfap, bin_size, kws = define_area_df(df_i, **agg_dic)
    dfap_tracked, bin_size, kws = define_area_df(df_i[df_i.nrtracks > 1], **agg_dic)
    
    df_dic = {'All': dfa, 'Tracked': dfa_tracked, 'Path': dfap, 'Path_tracked': dfap_tracked}
    
    return df_dic, bin_size, kws
    

def set_ranges(dfi, var, bin_nr, new_bin_var = False, range = 'linear', **kws):
    if var in cfg.varBinDic.keys():
        rangeDic = cfg.varBinDic[var]
        print(var, ': ', rangeDic)
        r_min = rangeDic['min'] 
        r_max = rangeDic['max']
        range_edges = rangeDic['levels']
        round = rangeDic['round'] 
    else:
        r_min = dfi[var].min()
        r_max = dfi[var].max()
        range_edges = 'actual'
        round = 0
    if 'iso_max' in kws.keys():
        if kws['iso_max']:
            print('iso_max override')
            r_max = kws['iso_max'][1]
        
    #df_vals = dfi[var].copy()
    if range == 'log':
        print('log colorbar')
        if r_min == 0:
            levels = np.geomspace(1,r_max,bin_nr)
        else:
            levels = np.geomspace(r_min,r_max,bin_nr)
    else:    
        levels = np.linspace(r_min,r_max,bin_nr)
    
    levels[-1] = levels[-1] + 1e-10
    #CHANGED 2023-11-30
    #bin_labels = np.round(np.mean(np.c_[levels[:-1], levels[1:]], axis=1),round)
    bin_labels = list(np.round(levels[:-1],round))
    #bin_labels = list(np.round(levels[1:],round))
    
    print(bin_labels)
    if new_bin_var:
        var1 = f'{var}_bin'
        if var1 in dfi.columns:
            dfi.pop(var1)
    else: 
        var1 = var
    
    if range_edges == 'clipped':
        print('clipped range edges')
        
        bins = levels.copy()
        print(bins, bins.dtype)
        bins[0] = min((bins[0],dfi[var].min()))
        bins[-1] = max(bins[-1],dfi[var].max())
        binned_var = pd.cut(dfi[var], bins, labels = bin_labels).astype('float64')
        #dfi.loc[:,var1] = binned_var
        dfi[var] = binned_var
        
    else: 
        print('actual range edges')
        dfi = dfi[(dfi[var] > r_min) & (dfi[var] < r_max)]
        
        #bin_labels = sorted(dfi[var].unique().tolist())
        #levels = bin_labels.copy()
        #levels[0] = r_min
        #levels.append(r_max)
        print(bin_nr, len(bin_labels))
        #dfi[var1] = pd.cut(dfi[var], bin_nr-1, labels = bin_labels).astype('float64')
        dfi[var] = pd.cut(dfi[var], bin_nr-1, labels = bin_labels).astype('float64')
    
    #levels = np.round(levels,round)
        
    #bin_colors = np.round(edges[:-1],0)
    #bins[0] = df[var].min()

    #print(bins, bin_labels)
    
        
        
        
        

        
        
        #bin_labels = levels.copy()
        #bin_var = var
    #print(limits)
    return dfi, levels, bin_labels, round

def df_stackplot(df1, 
              stackVar = 'ca_corr', 
              stackLevels = 21,
              x = 'sec', 
              range = 'linear',
              print_check = False,
              **kws
              ):
    
    
    
    df1, edges, labels, round = set_ranges(df1, stackVar, stackLevels, range = range, **kws)
    #edges_r = sorted(edges, reverse = True)
    df1 = df1.astype({stackVar:"category"})#, "b":"category"})
        
        
    agg_dic = dict(cond = False, 
                            roll = True, 
                            melt = False, 
                            x = x, 
                            hue = stackVar, 
                            col = 'inh', 
                            row = False, 
                            agg_var = False,
                            mean_vars = False,)
    
    kws.update({key:value for key, value in agg_dic.items() if key not in kws.keys()})
    if print_check: print(kws)
    dfg, kws1 = agg_plot(df1, **kws)
    if print_check: print(kws1)
    stackParams = dict(edges = edges, labels = labels, x = x, stackVar = stackVar, col = kws['col'], row = kws['row'], round = round)
    #kws_params.update({dim:})
    #df1 = df1.astype({stackVar:"float64"})
    return dfg, stackParams, kws1

def df_countmeans(dfi,hue_var='iso_vol',x_var='sec',win=(5,3),win_name='blackman',min_per=3, mean_vars = 'std'):#'blackman'tukey'
    
    #mean_vars=,'ys','zs']##'stab',
    
    
    #cum_mov_=['cont_cum', 'cont_xy_cum','cont_x_cum','cont_y_cum','dvz_cum'] 
    if mean_vars == 'std':
        mov_= ['dv_s','dvy_s', 'stab']#'dvz_s','cont_s''dvx_s_abs',
        dens_= ['nba_d_5',]#'nba_d_10'
        pos_= ['zs',]#'zs'
        shape_ = ['elong'] #'flatness
        #pos_= ['ys','zs','depth']
        fluo_= ['ca_corr']#,'c1_mean'
        stab_= ['nrtracks_s']
        mean_vars=fluo_ + stab_ + pos_ + shape_ + dens_ + mov_ 
        
    #if sum_vars = 'std':
    #    sum_vars = ''
        #+ cum_mov_
    #inj_zone_vol=(2/3)*pi*(37.5**3)
    #vol_step=inj_zone_vol/5


    dfi=scale_vars(dfi)
    dfi['nrtracks_s']=dfi['nrtracks']*3.1
    grouping_ls=['inh',hue_var,x_var,'path',]
    grouping_mean=['inh',hue_var,x_var]
    
    #dfi = dfi.sort_values(by = grouping_mean)
   # df_count=dfi.groupby(grouping_ls).size().groupby(level=[0,2,3]).cumsum().groupby(level=[0,1,2]).mean()#
    #mask=df_count.iloc[:,:]<count_thr
    
    df_dic = {'ALL': dfi[dfi.nrtracks >1], 
              'NEW': dfi[dfi.tracknr < 4], 
              #'STABLE' : dfi[(dfi.tracknr > 4) & (dfi.tracknr<(dfi.nrtracks-4))],
              'OLD': dfi[dfi.tracknr>(dfi.nrtracks-3)]
              
                  }
    
    
   
    dfg1=dfi.groupby(grouping_mean, observed = True)[mean_vars].mean()#.reset_index()#.groupby(level=[0,1,2]).mean()

    dfg1=dfg1.groupby(level=[0,1]).rolling(window=win[0], min_periods=min_per,center=True).mean().droplevel([0,1])#.reset_index()#win_type=win_name,
    dfg1=dfg1.groupby(level=[0,2]).rolling(window= win[1], min_periods=min_per,center=True).mean().droplevel([0,1])
    dfg1=dfg1.groupby(level=[0,1]).rolling(window= win[0] ,min_periods=min_per,center=True).mean().droplevel([0,1])#.reset_index()
    

    for key, df in df_dic.items():
        dfc = df.groupby(grouping_ls).size().groupby(level=[0,1,2]).mean()
        dfc = dfc.groupby(level=[0,1], ).rolling(window= win[0], min_periods=min_per,center=True).mean().droplevel([0,1])#observed = True
        dfc = dfc.groupby(level=[0,2], ).rolling(window= win[1], min_periods=min_per,center=True).mean().droplevel([0,1])#observed = True.reset_index().rename(columns={0:f'count_{key}'})win_type=win_name,
        dfcount_roll = dfc.groupby(level=[0,1], ).rolling(window= win[0], min_periods=min_per,center=True).mean().droplevel([0,1]).rename(f'count_{key}')#.reset_index().rename(columns={0:f'count_{key}'})#observed = Truewin_type=win_name,
        dfg1 = dfg1.merge(dfcount_roll, on = grouping_mean)# = dfg1.append(dfcount_roll)
        #dfg1.loc[:,f'count_{key}']=dfc_roll[f'count_{key}']
        #dfg1[f'count_{key}']=dfc_roll#[f'count_{key}']
        #.iloc[::-1]
        dfc = df.groupby(grouping_ls).size().groupby(level=[0,1,2]).mean().groupby(level=[0,2]).cumsum()#.reset_index().rename(columns={0:f'cumcount_{key}'})
        #dfc = dfc.groupby(level=[0,1], ).rolling(window=win[0],min_periods=min_per,center=True).mean().droplevel([0,1])#observed = True.reset_index().rename(columns={0:f'cumcount_{key}'})
        #dfc = dfc.groupby(level=[0,2], ).rolling(window= win[1], min_periods=min_per,center=True).mean().droplevel([0,1])#observed = True
        dfcum_roll = dfc.groupby(level=[0,1], ).rolling(window=win[0], min_periods=min_per,center=False).mean().droplevel([0,1]).rename(f'cumcount_{key}')#observed = True#.reset_index()
        dfg1 = dfg1.merge(dfcum_roll, on = grouping_mean)

    dfg1 = dfg1.reset_index()

    #dfg1.loc[:,'net_growth']=dfg1['count_NEW']-dfg1['count_OLD']#df_count_diff.reset_index()#df_count_all.groupby['inh','hue_var'].diff().reset_index() #dfg1['count_NEW']-dfg1['count_OLD']
    #dfg1.loc[:,'net_growth_p']=dfg1['p_NEW']-dfg1['p_OLD']#df_count_diff_p#
    #dfg1['mask']=mask

    return dfg1#, dfcount_roll, dfcum_roll#dfg1

# Statistics
#---------------------------------------------------------------------------
def build_df_statcounts(inh_order):

    time=['frame','time']
    hue_vars=[]
    new_vars=[]
    #var1,var2=thrombus_parts_menu()
    xtra_vars=huevars_stat_menu()
    for var in xtra_vars.values():
        if var in cfg.old_huevars_stat:
            hue_vars.append(var)
        else:
            new_vars.append(var)

    df_var_list=[time,hue_vars]  
    df=build_df_lists(df_var_list,inh_order)
    df=add_xtravars(df,new_vars)
    
    return df,xtra_vars

def stats_df(df_test,test_var):
    nrrows=len(inh_order)**2
    index = pd.MultiIndex.from_product([inh_order,inh_order])
    d = pd.DataFrame({'MWU': np.ones(nrrows),'ttest': np.ones(nrrows)},index=index)
    for i in (inh_order):
        for j in (inh_order):
            try:
                p=pg.mwu(df_test.loc[df_test['inh']==i,test_var],df_test.loc[df_test['inh']==j,test_var], alternative='less')
                d.loc[(i,j)]['MWU']=p['p-val']
                p=pg.ttest(df_test.loc[df_test['inh']==i,test_var],df_test.loc[df_test['inh']==j,test_var], alternative='less')
                d.loc[(i,j)]['ttest']=p['p-val']
            except TypeError:
                    print('TypeError in',test_var)
    return(d)


#from pingouin import ttest
def t_testing(df_i, test_var = 'area', cond_var = 'inh', type = 'ttest', alternative = 'greater', **kws):
    if cond_var == 'inh':
        conditions = inh_order
    else:
        conditions = df_i[cond_var].unique()
        
    if type == 'ttest':
        results = pg.ttest(df_i.loc[df_i[cond_var] == conditions[0], test_var], df_i.loc[df_i[cond_var] == conditions[1], test_var], alternative = alternative, **kws)
    elif type == 'mwu':
        results = pg.mwu(df_i.loc[df_i[cond_var] == conditions[0], test_var], df_i.loc[df_i[cond_var] == conditions[1], test_var], alternative = alternative, **kws)
    elif type == 'anderson':
        results = stats.anderson_ksamp(df_i.loc[df_i[cond_var] == conditions[0], test_var].to_numpy(),df_i.loc[df_i[cond_var] == conditions[1], test_var].to_numpy(), **kws)
    else:
        results = stats.ks_2samp(df_i.loc[df_i[cond_var] == conditions[0], test_var].to_numpy(),df_i.loc[df_i[cond_var] == conditions[1], test_var].to_numpy(), alternative = alternative, **kws) #, method = 'exact'
    return results

def t_tests(data1, change_var = 'quadrant', alternative = 'greater', **kws ):

    test_dic = {}
    ls_ = []
    test_order = cfg.var_order[change_var] if change_var in cfg.var_order.keys() else data1[change_var].unique()
    for value in test_order:
        print(value)
        data = data1[data1[change_var] == value]
        test_results = t_testing(data, alternative = alternative, **kws)
        try:
            test_results.insert(0, change_var, value)
        #test_results.loc[:,change_var] = value
        #test_dic.update({value:test_results})
            ls_.append(test_results)
            
            
        except:
            results = False
            print(change_var, value, test_results)
    results = pd.concat(ls_, axis = 0) if len(ls_) > 0 else pd.DataFrame()

    return results 
    #else: 
    #    print('finished')
    

def huevars_stat_menu():
    print('Apart from total plt counts, the following variables can also be included in statistical comparisons:')
    for c, value in enumerate(cfg.thr_huevars_stat,0):
        print(c, value) 
    col_vars=input("Enter your choice as 0-2 integers separated by spaces:")
    varlist = [int(x) for x in col_vars.split()]
    xtra_vars={}
    for nr in range(1,len(varlist)+1):
        xtra_vars.update({'var'+str(nr):cfg.thr_huevars_stat[varlist[nr-1]]})
        #print('Original values of xtra_vars:')
        #for key,value in xtra_vars.items():
         #   print(key,value)
    return xtra_vars#var1,var2

def xtravars_stat(thr_reg_var,time_var):
    xtra_vars={}
    xtra_vars.update({'var'+str(1):thr_reg_var})
    xtra_vars.update({'var'+str(2):time_var})
        #print('Original values of xtra_vars:')
        #for key,value in xtra_vars.items():
         #   print(key,value)
    return xtra_vars#var1,var2


def stats_counts(df,xtra_vars):
    ls_desc=[]
    ls_tests=[]
    dfg_auc=df.groupby(['inh','path'])[['pid']].count().rename(columns={'pid':'count'}).reset_index()    
    df_desc=dfg_auc.groupby(['inh'])[['count']].describe() 
    df_tests_auc=stats_df(dfg_auc,'count')
    xtra_vars.update({'value1':'All','value2':'All',
                     'ls_desc':ls_desc, 'ls_tests':ls_tests})
    xtra_vars=insertvars_statdf(df_desc,df_tests_auc,xtra_vars)
    if 'var1' in xtra_vars:
        values1=df[xtra_vars['var1']].unique().tolist()
        #print(values1)
        for value1 in values1: 
            #print(value1)
            xtra_vars.update({'value1':value1})
            dfg_auc=df[df[xtra_vars['var1']]==value1].groupby(['inh','path'])[['pid']].count().rename(columns={'pid':'count'}).reset_index()
            df_desc=dfg_auc.groupby(['inh'])[['count']].describe()
            df_tests_auc=stats_df(dfg_auc,'count')
            xtra_vars=insertvars_statdf(df_desc,df_tests_auc,xtra_vars) 
    if 'var2' in xtra_vars:
        values2=df[xtra_vars['var2']].unique().tolist()
        #print(values2)
        xtra_vars.update({'value1':'All'})
        for value2 in values2:
            #print(value2)
            xtra_vars.update({'value2':value2})
            dfg_auc=df[df[xtra_vars['var2']]==value2].groupby(['inh','path'])[['pid']].count().rename(columns={'pid':'count'}).reset_index()
            df_desc=dfg_auc.groupby(['inh'])[['count']].describe()
            df_tests_auc=stats_df(dfg_auc,'count')
            xtra_vars=insertvars_statdf(df_desc,df_tests_auc,xtra_vars)
        for value1 in values1:
            xtra_vars.update({'value1':value1})
            for value2 in values2:
                xtra_vars.update({'value2':value2})
                dfg_auc=df[(df[xtra_vars['var1']]==value1)&(df[xtra_vars['var2']]==value2)].groupby(['inh','path'])[['pid']].count().rename(columns={'pid':'count'}).reset_index()
                df_desc=dfg_auc.groupby(['inh'])[['count']].describe()
                df_tests_auc=stats_df(dfg_auc,'count')
                xtra_vars=insertvars_statdf(df_desc,df_tests_auc,xtra_vars)
    df_desc=pd.concat(xtra_vars['ls_desc'],axis=0)#keys=groups,
    df_tests=pd.concat(xtra_vars['ls_tests'],axis=0)#keys=groups,
    #print(df_desc,df_tests)
            
    return df_desc,df_tests



def insertvars_statdf(df_desc,df_tests_auc,xtra_vars):
    if 'var1' in xtra_vars:
        df_desc.insert(0, xtra_vars['var1'],xtra_vars['value1'])
        df_tests_auc.insert(0,xtra_vars['var1'],xtra_vars['value1']) 
        if 'var2' in xtra_vars:
            df_desc.insert(1, xtra_vars['var2'],xtra_vars['value2'])
            df_tests_auc.insert(1,xtra_vars['var2'],xtra_vars['value2'])
    xtra_vars['ls_desc'].append(df_desc)
    xtra_vars['ls_tests'].append(df_tests_auc)
    #xtra_vars.update({'ls_desc':ls_desc, 'ls_tests':ls_tests})
    return xtra_vars

#---------------------------------------------------------------------------
#FUNCTIONS FOR STORAGE OF DATA & PLOTS
#---------------------------------------------------------------------------

        
def makedir(results_folder):
    try:
        os.mkdir(results_folder)
    except FileExistsError: 
        print(f'Folder {results_folder} already exists')


def get_plot_path(filename):
    file_name=time.strftime("%Y%m%d") +'_'+ filename
    if mfc.save_inh_names:
        inhs = ' '
        for key, value in cfg.treatment_orders.items():
            if inh_order[0] in value:
                inhs += key
        file_name += inhs
    plot_path = f'{mfc.results_folder}\\{file_name}'
    return plot_path

def save_fig(test_var,transparent=True, plot_format = 'std'):#,formats,**xtra1,*xtra
    if mfc.save_figs:
    
        plot_path = get_plot_path(test_var)
        if plot_format == 'std':
            plot_format = mfc.plot_formats
            
        if plot_format == 'both':
            plt.savefig(plot_path +'.png',bbox_inches='tight', dpi=300,transparent=transparent)
            plt.savefig(plot_path +'.pdf',bbox_inches='tight',transparent=transparent) #plt.savefig(plot_path_svg+'.svg',bbox_inches='tight',transparent=transparent)
        elif plot_format == 'svg':
            plt.savefig(plot_path +'.pdf',bbox_inches='tight',transparent=transparent)
        elif plot_format == 'pdf':
            plt.savefig(plot_path +'.pdf',bbox_inches='tight',transparent=transparent)
        else:
            plt.savefig(plot_path +'.png',bbox_inches='tight', dpi=300,transparent=transparent)

def save_table(df,*xtra):#,formats,**xtra1
    if mfc.save_figs:   
        file_name=time.strftime("%Y%m%d") #+'_'+ test_var
        if mfc.save_inh_names==True:
            inhibitors='_'.join(inh_order)
            file_name=file_name+'_'+inhibitors
        for stuff in xtra:
            file_name=file_name+'_'+stuff
        plot_path_csv=f'{mfc.results_folder}\\{file_name}'
        df.to_csv(plot_path_csv+'.csv')

def calc_comp2(df):
    #z=df.zs
    #c=df.c0_mean
    z= df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'zs']
    c= df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'c0_mean']
    
    #d_max={'a':4.227,'b':-0.1116,'c':0.0011}
    d_max = {'a': 5.6122, 'b': -0.1728, 'c': 0.0018}
    #d_min={'a':159.465109,'b':-1.330306,'c':0.011518}
    
    #print(d_max,d_min)    
    rat_high=d_max['a']+d_max['b']*z+d_max['c']*(z**2)
    c_low=114.6
    rat = c.div(c_low).clip(lower = 1) 
    c_corr = 100*(rat-1)/(rat_high)
    #c_corr=100*(c-c_low)/(c_high-c_low)
    #c_corr=c_corr.clip(0)
    df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'ca_corr'] = c_corr
    df.loc[df.path.isin(['210520_IVMTR109_Inj2_DMSO_exp3', '210520_IVMTR109_Inj3_DMSO_exp3', '210520_IVMTR109_Inj4_DMSO_exp3', '210520_IVMTR109_Inj6_DMSO_exp3']), 'ca_corr'] = np.nan
    
    return df


def calc_comp_mips(df):
    z= df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'zs']
    c= df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'c0_mean']
    try:
        d_max=pd.read_csv('df_reg zz 98.csv').to_dict('records')[0]#'Calcium comp\\df_reg zz max_calc.csv'
        d_min=pd.read_csv('df_reg zz 1.csv').to_dict('records')[0]#'Calcium comp\\df_reg zz min_calc.csv'
        #a_max=df_max.a.item()b_max=df_max.b.item()c_max=df_max.c.item()
    except:
        d_max={'a':814.697988,'b':-22.386315,'c':0.196006}
        d_min={'a':159.465109,'b':-1.330306,'c':0.011518}
        
    print(d_max,d_min)    
    c_high=d_max['a']+d_max['b']*z+d_max['c']*(z**2)
    c_low=d_min['a']+d_min['b']*z+d_min['c']*(z**2)
    c_corr=100*(c-c_low)/(c_high-c_low)
    c_corr=c_corr.clip(0)
    
    #df['ca_corr'] = c_corr
    df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'ca_corr'] = c_corr
    df.loc[df.path.isin(['210520_IVMTR109_Inj2_DMSO_exp3', '210520_IVMTR109_Inj3_DMSO_exp3', '210520_IVMTR109_Inj4_DMSO_exp3', '210520_IVMTR109_Inj6_DMSO_exp3']), 'ca_corr'] = np.nan
    
    return df

def calc_fmax(df):
    z= df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'zs']
    f1= df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'c0_mean']
    
    #z=df.zs
    #f1=df.c0_mean
    #f = f1.sub(114.6).clip(lower = 0)
    f = f1.sub(114.6)#.clip(lower = 0)
    f.loc[f < 0] = np.nan
    d_max = {'a': 618.865, 'b': -20.0537, 'c': 0.1737}
    fmax=d_max['a']+d_max['b']*z+d_max['c']*(z**2)
    f_max = (f.div(fmax))*100
    
    df.loc[df.inh.isin(['Vehicle MIPS', 'MIPS']),'ca_corr'] = f_max
    df.loc[df.path.isin(['210520_IVMTR109_Inj2_DMSO_exp3', '210520_IVMTR109_Inj3_DMSO_exp3', '210520_IVMTR109_Inj4_DMSO_exp3', '210520_IVMTR109_Inj6_DMSO_exp3']), 'ca_corr'] = np.nan
    return df
#---------------------------------------------------------------------------
# GENERAL FUNCTION PACKAGE FOR FOR CONSTRUCTING GROUPED DATAFRAMES 



#---------------------------------------------------------------------------
# ROLLING FUNCTION
#---------------------------------------------------------------------------

def roll_(dfg, 
          group_vars,
          roll_func = 'mean', 
          roll_win = [3],
          min_periods = 1, 
          win_type = False, #'bartlett, 
          center = True,
          roll_seq = False, 
          **kws): 
    print(f'Starting rolling, kws: {kws}')
    print('Group_vars:', group_vars)
    #roll_axes = kws['roll_axes'] if 'roll_axes' in kws.keys() else [kws['id_dic']['x']] # Ex. A: ['iso_A', 'rho']
    #roll_seq = kws['roll_seq'] if 'roll_seq' in kws.keys() else [(0,0)] # If multiple rolling steps with different rolling axes and different windoxt. Ex. A: [1,0,1]
    if not roll_seq: 
        roll_seq = [(kws['plot_dic']['x'], roll_win[0])]
        #= roll_seq if roll_seq else [(group_vars[0],8)] # Ex. A: [('rho',5),('iso_A',3),('rho', 5)]
    print(roll_seq)
    for roll in roll_seq:
        print('Rolling:', roll[0], roll[1])
        
        
        level = [i for i,var in enumerate(group_vars) if var not in roll[0]]
        print('level', level)
        droplevels = [i for i,var in enumerate(level)]
        if win_type: 
            print(win_type)
            dfg = dfg.groupby(level = level).rolling(window = roll[1], min_periods = min_periods, center = center,  win_type = win_type).mean().droplevel(droplevels)
        else: 
            dfg = dfg.groupby(level = level).rolling(window = roll[1], min_periods = min_periods, center = center,).mean().droplevel(droplevels)
    return dfg

#---------------------------------------------------------------------------
# NORMALIZATION FUNCTION
#---------------------------------------------------------------------------
def agg_norm(df_agg, norm_vars,):
    #norm_level = [i for i,var in enumerate(id_vars) if var not in kws['norm_axes']]
    #agg = dfi.groupby(id_vars).agg(base)
    df_agg_norm = df_agg.groupby(norm_vars).transform(lambda x: (x - x.mean()) / x.std())
    #df_agg.groupby(level = norm_levels).transform(lambda x: (x - x.mean()) / x.std())
    return df_agg_norm

#---------------------------------------------------------------------------
# CUMULATIVE AGG FUNCTION
#---------------------------------------------------------------------------

def agg_cum(df_agg, cum_vars,):
    #norm_level = [i for i,var in enumerate(id_vars) if var not in kws['norm_axes']]
    #agg = dfi.groupby(id_vars).agg(base)
    #df_agg_cum = df_agg.groupby(level = cum_levels).cumsum()
    df_agg_cum = df_agg.groupby(cum_vars).cumsum()

    return df_agg_cum

#---------------------------------------------------------------------------
# Percent calculation function for counts
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# GENERAL AGG FUNCTION
#---------------------------------------------------------------------------

def agg_(dfi, base, id_vars, observed = True, *meanVars):
    #norm_level = [i for i,var in enumerate(id_vars) if var not in kws['norm_axes']]
    #print(id_vars, base, dfi.dtypes)
    
    #if base == 'size':
    #    df_agg = dfi.groupby(id_vars).agg(base).fillna(0).astype('float64')
    if base == 'count':
        df_agg = dfi.groupby(id_vars)['pid'].agg(base ).astype('float64')
    elif meanVars:
        #print(meanVars[0])
        df_agg = dfi.groupby(id_vars, observed = observed, numeric_only=True)[meanVars[0]].agg(base).astype('float64')#.fillna(0)#, observed = True
    else:
        df_agg = dfi.groupby(id_vars).agg(base, numeric_only=True).astype('float64')#.fillna(0)# , observed = True, observed = True

    return df_agg

#def def_mean_vars(mean_vars):

        
#    return mean_vars#agg_dic

#---------------------------------------------------------------------------
# group_df_2: CENTRAL FUNCTION THAT CALCULATES COUNTS AND MEANS OF VARIABLES
# THEN ADDS NEW SETS OF VARIABLES WITH NORMALIZED AND/OR CUMULATIVE VALUES 
#---------------------------------------------------------------------------

def group_df_2(dfi, 
               group_vars,
               norm_axes = [],
                cum_axes = [], 
                percent_axes = [],
                #bin_axes = [],
                roll = False, #['sec]
                agg_levels = ['base'],
                print_check = False,
                #melt = True, #melt_dims = ['y', 'row'], value_vars = 
                    #['std'] or ['std' + custom] or [custom]
                **kws):

    if print_check: print('group_vars', group_vars)
    #print('kws group_df_2', kws)
    # START WITH FIRST AGG
    #mean_vars = def_mean_vars(mean_vars)
    #dfi = dfi[group_vars + kws['mean_vars']]
    #agg_levels = ['base']
    display(dfi.head(3))
    
    if 'path' in group_vars:
        dfi = dfi[group_vars + ['pid']+ kws['mean_vars']]
        dfg_size = agg_(dfi, 'count', group_vars).rename('count')
    else: 
        dfi = dfi[kws['varsWithAgg'] + ['pid'] + kws['mean_vars']]
        
        dfg_size = agg_(dfi, 'count', kws['varsWithAgg'])#.fillna(0)
        dfg_size = agg_(dfg_size, 'mean', group_vars, observed = True).rename('count')
    
    if kws['mean_vars']:
        #display()    
        dfg_mean = agg_(dfi, 'mean', group_vars, kws['mean_vars'])
        #print('Group_vars:')
        
        
        #print(dfg_mean.columns.to_list())
        dfg = dfg_mean.merge(dfg_size, on = group_vars)#left_index = True)#on = id_vars)
        if print_check: print('After merge:',dfg.reset_index().columns.to_list())
        #print(dfg.head(3))
    else:
        dfg = dfg_size
        if print_check: print('After agg:',dfg.reset_index().head(5))#.columns.to_list())
    if roll:
        dfg = roll_(dfg, group_vars, **kws)
        if print_check: print('After roll:',dfg.reset_index().columns.to_list())
        #print('After roll:',dfg.head(5))
    #print('dfg before agg_levels:', dfg)
    #agg_levels = kws['agg_levels']
    
    if len(agg_levels)>1:
        print('more than one agg_level')
        dfg['agg_level'] = agg_levels[0]
        dfg_ = [dfg]    
        for agg_level in agg_levels[1:]:
            if print_check: print('agg_level_', agg_level)
            if agg_level == 'norm':
                norm_vars = [var for var in group_vars if var not in norm_axes]
                dfg_agg = agg_norm(dfg, norm_vars)
                #agg_levels.append('norm')
                #dfg_norm['agg_level'] = 'norm',
                #if roll:
                #   dfg_norm = roll_(dfg_norm, id_vars, **kws)
                #dfg_.append(dfg_norm)
        
            #elif agg_level == 'bin':
            #    bin_axis = bin_axes[n]
            #    bin_vars = group_vars + [bin_axis]
            #    agg_level += bin_axis
            #    if 'path' in group_vars:
            #        dfi = dfi[group_vars + kws['mean_vars']]
            #        dfg_size = agg_(dfi, 'size', bin_vars).rename('count')
            #    else: 
            #        dfi = dfi[kws['varsWithAgg'] + kws['mean_vars']]
            #        dfg_size = agg_(dfi, 'size', kws['varsWithAgg'])
            #        dfg_size = agg_(dfg_size, 'mean', group_vars).rename('count')
            
            elif agg_level == 'percent':
                #'plot_dic': {'x': 'sec', 'y': 'value', 'hue': 'dist_c', 'row': 'variable', 'col': 'inh'}
                #hue_var = kws['plot_dic']['hue']
                perc_vars1 = [var for var in group_vars if var not in percent_axes]
                perc_vars2 = perc_vars1.copy()
                perc_vars2.append(percent_axes[0])
                if print_check: print(perc_vars1, perc_vars2)
                dfg2 = agg_(dfg_size, 'sum', perc_vars2)
                dfg1 = agg_(dfg2, 'sum', perc_vars1)
                
                dfg_agg = dfg.copy().rename('count')
                dfg_agg = dfg2/dfg1
                if print_check: print(dfg_agg)
                if print_check: print('DONE!')
                #dfg_agg = dfg_agg.rename('')
                
            else:
                cum_vars = [var for var in group_vars if var not in cum_axes]
                dfg_agg = agg_cum(dfg, cum_vars)
                
            #agg_levels.append(agg_level)
            dfg_agg['agg_level'] = agg_level
            #if roll:
            #    dfg_cum = roll_(dfg_cum, id_vars, **kws)
            dfg_.append(dfg_agg)
        
        dfg = pd.concat(dfg_, axis = 0, ignore_index= False)#.reset_index()
    
    return dfg

#---------------------------------------------------------------------------
# group_df_1: FUNCTION THAT CALLS group_df_2 IN DIFFERENT WAYS DEPENDING ON 
# THE CONDITIONS (parameter: cond) CHOSEN 
#---------------------------------------------------------------------------

def group_df_1(dfi, 
                cond = False, 
                **kws):
    #print(kws)
    if cond:
        #print(cond)
        value_vars = kws['value_vars']
        
        #if cond == 'per_ctrl_noAgg':
        #    df_i, df_ctrl = dfi[(dfi.inh.isin(inh_order[1:]))], dfi[(dfi.inh ==  inh_order[0])]
        #    dfg_ctrl = group_df_2(df_ctrl, kws['nonAggVars'], **kws)
            
        #    val_vars = [var for var in value_vars if var != 'count']
            
            
        #    print('dfi columns', df_i.columns, 'ctrl columns:', dfg_ctrl.columns, 'value_vars:',value_vars)
        #    dfg = dfg_ctrl
        
        
        #else: 
        if cond == 'per_new':
            df_i, df_j = dfi[(dfi.tracknr<2)], dfi
            dfg_i = group_df_2(df_i, kws['id_vars'], **kws)
            dfg_j = group_df_2(df_j, kws['id_vars'], **kws)
            dfg_j.replace(0,1)
            dfg  = dfg_i.div(dfg_j)
            dfg = dfg.rename(cond)
        
        elif cond == 'per_unstable':
            df_i, df_j = dfi[(dfi.tracknr == dfi.nrtracks)], dfi
            dfg_i = group_df_2(df_i, kws['id_vars'], **kws)
            dfg_j = group_df_2(df_j, kws['id_vars'], **kws)
            dfg_j.replace(0,1)
            dfg  = dfg_i.div(dfg_j)
            dfg = dfg.rename(cond)
            
        #if cond == 'per_ctrl':
        
        
        else:
            print('Condition per_ctrl')
            df_i, df_j = dfi[(dfi.inh.isin(inh_order[1:]))], dfi[(dfi.inh ==  inh_order[0])]
            print('id_vars:',kws['id_vars'])
        
            dfg_i = group_df_2(df_i, kws['id_vars'], **kws) 
            if kws['print_check']: print(display(dfg_i.head(3)))
            dfg_j = group_df_2(df_j, kws['nonAggVars'], **kws)
            if kws['print_check']: print(display(dfg_j.head(3)))
            #dfg_j.loc[dfg_j['value'] == 0,:] = 1
            #min = dfg_j[dfg_j>0]
            dfg_j.replace(0,1)
            #dfg_i.reset_index('inh', inplace = True)
            dfg_j.reset_index('inh', drop = True, inplace = True)
            dfg  = dfg_i.div(dfg_j)
            if 'value' in dfg.columns:
                dfg = dfg.rename(columns= {'value':cond})
        #dfg = dfg_i
        #print('dfg_i', dfg_i)
        #print('dfg_j', dfg_j)
        #rint(value_vars)
        #
        #dfg[value_vars] = (dfg_i[value_vars]/dfg_j[value_vars])*100
        #print((100*dfg_i.reset_index()['value'])/dfg_j.reset_index()['value'])        
        #dfg['value2'] = (100*dfg_i.reset_index()['value'])/dfg_j.reset_index()['value'] 
        #dfg['value2'] = (dfg_i['value']/dfg_j['value'])*100
    
    else:
        #print('Calling group_df_2')
        dfg = group_df_2(dfi, kws['id_vars'], **kws)
    
    
    return dfg

#---------------------------------------------------------------------------
# agg_plot: THE FUNCTION YOU CALL. COLLECTS VARIABLES, CALLS fix_params 
# TO GENERATE LISTS & OTHER UTILS. THEN CALLS group_df_1 TO PRODUCE DATAFRAME 
# AND MELTS THE DATA IF PARAMETER melt IS TRUE

#---------------------------------------------------------------------------
        
def agg_plot(
    dfi, 
    x = 'tracknr' , #x, y, hue, row, col, plot_break_var, **kws):
    y = 'value', 
    hue = 'quadrant2', 
    row = 'variable', 
    col = 'tri_sec',  
    agg_var = 'path',# False
    plot_break_var = False, #id_var 2
    print_check = False,
    #melt = True, #melt_dims = ['y', 'row'], value_vars = 
    **kws
    ):
    
    all_var_names = ['hue', 'row', 'col', 'plot_break_var', 'x','y', 'agg_var', ]
    all_vars = [col, row, hue, plot_break_var, x, y, agg_var]
    #grouping_mean=['inh',hue_var,x_var]
    kws.update(fix_params(dfi.columns.to_list(), all_var_names, all_vars, agg_var, print_check = print_check, **kws))#, melt
    #print(kws)
    
    
    #print('Dataframe columns', dfi.columns.to_list())
    dfg = group_df_1(dfi, agg_var = agg_var, **kws)
    
    if kws['melt']:
        dfg = dfg.melt(ignore_index= False)#var_name = 'measure' , 
        #id_vars.append('measure')
        #dfg = dfg.reset_index().set_index(group_vars + ['measure'])
    if print_check: print('After melt:',dfg.reset_index().columns.to_list())
    
    return dfg.reset_index(), kws

#---------------------------------------------------------------------------
# fix_params: GENERATES LISTS, DICTIONARIES & OTHER UTILS. 

#---------------------------------------------------------------------------
    
def fix_params(columns, all_var_names, all_vars, agg_var, 
               melt = True, 
               mean_vars = ['nba_d_5', 'ca_corr'],
               cond = False,
               norm_axes = False, 
               cum_axes = False,
               percent_axes = False,
               print_check = False,
               **kws
               ):

    new_dic = {}
    
    agg_levels = ['base']
    for level,agg in zip(['norm', 'cum', 'percent'],[norm_axes, cum_axes, percent_axes]):
    #for level,agg in zip(['norm', 'cum'],[norm_axes, cum_axes]):
        if agg:
            agg_levels.append(level)
    if print_check: print('agg_levels:', agg_levels)
    #CHECKS PLOT ARGUMENTS 
    plot_vars = [var for var in all_vars if var]
    plot_dic = {id:var for id,var in zip(all_var_names, all_vars) if var}
    
    meltSubstrings = ['variable',  'value', 'agg_level'] #if melt else [] #CHANGED 231104: took away last part!!!
    meltVars = [var for var in plot_vars if any([substr in var for substr in meltSubstrings])]
    nonMeltVars = [var for var in plot_vars if var not in meltVars]
    badVars = [var for var in nonMeltVars if var not in columns]
    id_vars = [var for var in nonMeltVars if var not in badVars]
    
    
    #print(id_vars, melt_vars, non_melt_vars, bad_vars)
    
    if melt:
        #if len(meltVars) in [2,3]:
            #id_vars = nonMelt#[var for var in plot_vars if var not in meltVars]
        #else:
        if print_check: 
            if len(meltVars) < 2:
                print('Too few melted var names selected. For melting to work, you must enter one variable with "value" and one with "variable" in name\n',
                        f'The variables {meltVars} will be ignored')
                
                    
            elif len(meltVars) > 3:
                #melt = False
                print('You have chosen four or more melted variables. For melting to work, you must choose one value var and one variable\n',
                        f'The variables {meltVars} will be ignored')
            else:
                print('MeltVars:',meltVars)
            
    
    id_dic = {key:value for key, value in plot_dic.items() if value in id_vars}
    plot_dic = {id:var for id,var in plot_dic.items() if var != agg_var}
    if not melt:
        plot_dic ={id:var for id,var in plot_dic.items() if var not in meltVars}
    
    nonAggVars = [value for key,value in id_dic.items() if key != 'agg_var' and value != 'path']
    varsWithAgg = id_vars if agg_var else id_vars + ['path']
    #dim_dic = {id:var for id,var in id_dic.items() if var != 'path'}
    #nonAggLevels = [i for i,var in enumerate(nonAggVars)]
    
    
    #cumLevels = [i for i,var in enumerate(id_vars) if var not in kws['cum_axes']]
    #normLevels = [i for i,var in enumerate(id_vars) if var not in kws['norm_axes']]
        
    if badVars:
        if print_check: print(f'Cannot interpret variables {badVars}, they will be ignored!!!')
        
    if mean_vars:
        
        mov_= ['dv_s','dvy_s', 'stab']#mov_= ['dvx_s_abs','dv_s','dvy_s', 'stab']#'dvz_s','cont_s'
        dens_= ['nba_d_5',]#'nba_d_10'
        pos_= ['zs',]#'zs'
        shape_ = ['elong'] #'flatness
        #pos_= ['ys','zs','depth']
        fluo_= ['ca_corr','c1_mean']
        stab_= ['nrtracks_s']
        std_mean_vars=fluo_ + stab_ + pos_ + shape_ + dens_ + mov_ 
        
        if 'std' in mean_vars:
            vars = std_mean_vars
            xtra_vars = [var for var in mean_vars if var not in (std_mean_vars + ['std'])]
            vars += xtra_vars
        else:
            vars = mean_vars
    else:
        vars = []#['pid']
    
    
    value_vars = vars + ['count']
        
    new_dic.update({'all_var_names': all_var_names, 
                'all_vars': all_vars,
                'plot_vars': plot_vars,
                'plot_dic': plot_dic,
                'melt':melt,
                'cond':cond,
                'id_vars': id_vars,
                'nonAggVars': nonAggVars,
                'varsWithAgg': varsWithAgg,
                'mean_vars':vars,
                'value_vars': value_vars,
                'agg_levels':agg_levels,
                'print_check':print_check
                #'nonAggLevels': nonAggLevels,
                #'cumLevels':cumLevels,
                #'normLevels': normLevels,
                })
    
    return new_dic#, #melt

 #PARAMETERS THAT CAN BE SET WHEN CALLING agg_plot:
 
    #dfi,       Dataframe
    
    # Columns in produced dataframe         
    #x = 'tracknr' , #x, y, hue, row, col, plot_break_var, **kws):
    #y = 'value', 
    #hue = 'quadrant2', 
    #row = 'variable', 
    #col = 'tri_sec',  
    #agg_var = 'path',# False
    #plot_break_var = False, 
    
    # SETTINGS 
    #melt = True,               IF TRUE, NAME ONE COLUMN (e.g. y) "value" 
    #                           AND ONE "variable",
    
    #mean_vars = ['nba_d_5', 'ca_corr'],
    
    # cond = False,             EITHER False, per_unstable, per_new or per_ctrl 
    
    # norm_axes = False,        CHOOSE AXES TO NORMALIZE AGAINST 
    # cum_axes = False,         CHOOSE AXES TO USE FOR CUMULATIVE COUNTS 
    # percent_axes = False,     CHOOSE AXES TO USE FOR PERCENT CALCULATIONS
    # REMOVED THIS: bin_axes = False          CHOOSE BINNED VARIABLE TO USE AS COLORBAR IN AREA PLOTS ETC
    #roll = False, #['sec]      CHOOSE IF TO APPLY ROLLING FUNCTIONS ON OUTPUT VARIABLES


    
    #roll_func = 'mean', 
    #min_periods = 1, 
    #win_type = False, #'bartlett, 
    #center = True,
    #roll_seq = False,          # CHOOSE ROLL AXES AND WINDOW, YOU CAN CHOOSE MULTIPLE ROLLING
    #                           STEPS e.g. A: [('rho',5),('iso_A',3),('rho', 5)]
    
    
    
    
