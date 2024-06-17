import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
import itertools


#---------------------------------------------------------------------------
#df_path = Dataframe specifyting paths for files stored on computer
#----------------------------------------------------------------------
df_paths=pd.read_csv('file_paths.csv')
#df_paths=pd.read_csv('file_paths - laptop.csv')
#----------------------------------------------------------------------
#Settings for plots
blue_red=sns.diverging_palette(250, 15, s=75, l=40,center="dark",as_cmap=True)#sns.palplot(sns.diverging_palette(250, 15, s=75, l=40,center="dark"))
cmaps={'div': 'cmc.berlin','seq':'turbo','seq_r':'viridis_r'}#,'viridis'RdYlBu''icefire''RdYlBu_r''turbo''turbo_r''Spectral''viridis''cmc.batlow'#'div':'icefire','div':'cet_CET_D4'
heat_mean_vars_=['ca_corr','dv','nba_d_5','nba_d_10','nrtracks','cont_s','cont_s2','down',]

heat_mean_dic = {
  'inside_injury': 
  {
    'ca_corr':{'var':'ca_corr','v':{True:{'vmax':70,'vmin':20},False:{'vmax':40,'vmin':20}},'cmap':'seq'},#0:{'var':'ca_corr','v':{'vmax':80,'vmin':10}},
    'dv':{'var':'dv','v':{True:{'vmax':2,'vmin':0.5},False:{'vmax':2,'vmin':1}},'cmap':'seq'},
    'nba_d_5':{'var':'nba_d_5','v':{True:{'vmax':9,'vmin':6.5},False:{'vmax':12,'vmin':7}},'cmap':'seq_r'},#2:{'var':'nba_d_5','v':{'vmax':10,'vmin':6}},
    'nba_d_10':{'var':'nba_d_10','v':{True:{'vmax':10,'vmin':7.5},False:{'vmax':12,'vmin':9}},'cmap':'seq_r'},#3:{'var':'nba_d_10','v':{'vmax':15,'vmin':8}},
    'nrtracks':{'var':'nrtracks','v':{True:{'vmax':175,'vmin':50},False:{'vmax':80,'vmin':10}},'cmap':'seq'},#4:{'var':'nrtracks','v':{'vmax':170,'vmin':30}},
    'cont_s':{'var':'cont_s','v':{True:{'vmax':250,'vmin':0},False:{'vmax':250,'vmin':0}},'cmap':'seq'},
    'cont_s2':{'var':'cont_s','v':{True:{'vmax':250,'vmin':-100},False:{'vmax':250,'vmin':-100}},'cmap':'div','center':0},
    'down':{'var':'down','v':{True:{'vmax':100,'vmin':-100},False:{'vmax':75,'vmin':-75}},'cmap':'div','center':0},#7:{'var':'down','v':{'vmax':150,'vmin':-50}},
  },

'minute': 
  {
    'ca_corr':{'var':'ca_corr','v':{True:{'vmax':70,'vmin':20},False:{'vmax':40,'vmin':20}},'cmap':'seq'},#0:{'var':'ca_corr','v':{'vmax':80,'vmin':10}},
    'dv':{'var':'dv','v':{True:{'vmax':2,'vmin':0.5},False:{'vmax':2,'vmin':1}},'cmap':'seq'},
    'nba_d_5':{'var':'nba_d_5','v':{True:{'vmax':9,'vmin':6.5},False:{'vmax':12,'vmin':7}},'cmap':'seq_r'},#2:{'var':'nba_d_5','v':{'vmax':10,'vmin':6}},
    'nba_d_10':{'var':'nba_d_10','v':{True:{'vmax':10,'vmin':7.5},False:{'vmax':12,'vmin':9}},'cmap':'seq_r'},#3:{'var':'nba_d_10','v':{'vmax':15,'vmin':8}},
    'nrtracks':{'var':'nrtracks','v':{True:{'vmax':175,'vmin':50},False:{'vmax':80,'vmin':10}},'cmap':'seq'},#4:{'var':'nrtracks','v':{'vmax':170,'vmin':30}},
    'cont_s':{'var':'cont_s','v':{True:{'vmax':250,'vmin':0},False:{'vmax':250,'vmin':0}},'cmap':'seq'},
    'cont_s2':{'var':'cont_s','v':{True:{'vmax':250,'vmin':-100},False:{'vmax':250,'vmin':-100}},'cmap':'div','center':0},
    'down':{'var':'down','v':{True:{'vmax':100,'vmin':-100},False:{'vmax':75,'vmin':-75}},'cmap':'div','center':0},#7:{'var':'down','v':{'vmax':150,'vmin':-50}},
  },

False: 
  {
    'ca_corr':{'var':'ca_corr','v':{True:{'vmax':70,'vmin':20},False:{'vmax':40,'vmin':20}},'cmap':'seq'},#0:{'var':'ca_corr','v':{'vmax':80,'vmin':10}},
    'dv':{'var':'dv','v':{True:{'vmax':2,'vmin':0.5},False:{'vmax':2,'vmin':1}},'cmap':'seq'},
    'nba_d_5':{'var':'nba_d_5','v':{True:{'vmax':9,'vmin':6.5},False:{'vmax':12,'vmin':7}},'cmap':'seq_r'},#2:{'var':'nba_d_5','v':{'vmax':10,'vmin':6}},
    'nba_d_10':{'var':'nba_d_10','v':{True:{'vmax':10,'vmin':7.5},False:{'vmax':12,'vmin':9}},'cmap':'seq_r'},#3:{'var':'nba_d_10','v':{'vmax':15,'vmin':8}},
    'nrtracks':{'var':'nrtracks','v':{True:{'vmax':175,'vmin':50},False:{'vmax':80,'vmin':10}},'cmap':'seq'},#4:{'var':'nrtracks','v':{'vmax':170,'vmin':30}},
    'cont_s':{'var':'cont_s','v':{True:{'vmax':250,'vmin':0},False:{'vmax':250,'vmin':0}},'cmap':'seq'},
    'cont_s2':{'var':'cont_s','v':{True:{'vmax':250,'vmin':-100},False:{'vmax':250,'vmin':-100}},'cmap':'div','center':0},
    'down':{'var':'down','v':{True:{'vmax':100,'vmin':-100},False:{'vmax':75,'vmin':-75}},'cmap':'div','center':0},#7:{'var':'down','v':{'vmax':150,'vmin':-50}},
  },

'phase': 
  {
    'ca_corr':{'var':'ca_corr','v':{True:{'vmax':70,'vmin':20},False:{'vmax':40,'vmin':20}},'cmap':'seq'},#0:{'var':'ca_corr','v':{'vmax':80,'vmin':10}},
    'dv':{'var':'dv','v':{True:{'vmax':2,'vmin':0.5},False:{'vmax':2,'vmin':1}},'cmap':'seq'},
    'nba_d_5':{'var':'nba_d_5','v':{True:{'vmax':9,'vmin':6.5},False:{'vmax':12,'vmin':7}},'cmap':'seq_r'},#2:{'var':'nba_d_5','v':{'vmax':10,'vmin':6}},
    'nba_d_10':{'var':'nba_d_10','v':{True:{'vmax':10,'vmin':7.5},False:{'vmax':12,'vmin':9}},'cmap':'seq_r'},#3:{'var':'nba_d_10','v':{'vmax':15,'vmin':8}},
    'nrtracks':{'var':'nrtracks','v':{True:{'vmax':175,'vmin':50},False:{'vmax':80,'vmin':10}},'cmap':'seq'},#4:{'var':'nrtracks','v':{'vmax':170,'vmin':30}},
    'cont_s':{'var':'cont_s','v':{True:{'vmax':250,'vmin':0},False:{'vmax':250,'vmin':0}},'cmap':'seq'},
    'cont_s2':{'var':'cont_s','v':{True:{'vmax':250,'vmin':-100},False:{'vmax':250,'vmin':-100}},'cmap':'div','center':0},
    'down':{'var':'down','v':{True:{'vmax':100,'vmin':-100},False:{'vmax':75,'vmin':-75}},'cmap':'div','center':0},#7:{'var':'down','v':{'vmax':150,'vmin':-50}},
  }
}





#---------------------------------------------------------------------------
#Lists and dictionaries of short and long treatment names
#Short names are used in file names and long names are used in graphs etc
#----------------------------------------------------------------------

saline_ =['_saline_','_biva_','_cang_']
PAR4_ =['_ctrl_','_par4--_','_par4+-_','_par4-+_','_par4--biva_','_biva_']
MIPS_ =['_mips_','_veh-mips_','_asa-veh_','_salgav-veh_']
ASA_ =['_salgav_','_asa_']
SQ_ =['_veh-sq_','_sq_']
CMFDA_ =['_ctrl_','_cmfda_']
All_=['_biva_','_cang_','_ctrl_',
                '_mips_','_saline_','_sq_','_veh-mips_',
                '_veh-sq_','_par4+-_','_par4-+_','_par4--biva_',
                 '_par4--_','_asa-veh_','_asa_', '_salgav-veh_',
                 '_salgav_'
               ]
all_demo_= ['_saline_', '_biva_', '_cang_', '_veh-sq_', '_sq_', '_asa-veh_', '_asa_', '_veh-mips_', '_mips_', '_ctrl_', '_par4--_', '_par4--biva_', '_salgav-veh_', '_salgav_'] 
saline_demo_=['_saline_','_biva_','_cang_']
sq_demo_=['_veh-sq_', '_sq_']
asa_demo_=['_asa-veh_', '_asa_']
par4_demo_=['_ctrl_', '_par4--_', '_par4--biva_']
mips_demo_=['_mips_','_veh-mips_']
salgav_demo_=['_salgav-veh_', '_salgav_']
demo_ls_ls_=[saline_demo_,sq_demo_,asa_demo_,par4_demo_,mips_demo_,salgav_demo_]

simple_=['_par4--_','_cang_']


shortnames_ =['_biva_','_cang_','_cmfda_','_ctrl_',
                '_mips_','_saline_','_sq_','_veh-mips_',
                '_veh-sq_','_par4+-_','_par4-+_','_par4--biva_',
                 '_par4--_','_asa-veh_','_asa_', '_salgav-veh_',
                 '_salgav_','_c2actrl_','_c2akd_','_df_demo_',
               ]
longnames_ =['Bivalirudin','Cangrelor','CMFDA','Control',
                 'MIPS','Saline','SQ','Vehicle MIPS',
                 'Vehicle SQ','PAR4+/-','PAR4-/+','PAR4-/- + biva',
                 'PAR4-/-','ASA + Vehicle','ASA','Salgav + Vehicle',
                 'Salgav','C2alpha+','C2alpha-','Demo Injuries'
                ]

expseries_listnames=['Saline cohort','Thromin-PAR4 cohort','MIPS cohort','ASA cohort','SQ cohort','CMFDA cohort',
                     'All Treatments','Simple']
treatments_=[saline_,PAR4_,MIPS_,ASA_,SQ_,CMFDA_,All_,simple_]
#treatments_dic=dict(saline_,PAR4_,MIPS_,ASA_,SQ_,CMFDA_,All_,simple_)
shorttolong_dic = dict(zip(shortnames_, longnames_))
longtoshort_dic = dict(zip(longnames_,shortnames_))

MIPS_order = ['Vehicle MIPS', 'MIPS']
cang_order = ['Saline','Cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['Vehicle SQ', 'SQ']
treatment_orders=dict(mips=MIPS_order,cang=cang_order,sq=SQ_order)
inh_order_ = MIPS_order + cang_order + SQ_order
pal_MIPS=dict(zip(MIPS_order, sns.color_palette('Oranges',n_colors=len(MIPS_order))))

#'PAR4+/-','PAR4-/+','PAR4-/- + biva',
#                 'PAR4-/-','ASA + Vehicle','ASA','Salgav + Vehicle',
#                 'Salgav','C2alpha+','C2alpha-','Demo Injuries'
                
#---------------------------------------------------------------------------
#Lists of variables for certain applications
#----------------------------------------------------------------------
xtra_vars_=['position','inside_injury','injury_zone','height','z_pos','phase','minute']
old_xtra_vars_=['position','inside_injury','height','z_pos']
new_xtra_vars_=['injury_zone','phase','minute','quadrant','quadrant1']
thr_reg_vars=['position','inside_injury','injury_zone','quadrant','quadrant1','quadrant2']
time_vars=['sec','frame','min','phase']
sec_vars = ['tled','sec','frame', 'hsec']

#scale_vars=['dv','dvx','dvy','dvz','cont', 'cont_xy']
scale_vars=['dv','dvy','azi_diff']
#---------------------------------------------------------------------------
# Lists specifying orders of variables
#---------------------------------------------------------------------------
bol_order=[True,False]
mov_class_order=['still','contractile','loose','none']
mov_class_order1=['still','contractile','loose']
movement_order=['immobile','contracting','drifting','unstable','none']
movement_order1=['immobile','contracting','drifting','unstable']
movements_order2=['immobile','contracting','drifting']
position_order=['head','tail','outside']
#height_order=['bottom','middle','top']
phase_order=['Early','Mid','Late']
Phase_order=['Growth','Consolidation']
quadrant_order=['anterior','lateral','posterior']
quadrant1_order=['core','anterior','lateral','posterior']
quadrant2_order=['center','anterior','lateral','posterior']
quadrant3_order=['aa', 'al', 'pl', 'pp']#['A','AL','PL','P']
change_order = ['Number new', 'Number unstable','Net Growth']
hsec_order = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-600']
trisec_order = ['0-100', '100-300', '300-600']
size_order = ['small', 'large']
condition_order = ['vehicle', 'treatment']
region_order = [f'_{n:.0f}' for n in range(10)]
region_order2 = [f'_{n:.0f}' for n in np.arange(0, 110, 10)]
#regio_order = [r[0] + r[1] for r in itertools.product(quadrant3_order, region_order)]#[1:])]
regio_order = [r[0] + r[1] for r in itertools.product(quadrant3_order, region_order[::-1])]
regio_order2 = [r[0] + r[1] for r in itertools.product(quadrant3_order, region_order2)]#[1:])]
part_order = ['inner', 'outer']
cohort_order = ['MIPS','cang','SQ']
#regio_order = ['ii'] + [r[0] + r[1] for r in itertools.product(quadrant3_order, region_order[1:])]
#regio_order = [r[0] + r[1] for r in itertools.product(quadrant3_order, region_order)]

#size_quant_order = [0, 0.25, 0.5, 0.75]
#iso_A = 'numeric'
#iso_vol = 'numeric'
# Lists specifying orders of variables
#---------------------------------------------------------------------------


var_order=dict(inside_injury=bol_order, mov_class=mov_class_order1, 
               movement_type=movements_order2, injury_zone=bol_order, 
               position=position_order, tri_sec = trisec_order, phase=Phase_order, Phase=phase_order,
               quadrant=quadrant_order,quadrant1=quadrant1_order,quadrant2=quadrant2_order, quadrant3 = quadrant3_order,
               MIPS=MIPS_order,cang=cang_order,SQ=SQ_order, inh = inh_order_, measure=change_order, 
               hsec = hsec_order,size = size_order, condition = condition_order, region = region_order, regio = regio_order,
               region2 = region_order2, regio2 = regio_order2,
               part = part_order, cohort = cohort_order, 
               #size_quant = size_quant_order, 
               )  #height=height_order,    


units = dict(minutes = 'min', hsec = 's', tled = 'Time (s)', size = 'thrombi', inh = '')
labels = dict(minutes = 'Time (min)', tled = 'Time (s)', hsec = 'Time (s)', size = 'Outer radius', inh = '')
    
#---------------------------------------------------------------------------
# Lists specifying specific experiments
#---------------------------------------------------------------------------
xtra_mips = ['210520_IVMTR109_Inj2_DMSO_exp3', '210528_IVMTR111_Inj6_DMSO_exp3',
       '210528_IVMTR110_Inj4_MIPS_exp3', '210528_IVMTR111_Inj4_DMSO_exp3',
       '210531_IVMTR112_Inj4_MIPS_exp3', '210607_IVMTR118_Inj3_MIPS_exp3',
       '210528_IVMTR110_Inj5_MIPS_exp3', '210531_IVMTR112_Inj5_MIPS_exp3',
       '210528_IVMTR111_Inj3_DMSO_exp3', '210528_IVMTR110_Inj6_MIPS_exp3',
       '210528_IVMTR110_Inj3_MIPS_exp3', '210531_IVMTR112_Inj6_MIPS_exp3',
       '210607_IVMTR118_Inj4_MIPS_exp3', '210607_IVMTR118_Inj6_MIPS_exp3',
       '210531_IVMTR112_Inj3_MIPS_exp3', '210520_IVMTR109_Inj4_DMSO_exp3',
       '210520_IVMTR109_Inj6_DMSO_exp3', '210528_IVMTR111_Inj7_DMSO_exp3',
       '210520_IVMTR109_Inj3_DMSO_exp3']


#---------------------------------------------------------------------------
# Dictionary specifying binning ranges for commonly used variables
#---------------------------------------------------------------------------


ranges = dict(
  #fsec = {'max': 600},
  z1 = {'max': 64, 'min': 0, },
  zs = {'max': 64, 'min': 0, },
  track_time = {'max': 100, 'min': 3, 'round': 0, 'levels':'clipped'},
  res_time = {'max': 100, 'min': 3, 'round': 0, 'levels':'clipped'},
  nrtracks = {'max': 20, 'min': 1, 'round': 1, 'levels': 'clipped'},
  fsec = {'max': 600, 'min': 0, 'round': 0},
  ca_corr = {'max': 100, 'min':0, 'levels' : 'clipped'},
  ca_corr_1 = {'max': 100, 'min':0, 'levels' : 'clipped'},
  ca_1 = {'max': 100, 'min':0, 'levels' : 'clipped'},
  ca_corr_sum_3 = {'max': 30, 'min':-30, 'levels' : 'clipped'},
  stab = {'max': 4, 'levels' : 'clipped', 'round': 4},
  dv = {'max':3, 'round' : 4},
  iso_A = {'max': 100, 'levels' : 'clipped'},
  dist_c = {'max': 100, 'min':0, 'round':0, 'levels':'clipped'},
  rho = {'max': 90, 'min':-90,},
  z_bin = {'max': 64, 'min': 0, },
  z1_bin = {'max': 64, 'min': 0, },
  iso_Area =  {'max': 15000, 'min': 200},
  iso_Areas =  {'max': 15000, 'min': 200},
  iso_volumes =  {'max': 1500000, 'min': 50000, 'levels' : 'clipped'},
  nba_d_5 = {'min': 5.5, 'max': 8, 'levels': 'clipped', 'round': 3},
  nba_d_5_1 = {'min': 5, 'max': 10, 'levels': 'clipped', 'round': 3},
  per_unstable = {'min': 0, 'max': 1, 'levels': 'clipped', 'round':3},
)

std_range = dict(
  levels = 'actual',
  min = 0,
  round = 0
)

varBinDic = {}
for key, dic in ranges.items():
  key_dic = {}
  for key1, value in dic.items():
    key_dic.update({key1: value})
  key_dic.update({key2:value for key2, value in std_range.items() if key2 not in key_dic.keys()})
  varBinDic.update({key:key_dic})
    
#---------------------------------------------------------------------------
# Dictionary specifying names of corresponding categorical variables for some continuous variables
#---------------------------------------------------------------------------

rankCatVars = {
  'dv': 'Movement', 
  'track_time': 'Cohesion',
  'stab': 'Stability',
  'nba_d_5': 'Cluster_density',
  'count': 'Density'
}
posCatVars_ = ['Movement', 'Cohesion', 'Density']
negCatVars_ = ['Cluster_density', 'Stability']

catRanks_ = ['Low', 'Medium', 'High']

#cat_rank_vars = []
  
  
varDic = dict(
    std={
        'iso_max':(0,120),#64,#70,
        'ylim':False,
        'sup':False,
        'axhline':False,
        'y_lab':False,
        'meas': 'mean',
        'logscale': False,
        'linewidth': 1.5
    },
    #---------------------------------------------------
    # Geometry
    #---------------------------------------------------
    
    x_bin = {
        #'isovol_max':64,
        'iso_max':(0,100),
        'ylim': (0,100),
        'axhline':True,
        'sup':f'x (μm)',
        'logscale': False,
        'y_lab': f'|x| (μm)',# f'Distance from center of injury (μm)',
        
    },
    
    x_bin_abs = {
        #'isovol_max':64,
        'iso_max':(0,100),
        'ylim': (0,100),
        'axhline':True,
        'sup':f'x (μm)',
        'logscale': False,
        'y_lab': f'x (μm)',# f'Distance from center of injury (μm)',
        
    },
    
    y_bin = {
        #'isovol_max':64,
        'iso_max':(0,100),
        'ylim': (0,100),
        'axhline':True,
        'sup':f'y (μm)',
        'logscale': False,
        'y_lab': f'y (μm)',# f'Distance from center of injury (μm)',
        
    },
    
     z_bin = {
        #'isovol_max':64,
        'iso_max':(0,100),
        'ylim': (0,100),
        'axhline':True,
        'sup':f'z (μm)',
        'logscale': False,
        'y_lab': f'z (μm)',# f'Distance from center of injury (μm)',
        
    },
    
    dist_c = {
        #'isovol_max':64,
        'iso_max':(0,100),
        'ylim': (0,100),
        'axhline':True,
        'sup':f'Dist to center of injury (μm)',
        'logscale': False,
        'y_lab': f'ρ (μm)',# f'Distance from center of injury (μm)',
        
    },
    
    iso_A = {
        #'isovol_max':64,
        'iso_max':(0,100),
        'ylim': (0,100),
        'axhline':True,
        'sup':f'Distance from center of injury (μm)',
        'logscale': False,
        'y_lab':f'ρ (μm)',#f'Dist to center of injury (μm)',
        
    },
    iso_Area = {
        #'isovol_max':64,
        'iso_max':(0,100),
        'ylim': (0,100),
        'axhline':True,
        'sup':f'Area (μm²)',
        'logscale': False,
        #'y_lab':f'Area (μm²)',#f'Dist to center of injury (μm)',
        'y_lab':f'Dist to center of injury (μm)',
        
    },
        
    track_time = {
        #'isovol_max':64,
        'iso_max':(4,600),
        'ylim': (4,600),
        'axhline':True,
        'sup':f'Cohesion lifetime (s)',
        'logscale': True,
        'cbar_ticks': [10, 100, 1000],
        'y_lab':f'Cohesion lifetime (s)',
        
    },
    
    #---------------------------------------------------
    # CUMULATIVE MOVEMENTS
    #---------------------------------------------------
    cont_cum={
        #'isovol_max':64,
        'iso_max':(0,120),
        'ylim': False,
        'axhline':True,
        'sup':'Net total platelet contraction (all axes)',
        'y_lab':r"Net total contraction ($\mu$m)",
        
    },
    cont_xy_cum={
        #'isovol_max':64,
        'ylim': False,
        'axhline':True,
        'sup':'Net contraction, XY-plane',
        'y_lab':r"Net contraction, XY-plane ($\mu$m)",
        
    },
    cont_x_cum={
        #'isovol_max':64,
        'ylim': False,
        'axhline':True,
        'sup':'Net contraction, X axis',
        'y_lab':r"Net contraction in x axis ($\mu$m)",
        
    },
    cont_y_cum={
        #'isovol_max':64,
        'ylim': False,
        'axhline':True,
        'sup':'Net platelet contraction, Y axis',
        'y_lab':r"Net contraction, Y axis ($\mu$m)",
        
    },
    dvz_cum={
        #'isovol_max':64,
        'ylim': False,
        'axhline':True,
        'sup':'Net movement, z axis',
        'y_lab':r"Net movement, z axis ($\mu$m)",
        
    },
    #---------------------------------------------------
    # Different variables
    #---------------------------------------------------
    ca_corr={
        'iso_max': (0,80),
        'ylim': (False, 60),#80,
        'axhline':False,
        'sup':'Platelet corrected calcium',
        'y_lab':r"Calcium (AU)",
        #'y_lab':r"Corrected CAL520 fluorescence (AU)",
        'cbar_ticks': [0, 10, 20, 30, 40],#np.arange(0, 100, 20),
        'logscale': False,
    },
    
    cont_s={
        'iso_max':(0, 62),#50,
        'ylim': False,
        'axhline':True,
        'sup':'Total platelet contraction',
        'y_lab':r"Contraction (nm/s)",
    },
    nba_d_5={
        'iso_max':(0,84),#50,
        'ylim': (False, 15),#10,
        'axhline':False,
        'sup':'Average distance to 5 closest platelets',
        'y_lab':r"Interplatelet distance ($\mu$m)",
        
    },
    nba_d_10={
        'iso_max': (0,84),#50,
        'ylim': (False, 20),
        'axhline':False,
        'sup':'Average distance to 10 closest platelets',
        'y_lab':r"Interplatelet distance ($\mu$m)",
        'logscale': True,
    },
    c1_mean={
        #'isovol_max':64,
        'iso_max': (0,64),
        'ylim': (False, 500),#False,
        'sup':'Fibrin fluorescence',
        'axhline':False,
        'y_lab':r"Mean plt fibrin fluorescence (AU)",
    },
    stab={
        #'iso_max':(0, 84),
        'iso_max':(0, 84),
        'ylim': (False, 10),
        'sup':'Stability',
        'axhline':True,
        'y_lab':r"Instability ($\mu$m/s)",
    },
   
    elong = {
        'iso_max':(0, 80),
        #'ylim': 8,
        'sup':'Plt Elongation',
        'axhline':True,
        'y_lab':"Elongation",
    },
    #---------------------------------------------------
    # MOVEMENTS
    #---------------------------------------------------
    dv_s={
        'iso_max':(0, 75),
        'ylim': (False, 800),
        'axhline':True,
        'sup':'Movement, all axes',
        'y_lab':"Movement (nm/s)",
    },
    dvx_s_abs={
        'iso_max':(0, 62),
        'ylim': (False, 300),
        'axhline':True,
        'sup':'Movement, x axis',
        'y_lab':"Movement, x axis (nm/s)",
    },
    dvy_s={
        'iso_max':(0, 62),
        'ylim': False,
        'axhline':True,
        'sup':'Movement, y axis',
        'cbar_ticks': [-200, -100, 100, 200],#np.arange(0, 100, 20),
        'y_lab':"Net movement, y axis (nm/s)",
    },
    dvz_s={
        'iso_max':(0, 64),
        'ylim': False,
        'axhline':True,
        'sup':'Movement, Z axis',
        'y_lab':"Movement, Z axis (nm/s)",
    },
    cont_xy_s={
        #'isovol_max':64,
        'ylim': (False, 350),
        'axhline':True,
        'sup':'Mean platelet contraction in XY-plane',
        'y_lab':"Contraction (nm/s)",
        },

    #---------------------------------------------------
    # COUNTS
    #---------------------------------------------------
    count_ALL={
        'iso_max':(0, 100),#'iso_max':(37, 100),#70,
        'ylim': False,
        'sup': 'No platelets per isovolume',#'Average platelet packing density per isovolume',
        'axhline':False,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Platelet count, iso',#Density (platelets/\N{GREEK SMALL LETTER MU}m\N{SUPERSCRIPT THREE})'
        'linewidth': 2,
    },
        count_NEW={
        'iso_max':(0, 100),#'iso_max':(37, 100),
        'ylim': False,
        'sup': 'No platelets per isovolume',#'Average platelet packing density per isovolume',
        'axhline':False,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Platelet count, iso'#Density (platelets/\N{GREEK SMALL LETTER MU}m\N{SUPERSCRIPT THREE})'
    },
        count_OLD={
        'iso_max':(0, 100),#'iso_max':(37, 100),
        'ylim': False,
        'sup': 'No platelets per isovolume',#'Average platelet packing density per isovolume',
        'axhline':False,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Platelet count, iso'#Density (platelets/\N{GREEK SMALL LETTER MU}m\N{SUPERSCRIPT THREE})'
    },
    cumcount_ALL={
        'iso_max':(0, 100),#70,
        'ylim': False,
        'sup': 'No platelets per isovolume',#'Average platelet packing density per isovolume',
        'axhline':False,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Cum platelet count'#Density (platelets/\N{GREEK SMALL LETTER MU}m\N{SUPERSCRIPT THREE})'
    },
    cumcount_NEW={
        'iso_max':(0,100),#70,
        'ylim': False,
        'sup': 'No platelets per isovolume',#'Average platelet packing density per isovolume',
        'axhline':False,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Cum count, new plts'#Density (platelets/\N{GREEK SMALL LETTER MU}m\N{SUPERSCRIPT THREE})'
    },
    cumcount_OLD={
        'iso_max':(0, 100),#70,
        'ylim': False,
        'sup': 'No platelets per isovolume',#'Average platelet packing density per isovolume',
        'axhline':False,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Cum count, new plts'#Density (platelets/\N{GREEK SMALL LETTER MU}m\N{SUPERSCRIPT THREE})'
    },
    p_OLD={
        #'isovol_max':64,
        'ylim': False,
        'sup':'Fraction unstable platelets per isovolume',
        'axhline':False,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Unstable platelets (%)'
    },
    p_NEW={
        #'isovol_max':64,
        'ylim': False,
        'sup':'Fraction newly recruited plts (%)',
        'axhline':False,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Fraction newly recruited plts (%)'
    },
    net_growth={
        #'isovol_max':64,
        'ylim': False,
        'sup':'Fractional growth (%)',
        'axhline':True,
        #'y_lab':"Average platelet density in isovolume (platelets/$\mu$m)",#{density:.2f}
        'y_lab':f'Fractional growth (%)'
    },
    
    #---------------------------------------------------
    # POSITIONS
    #---------------------------------------------------
    ys={
        #'isovol_max':64,#70,
        'ylim': (False, 80),
        'axhline':True,
        'sup':'Thrombus center of gravity, Y axis',
        'y_lab':'Center of gravity, Y axis',
    },
    zs={
        'iso_max': (0,70) ,#70,
        'ylim': (False, 50), #False,
        'axhline':False,
        'sup':'Thrombus center of gravity, Z axis',
        'y_lab':'Center of gravity, Z axis',
    },
    
   
    )
    
#---------------------------------------------------------------------------