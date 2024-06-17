from numpy import save
import pandas as pd
from pathlib import Path
from magicgui import magicgui
#from numpy import False_
import config as cfg
import data_func_magic as dfc
#import high_level_func_magic as hlc
from types import SimpleNamespace
import numpy as np
import plot_func as plot

#---------------------------------------------------------------------------
# Some global variables for use in magicgui
#---------------------------------------------------------------------------


controls_=['_ctrl_', '_saline_','_veh-mips_', '_salgav-veh_','_salgav_','_veh-sq_','_c2actrl_','_df_demo_']
Controls_=[cfg.shorttolong_dic[ctrl] for ctrl in controls_]
Controls_=Controls_+['None']
treatments_=['_biva_','_cang_','_mips_','_asa_','_asa-veh_','_sq_','_cmfda_','_par4--_','_par4+-_','_par4-+_','_par4--biva_','_c2akd_']
Treatments_=[cfg.shorttolong_dic[treat] for treat in treatments_]
Treatments_=Treatments_+['None']
Analysis_=[ 
    ('Plt counts over time', 'timecounts'), 
    ('Means of variables over time','timemeans'), 
    #('Plots showing dependencies between two variables','varcorr'),
    ('Calcium stuff','ca_vars'),
    ('Statistical Calculations','stats'),
    ('Quality control & Outlier Detection','outliers'),
    ('Heatmaps', 'heatmaps'), 
    ('Trajectories & maps','traj'),
    ('Customized functions','custom' ),
    ]

#---------------------------------------------------------------------------
# Menu function created with MagicGui
#---------------------------------------------------------------------------

@magicgui(
    Controls = {'choices':Controls_,'allow_multiple':True},
    Treatments = {'choices':Treatments_,'allow_multiple':True},
    max_thr={"widget_type": "Slider", "min": 1, "label":"Upper tracking threshold"},
    min_thr={"widget_type": "Slider", "max": 200, "label":"Lower tracking threshold"},

    Save_figs={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [("Yes", True), ("No", False)],
        "label" : "Save Figures & Tables?"
    },
    File_formats={ 
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        #"choices": [(".png", 1), (".svg", 2),("both", 3)],
        "choices": [("png"), ("pdf"),("both")],
    },
    include_names={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [("Yes", True), ("No",False)],
        "label":"Include treatment names in filenames?",
    },
    del_outliers={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [("Yes", True), ("No", False)],
        "label":"Remove Outliers?",
    },
    file_folder={
        "name":"Folder_path",
        "label": "Choose a Folder for files:",
        "mode":'d'
        },  
    
    Analysis = {'choices':Analysis_,'allow_multiple':True},

    thr_reg_var = {
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        'choices': cfg.thr_reg_vars,
        "label":"Select thrombus region \nvariable to use in analysis:",
    },
    time_var = {
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        'choices': cfg.time_vars,
        "label":"Select xtra time \nvariable to use in analysis:",
    },
    call_button="Run analysis",
    layout="vertical",
    persist=True,
)
def input( Controls = ['Saline'], 
                Treatments = ['Cangrelor'], 
                max_thr = 200, min_thr = 1, 
                Save_figs = False, 
                File_formats = "png", 
                include_names = True, 
                del_outliers = True, 
                file_folder = Path.home(), 
                Analysis = ['timecounts'], 
                thr_reg_var='injury_zone',
                time_var='phase'
                ):
    global results_folder, analysis
    results_folder, analysis = file_folder, Analysis
    
    #return Treatments
    #inh_order.append(treat for treat in Treatments)
    #vars=[(keys,values) for keys,values in locals()]
    #print(locals().keys(),locals().values())
    #return locals().items()
    #if 'custom' in Analysis:
    #    return df

@input.called.connect
def start_processing():
    n=SimpleNamespace(**{k: p.default for k,p in input.__signature__.parameters.items()})
    input.close()
    df=build_dataframe()
    if n.Analysis != ['custom']: 
        print(f'{73 * "-"}\nStarting Data Analysis\n{73 * "-"}')
        masterfunc(df,n.thr_reg_var,n.time_var,n.Analysis)
    #input_var_dic=locals()
        print(f'{73 * "-"}\nAnalysis Completed\n{73 * "-"}')
        
    #func_b.input.value = value
    return df
    

def start_menu():
    input.show(run=True)

def build_dataframe():
    global inh_order, save_figs, plot_formats, save_inh_names, results_folder
    n=SimpleNamespace(**{k: p.default for k,p in input.__signature__.parameters.items()})
    inh_order=[]
    if n.Controls:
        if 'None' not in n.Controls:
            inh_order = n.Controls#.append(ctrl for ctrl in Controls)
        #if 'Demo Injuries' in Controls:
    if n.Treatments:
        if 'None' not in n.Treatments:
            inh_order += n.Treatments
    print(inh_order)
    #global save_figs, plot_formats,save_inh_names
    save_figs = n.Save_figs
    
    results_folder,analysis = n.Folder_path, n.Analysis
    plot_formats = n.File_formats
    print('plot_formats:',plot_formats)
    save_inh_names = n.include_names
    print(f'{73 * "-"}\nRun started, loading dataframe\n{73 * "-"}')
    print('Analysis value',n.Analysis)
    df_var_list=make_varlist(n.Analysis,n.thr_reg_var)
    
    df = dfc.build_df_lists(df_var_list,inh_order)
    df_cols=df.columns.tolist()
    if 'nrtracks' in df_cols:    
        df=df.loc[(df['nrtracks']>n.min_thr)&(df['nrtracks']<n.max_thr),:]
    if n.del_outliers:
        outliers=pd.read_csv('df_outliers.csv')
        df=df[~df.path.isin(outliers.path)]
    df=df.reset_index(drop=True)
    
    if n.thr_reg_var not in df_cols:
            df=dfc.add_xtravar(df,n.thr_reg_var)
    if n.time_var not in df_cols:
            df=dfc.add_xtravar(df,n.time_var)
    print(f'{73 * "-"}\nDataframe loaded\n{73 * "-"}')
    return df

def make_varlist( runs = ['timecounts'],thr_reg_var='inside_injury'): #Testa att göra om var_ls_ till tuple
    #print(runs)
    df_var_list=[['path', 'inh','particle']]
    #print(runs)
    timecount_vars_=['frame','time','nrtracks','tracknr','x_s','ys','zs','dvy','dvx','dvz','cont',]#,
    timemean_vars_=['x_s','ys','zs','dvy','dvx','dvz','dv','frame','time','stab', 'dist_cz',
    'cont', 'ca_corr','c0_mean', 'c1_mean','nba_d_5','nba_d_10','nrtracks','tracknr','elong', 'flatness']#'position','inside_injury', 'height',
    #varcorr_vars_ = ['position','inside_injury','height','dist_cz','frame','time','minute', 
    #'dvy', 'dv','stab','mov_class', 'movement','dvz','cont','cont_tot', 'c0_mean', 'c1_mean', 
    #'ca_corr','nba_d_5','nba_d_10','nba_d_15','nrtracks','tracknr']
    ca_vars_ = ['ca_corr','c0_mean', 'c0_max', 'nba_d_5','nrtracks','tracknr', 'stab', 'zs','mouse'] #'c1_mean', 'nba_d_10','nba_d_15',
    stat_vars_ = ['frame','time','nrtracks']
    outlier_vars_ = ['c0_mean', 'c0_max','c1_mean', 'c1_max','c2_mean', 'c2_max','nrtracks','tracknr','tracked']
    heatmap_vars_= ['zs','dist_c','depth','frame','time', 'dvy', 'dv','stab','mov_class', 'movement','dvz', 
    'cont_s','cont_tot','ca_corr', 'c0_mean','c1_mean','nba_d_5','nba_d_10','nrtracks','tracknr','exp_id','inside_injury']
    traj_vars_=['frame','time','x_s','ys','zs', 'dvx','dvy','dvz', 'ca_corr','c1_mean','cont','cont_tot','mov_class','movement','stab','tracknr','depth',]
    custom_vars_=['all_vars']

    #run_names_=['timecounts','timemeans','varcorr','stats','outliers','heatmaps','traj','custom']
    run_names_=['timecounts','timemeans','ca_vars','stats','outliers','heatmaps','traj','custom']
    
    #var_ls_=[timecount_vars_,timemean_vars_,varcorr_vars_,stat_vars_,outlier_vars_,heatmap_vars_,traj_vars_,custom_vars_,]
    var_ls_=[timecount_vars_,timemean_vars_,ca_vars_,stat_vars_,outlier_vars_,heatmap_vars_,traj_vars_,custom_vars_,]
    run_var_dic=dict(zip(run_names_, var_ls_))
    
    df_var_list += [value for key,value in run_var_dic.items() if key in runs]


    col_list = [item for sublist in df_var_list for item in sublist]
    
    if thr_reg_var in cfg.old_xtra_vars_:
        col_list.append(thr_reg_var)
    elif thr_reg_var in cfg.new_xtra_vars_:
        if thr_reg_var=='quadrant':
            new_vars=['x_s','ys']
            col_list+=new_vars
        elif thr_reg_var=='quadrant1':
            new_vars=['x_s','ys','zs']
            col_list+=new_vars
        elif thr_reg_var=='quadrant2':
            new_vars=['x_s','ys','zs']
            col_list+=new_vars
        elif thr_reg_var=='injury_zone':
            new_vars=['dist_cz']
            col_list+=new_vars

    col_list=list(dict.fromkeys(col_list))
    return col_list

    
#-------------------------------------------------------------------------------------------------------------
# Master functions that automatically load dataframes and execute plots
#-------------------------------------------------------------------------------------------------------------

#import menu_func as mfc
#import calc_func as ca

#-------------------------------------------------------------------------------------------------------------
def masterfunc(df,thr_reg_var,time_var,analysis_):
    plot.settings()
    #['timecounts','timemeans','varcorr','stats','outliers','heatmaps','custom']
    choice_made=False
    if 'timecounts' in analysis_:
        run_timecountplots(df,thr_reg_var)
        choice_made=True
    if 'timemeans' in analysis_:
        run_timemeanplots(df,thr_reg_var)
        choice_made=True
    if 'varcorr' in analysis_:
        run_catplotmeans(df,thr_reg_var)
        choice_made=True
    if 'stats' in analysis_:
        do_stats(df,thr_reg_var,time_var)
        choice_made=True
    if 'outliers' in analysis_:
        run_outliers(df)
        choice_made=True
    if 'heatmaps' in analysis_:
        run_heatmaps(df,'inside_injury')#You can change the region variable here!!!!
        choice_made=True
    if 'traj' in analysis_:
        run_traj(df)#You can change the region variable here!!!!
        choice_made=True
    if choice_made:
        print('Finished analysis')
    else:
        print('No analysis chosen.')
    

#-------------------------------------------------------------------------------------------------------------
# Master function that automatically runs a set of timecount lineplots for treatments defined in inh_order
#-------------------------------------------------------------------------------------------------------------

def run_timecountplots(df,xtra_var):
    print(f'{73 * "-"}\nStarting analysis of platelet counts over time\n{73 * "-"}')
    plot.lineplot_count_indexp(df,xtra_var)
    plot.lineplot_count_all(df)
    plot.lineplot_count_reg(df,col=xtra_var)
    if xtra_var != 'inside_injury':
        plot.lineplot_count_reg(df,col='inside_injury')
    if xtra_var != 'position':
        plot.lineplot_count_reg(df,col='position')
    plot.lineplot_newplts(df,col=xtra_var)  
    plot.lineplot_pltperc(df,col=xtra_var)
    plot.perc_unstable_indexp(df,col=xtra_var )

#-------------------------------------------------------------------------------------------------------------
# Master function that automatically runs a set of time-mean lineplots for treatments defined in inh_order
#-------------------------------------------------------------------------------------------------------------

def run_timemeanplots(df,col_var):
    print(f'{73 * "-"}\nStarting analysis of variable means over time\n{73 * "-"}')
    df=dfc.cont_xy_var(df)
    df=dfc.scale_var(df,'cont_xy')
    df=dfc.scale_var(df,'dvz')
    df=dfc.scale_var(df,'dvy')
    plot.run_meantime_plots(df,col_var)

#-------------------------------------------------------------------------------------------------------------
# Master function that automatically runs a set of time-mean catplots for treatments defined in inh_order
#-------------------------------------------------------------------------------------------------------------

def run_catplotmeans(df,hue_var):
    
    columns_=df.columns.tolist()
    
    
    # Movements over time
    print(f'{73 * "-"}\nStarting analysis of platelet movements over time\n{73 * "-"}')
    plot.catplot_dvzmin(df, hue_var)
    plot.catplot_dvzminpos(df, hue_var)
    plot.catplot_dvyminpos(df, hue_var)

    #Fibrin vs other variables
    if 'c1_mean' in columns_:
        catplots_fibrin(df)
    
    #Dist_cz vs other variables
    if 'dist_cz' in columns_:

        lineplots_distc(df)
        lineplots_isovol(df)
        

# Calculates fibrin quantile bins and executes plots comparing calcium fluorescence with other measures
#-------------------------------------------------------------------------------------------------------------
def catplots_fibrin(df):
    #Binning
    dfg=df[(~df.inh.isin(['Bivalirudin','PAR4-/- + biva']))&(df.height=='bottom')].copy()
    binned_var='c1_mean'
    bins=10
    #dfg,bin_var,bin_order=qbinning_labels(dfg,binned_var,bins)
    dfg,bin_var,bin_order=dfc.qbinning_quant(dfg,binned_var,bins)
    dfg=dfc.phase_var(dfg)
    
    #Fibrin plots
    print(f'{73 * "-"}\nStarting analysis of fibrin fluorescence in core \n{73 * "-"}')
    plot.catplot_fibrin_percunstable(dfg,bin_var,bin_order)
    plot.catplot_fibrin_dvy(dfg,bin_var,bin_order)
    plot.catplot_fibrin_percunstable_phases(dfg,bin_var,bin_order)
    plot.catplot_fibrin_mov(dfg,bin_var,bin_order)
    plot.catplot_fibrin_stab(dfg,bin_var,bin_order)
    plot.catplot_fibrin_cacorr(dfg,bin_var,bin_order)

# Lineplots with distance from center on x-axis
#-------------------------------------------------------------------------------------------------------------
def lineplots_distc(dfg):
    #Rounds distance to closest integer
    print(f'{73 * "-"}\nStarting analysis of distance from center \n{73 * "-"}')
    dfg['dist_cz']=np.round(dfg.loc[:,'dist_cz'].copy(),decimals=0)
    dfg=dfg[dfg.dist_cz<100].copy()
    #dfg = dfg.astype({'dist_cz': int})
    
    #binned_var='dist_cz'
    #bins=20
    #dfg,bin_var,bin_order=qbinning_labels(dfg,binned_var,bins)
    #dfg,bin_var,bin_order=qbinning_quant(dfg,binned_var,bins)
    
    dfg=dfc.phase_var(dfg)
    
    #dist_c plots
    plot.lineplot_distcz_mov(dfg)
    plot.lineplot_distcz_stab(dfg)
    plot.lineplot_distcz_nrtracks(dfg)
    plot.lineplot_distcz_nba(dfg)
    plot.lineplot_distcz_cacorr(dfg)

# Lineplots with isovolumetric bins of distc_z on x-axis
#-------------------------------------------------------------------------------------------------------------
def lineplots_isovol(dfg):
    #Rounds distance to closest integer
    print(f'{73 * "-"}\nStarting analysis of isovolumetric layers \n{73 * "-"}')
    dfg=dfg[dfg['dist_cz']<125].copy()
    dfg=dfc.isovol_bin_var(dfg)
    dfg=dfc.phase_var(dfg)
    plot.lineplot_count_isovol(dfg)
    plot.lineplot_countmov_isovol(dfg)

#-------------------------------------------------------------------------------------------------------------
# Master function that automatically runs a set of heatmaps for treatments defined in inh_order
#-------------------------------------------------------------------------------------------------------------

def run_heatmaps(df,hue_var):
    
    
    plot.def_global_col_heat(hue_var) #Defines hue var in plot package as heat_col_var
    params_heatmap=plot.params_choice(choice='heat') #set standard plot params 
    #ls_hue=[hue_var]

    df=dfc.led_bins_var(df) # makes time and z bins
    df['down']=-df['dvz_s'] #df['down']=-df['dvz_s']
    regions=sorted(df[hue_var].unique().tolist(),reverse=True) #Returns the sorted values of hue_var  

    if len(dfc.inh_order)>3:
        group1,group2=dfc.chunks(dfc.inh_order,2)
        groups_ls=[group1,group2]
        params_heatmap.update({'group1':group1,'group2':group2,'groups':group1,'groups_ls':groups_ls})
    else:
        group1=dfc.inh_order
        group2=False
        groups_ls=False
        groups=dfc.inh_order
        params_heatmap.update({'group1':group1,'groups':group1,'groups_ls':False})

# Count heatmaps
#-------------------------------------------------------------------------------------------------------------
    print(f'{73 * "-"}\nStarting analysis of Platelet count heatmaps \n{73 * "-"}')
    params_heatmap.update({'groups':group1,'var':'pid','regions':regions})#'vmax':100,'vmin':0,,'cmap':'seq'
    plot.do_heatmaps_count(df,**params_heatmap)#hue_var,


# Mean heatmaps
#-------------------------------------------------------------------------------------------------------------    
    print(f'{73 * "-"}\nStarting analysis of mean variable heatmaps \n{73 * "-"}')

    #heat_mean_dic={
    #    0:{'var':'ca_corr','v':{'vmax':70,'vmin':10},'cmap':'seq'},#0:{'var':'ca_corr','v':{'vmax':80,'vmin':10}},
   #     1:{'var':'dv','v':{'vmax':2,'vmin':0.3},'cmap':'seq'},
    #    2:{'var':'nba_d_5','v':{'vmax':8,'vmin':6},'cmap':'seq'},#2:{'var':'nba_d_5','v':{'vmax':10,'vmin':6}},
    #    3:{'var':'nba_d_10','v':{'vmax':13,'vmin':8},'cmap':'seq'},#3:{'var':'nba_d_10','v':{'vmax':15,'vmin':8}},
    #    4:{'var':'nrtracks','v':{'vmax':150,'vmin':30},'cmap':'seq'},#4:{'var':'nrtracks','v':{'vmax':170,'vmin':30}},
    #    5:{'var':'cont_s','v':{'vmax':250,'vmin':0},'cmap':'seq'},
    #    6:{'var':'cont_s','v':{'vmax':250,'vmin':-100},'cmap':'div'},
    #    7:{'var':'down','v':{'vmax':150,'vmin':-50},'cmap':'div'},#7:{'var':'down','v':{'vmax':150,'vmin':-50}},
    #     }
    heat_mean_dic=cfg.heat_mean_dic[hue_var]
    params_heatmap.update({'count_thr':5})
    for n in heat_mean_dic.keys():
        heat_dic=heat_mean_dic[n]
        print('Mean values for:',heat_dic['var'])
        params_heatmap.update(heat_dic)
        #params_heatmap.update({'var':heat_dic['var'],'vmax':heat_dic['v']['vmax'],'vmin':heat_dic['v']['vmin'],'cmap':heat_dic['cmap']})
        plot.do_heatmaps_mean(df,**params_heatmap)#hue_var,
    

# Perc heatmaps
#-------------------------------------------------------------------------------------------------------------    
    print(f'{73 * "-"}\nStarting analysis of Percentile Count heatmaps \n{73 * "-"}')
    heat_per_dic={
        0:{'desc':'new','v':{'vmax':0.7,'vmin':0}},
        1:{'desc':'unstable','v':{'vmax':0.7,'vmin':0}},
        2:{'desc':'contracting','v':{'vmax':0.6,'vmin':0}},
        3:{'desc':'drifting','v':{'vmax':0.7,'vmin':0}},
              }
    
    var='pid'
    params_heatmap.update({'var':var,'count_thr':10,'cmap':'seq'})
    for n in heat_per_dic.keys():
        heat_dic=heat_per_dic[n]
        print('Fraction:',heat_dic['desc'])
        params_heatmap.update({'vmax':heat_dic['v']['vmax'],'vmin':heat_dic['v']['vmin'],'desc':heat_dic['desc'],'cmap':'seq'})
        plot.do_heatmaps_perc(df,**params_heatmap)#hue_var,



def run_heatmaps_count(df,hue_var):
    hues_ls=df[hue_var].unique().tolist()
    params_heatmap=plot.create_params()['heatmap']
    params_heatmap.update({})

#-------------------------------------------------------------------------------------------------------------    
# Trajectories
#-------------------------------------------------------------------------------------------------------------    

def run_traj(df):
    df=dfc.time_bin_var(df)
    for series in cfg.demo_ls_ls_:
        inh_subset=[cfg.shorttolong_dic[inh] for inh in series]
        print(inh_subset)
        pc=df[df.inh.isin(inh_subset)]
        #print(pc_grp.head())
        dfg=dfc.movesum_timebin_var(pc)
        dfg=dfc.movclass_timebin_var(dfg,5)
        xtra_params={}
        plot.t_traj_mov(dfg,'t_tracknr',**xtra_params)

    


# ---------------------------------------------------------------------------
# Quality Control & Outlier detection
# ---------------------------------------------------------------------------
def run_outliers(df):
    inh_order=dfc.expseries_menu()
    df_count=plot.outliers_count1(df)
    df_count.insert(0,'measure','count')
    df_fluo=plot.outliers_fluo(df)
    df_fluo.insert(0,'measure','fluo')
    df_nrtracks,df_inliers=plot.outliers_nrtracks(df)
    df_nrtracks.insert(0,'measure','nrtracks')
    df_inliers.insert(0,'measure','no outlier')
    df_outliers=pd.concat([df_count,df_fluo,df_nrtracks],axis=0)
    df_outliers=df_outliers.reset_index()
    #outliers_ls=df_outliers_menu(df_outliers)
    df_outliers=dfc.df_outliers_menu(df_outliers,df_inliers)
    
        
    #return df_outliers
  
#---------------------------------------------------------------------------
# Statistics 
#---------------------------------------------------------------------------
def do_stats(df,thr_reg_var,time_var):
    
    #measure=int(input("Do you want to calculate statistics on:\n\n(1) Plts Counts\n(2) Means of variables\n"))
    
    #if measure == 1:
        #df,xtra_vars=dfc.build_df_statcounts(inh_order)
    xtra_vars=dfc.xtravars_stat(thr_reg_var,time_var)
    df_desc,df_tests=dfc.stats_counts(df,xtra_vars)
    test_var='plt_count'
    
    #print(df_desc,df_tests)
    dfc.save_table(df_desc,'Descriptive stats'+'_'+test_var)
    dfc.save_table(df_tests,'Statistical tests'+'_'+test_var)
    return df_desc,df_tests


    
