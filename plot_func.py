

#---------------------------------------------------------------------------
# PLOTTING FUNCTIONS
#---------------------------------------------------------------------------
from cmath import phase
from click import style
#from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import os
import time
#import imp
import warnings
import sys
import statsmodels.api as sm
import itertools
import math as m
from IPython.display import clear_output   
import pingouin as pg
import tabulate
from scipy import ndimage
from scipy import stats
import data_func_magic as dfc
import config as cfg
from importlib import reload
from matplotlib.gridspec import GridSpec
from matplotlib import colors
#import swatch
from math import pi
from matplotlib.colors import from_levels_and_colors, LogNorm, Normalize, TwoSlopeNorm
pd.options.mode.chained_assignment = None  # default='warn'
from IPython.display import display
import seaborn.objects as so
from seaborn import axes_style
import colormaps as cmaps
import cmasher as cmr
import cmcrameri.cm as cmc

#import cmcrameri
#import matp
#reload(cfg)



leg_titles=dict(inh='Treatment',position='Position', inside_injury='Inside Injury XY', 
            injury_zone='Inside Injury Zone',path='Experiment',movement='Movement',quadrant='Quadrant',quadrant1='Core/Quadrant',measure = 'Count', size = 'Size', cohort = 'Treatment')

#pal_MIPS = dict(zip(cfg.MIPS_order, sns.color_palette('Blues')[2::3]))
pal_MIPS = dict(zip(cfg.MIPS_order, ("#575756", "#5ea2cb")))#("#5ea2cb", "#575756", )

pal_cang = dict(zip(cfg.cang_order, sns.color_palette('Oranges')[2::3]))

pal_SQ = dict(zip(cfg.SQ_order, sns.color_palette('Greens')[2::3]))

pal1={**pal_MIPS,**pal_cang,**pal_SQ}

pal_MIPS2 = dict(zip(cfg.MIPS_order, sns.color_palette('plasma')[0::3]))

pal_cang2 = dict(zip(cfg.cang_order, sns.color_palette('plasma')[0::4]))

pal_SQ2 = dict(zip(cfg.SQ_order, sns.color_palette('plasma')[0::5]))

pal2={**pal_MIPS2,**pal_cang2,**pal_SQ2}

def settings(palette1='Paired'):#palette='Set2''Set1'
    # Settings and Parameters
    #----------------------------------------------------------------------
    warnings.simplefilter(action = "ignore", category = FutureWarning)
    #plt.rcParams['figure.constrained_layout.use'] = False

    pd.set_option('display.max_rows', 300)
    pd.set_option('display.max_columns', 300)
    #pd.option_context('display.max_rows', None, 'display.max_columns', None)
    
    #AESTHETICS
    #Parameters for image export 
    #---------------------------------------------------------------------------
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['image.cmap'] = 'viridis'#'turbo'#'viridis' 'magma'
    
    if palette1 == 'adobe':
        colors=[dic for dic in swatch.parse('AdobeColor Illustration.ase')[0]['swatches']]#AdobeColor Campaign Behavior.ase'
        palette1=[]
        pal_=[]
        for c in colors:
            pal_.append(c['data']['values'])
        for c in pal_:
            palette1.append(tuple(c))
        #sns.set_palette(pal1)#len(dfc.inh_order)
    #elif palette1 == 'custom':
        #sns.set_palette('Paired',len(dfc.inh_order))

    #Plot styles 
    #---------------------------------------------------------------------------
    #sns.set_style('darkgrid')
    #sns.set_style('whitegrid')
    sns.set_context("paper")
    sns.set_style('ticks')# 'ticks'
    
    #sns.set_context("talk") #paper,notebook,talk,poster
    
    #plt.rcParams['image.cmap'] = 'jet'
    plt.rcParams['image.interpolation'] = 'none'
    sns.set_palette(palette1)
    # CHOOSE COLOR PALETTE 
    # ------------------------------------------------------------------------
    #sns.set_palette('Dark2')# palette='tab20'sns.set_palette('Paired')
    #sns.set_palette('viridis',len(inh.order))#Set3'Set1'
    
    
        

# Functions for setting plot params
#---------------------------------------------------------------------------
def global_plot_settings(style = 'ticks',
                         context = 'custom',
                         font="Arial",
                         #font_scale = 1.0,
                         #fonts = 'small',
                         #ticks = 'small',
                         #linewidth = 'thinn',
                         size = 'small',
                         tick_color = '0.2', 
                         
                         ):
    
    #if context == 'custom':
    #    if fonts == 'small' and ticks == 'thinn':
    #        context = context_settings()
   # context_dic = {}
    if size == 'small':
        context_dic = dict(
            text_large = 7, 
            text_small = 6, 
            linewidth_thick = 0.75, 
            linewidth_thin = 0.5,
            ticksize_large = 3.0, 
            ticksize_small = 0, 
                )
    #elif fonts == 'normal':
    else: 
        context_dic = dict(
            text_large = 9, 
            text_small = 8, 
            linewidth_thick = 1, 
            linewidth_thin = 0.75,
            ticksize_large = 4.0, 
            ticksize_small = 0, 
                )

    context = context_settings(**context_dic)

    
    
    custom_params = {'text.color': tick_color, 'xtick.color': tick_color, 
                     'ytick.color': tick_color, 'pdf.fonttype' : 42, 'ps.fonttype' : 42 
                     }
    
    sns.set_theme(style= style, context = context, rc=custom_params)
    so.Plot.config.theme.update(**context)
    #so.Plot.config.theme(axes_style("whitegrid")
    #so.Plot.config.theme["axes.style"] = style
    so.Plot.config.theme.update(axes_style(style))
    
    #plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams.update(**context)
    #plt.style.use('./images/presentation.mplstyle')
def Plot_context(text_small = 6, text_large = 8, tick_color = '0.2'):
    SMALL_SIZE = text_small
    MEDIUM_SIZE = text_small
    BIGGER_SIZE = text_large

    so.Plot.config('font', size=SMALL_SIZE)          # controls default text sizes
    so.Plot.config('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    so.Plot.config('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    so.Plot.config('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    so.Plot.config('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    so.Plot.config('legend', fontsize=SMALL_SIZE)    # legend fontsize
    so.Plot.config('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    so.Plot.config('axes.edgecolor', tick_color)
    so.Plot.config('axes.xtick.color', tick_color)
    so.Plot.config('axes.ytick.color', tick_color)

def context_settings(text_large = 7, text_small = 6, 
                     linewidth_thick = 0.75, linewidth_thin = 0.5,
                     ticksize_large = 3.0, ticksize_small = 0, 
                     ):
    
    #text_medium = 7 
    
    context_dic = {}

    large_text = ["font.size", 'legend.title_fontsize', "axes.titlesize"]
    small_text = ["xtick.labelsize", "ytick.labelsize", "legend.fontsize", "axes.labelsize", ]

    
    context_dic.update({text:text_large for text in large_text})
    context_dic.update({text:text_small for text in small_text})


    thin_lines = ["axes.linewidth", "grid.linewidth",
        "patch.linewidth", "xtick.major.width", "ytick.major.width",
        "xtick.minor.width", "ytick.minor.width",]
    thick_lines = ["lines.linewidth", "lines.markersize",]

    context_dic.update({line:linewidth_thin for line in thin_lines})
    context_dic.update({line:linewidth_thick for line in thick_lines})
    
    large_ticks = ['xtick.major.size','ytick.major.size',]
    small_ticks = ['xtick.minor.size', 'ytick.minor.size']
    
    context_dic.update({tick:ticksize_large for tick in large_ticks})
    context_dic.update({tick:ticksize_small for tick in small_ticks})
    
    return context_dic

#'axes.linewidth': 0.8,
 #'grid.linewidth': 0.8,
 #'lines.linewidth': 1.5,
 #'lines.markersize': 6.0,
 #'patch.linewidth': 1.0,
 #'xtick.major.width': 0.8,
 #'ytick.major.width': 0.8,
 #'xtick.minor.width': 0.6,
 #'ytick.minor.width': 0.6,
    
    
    #return context_dic
    #sns.set_context(context_dic)
    
def pyplot_context(text_small = 6, text_large = 8, tick_color = '0.2'):
    SMALL_SIZE = text_small
    MEDIUM_SIZE = text_small
    BIGGER_SIZE = text_large

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.rc('axes.edgecolor', tick_color)
    plt.rc('axes.xtick.color', tick_color)
    plt.rc('axes.ytick.color', tick_color)



#---------------------------------------------------------------------------
# Functions for ordering plots 
#---------------------------------------------------------------------------

 
 #---------------------------------------------------------------------------
# CHECK ORDER FOR VAR, ONLY FOR CATEGORICAL VARIABLES

def var_order(var):
    if var == 'inh':
        order = dfc.inh_order
    #
    #    order = False
    elif var in cfg.rankCatVars.values():
        order = cfg.catRanks_
    
    elif var in cfg.var_order:
        order = cfg.var_order[var]
        #print(f'Found variable {var} in var_order dictionary')
    else: 
        print(f'Did not find variable {var} in var_order dictionary')
        order = False
    return order

#---------------------------------------------------------------------------
# CREATE DICTIONARIES WITH ORDERS FOR X, ROW, COL, HUE

#---------------------------------------------------------------------------
# ONLY CATEGORICAL VARIABLES 
def order_vars(x = 'count', hue = 'inh', col = 'tri_sec', row = 'quadrant', **kws):
    ordered_vars = [x, hue, col, row]
    var_names = ['x','hue','col', 'row']
    plot_dic = {key:value for key,value in zip(var_names, ordered_vars) if value}
    
    order_name_ls = ['order','hue_order', 'col_order', 'row_order']
    
    order_dic = {key:var_order(value) for key, value in zip(order_name_ls, ordered_vars) if value}
    order_dic.update(**{key:value for key, value in kws.items() if key in order_name_ls})  
    plot_dic.update({key:value for key,value in order_dic.items() if value})
    print(plot_dic)
    return plot_dic

#---------------------------------------------------------------------------
# CATEGORICAL AND NUMERICAL VARIABLES 

def order_vars2(df, x = False, hue = 'iso_A', col = 'tri_sec', row = 'quadrant',
                print_check = True, **kws):
    
    order_name_ls = ['order','hue_order', 'col_order', 'row_order']
    var_names = ['x','hue','col', 'row']
    ordered_vars = [x, hue, col, row]
    
    plot_dic = {key:value for key,value in zip(var_names, ordered_vars) if value}
    order_dic = {key:value for key,value in zip(order_name_ls, ordered_vars) if value}
    print(plot_dic, order_dic)
    
    
    ordered_vars = []
    order_names = []
    orders = []
    for order_name, var in order_dic.items(): 
        print(order_name, var)
        print(f'Ordering {var}')
        values = sorted(list(df[var].dropna().unique()))
        order = var_order(var)
        if order:
            order_ls = [n for n in order if n in values]
        else: 
            order_ls = values
        print(f'Order: {order_ls}')
        
        order_names.append(order_name)
        #ordered_vars.append(var)
        orders.append(order_ls)    
            
    order_dic = {order_name:order for order_name, order in zip(order_names, orders) if order}
    plot_dic.update({key:value for key,value in order_dic.items() if value})
    #print(plot_dic)
    return plot_dic

#---------------------------------------------------------------------------
# CATEGORICAL AND NUMERICAL VARIABLES, RETURNS FALSE IF DIM NOT IN PLOT

def order_vars3(df, 
                var_dic = {'col':'tri_sec', 'row': 'quadrant', 'hue': 'inh'},
                #dims_to_order = ['col', 'row'], #'x','hue',
                #vars = ['tri_sec', 'quadrant'],
                #x = False, hue = 'iso_A', col = 'tri_sec', row = 'quadrant',
                print_check = False, **kws):
    
    orderName_dic = {'x':'order','hue': 'hue_order', 'col':'col_order', 'row': 'row_order'}
    
    print(f'Order vars3 received the following dictionary for ordering: {var_dic}')
    
    order_dic = {}
    for key, value in var_dic.items():
        print(key, value)
        if df[value].cat.ordered:
           order_dic.update({orderName_dic[key]:df[value].cat.categories})
        elif key in orderName_dic.keys():
            order_dic.update({orderName_dic[key]:var_dic[key]})
        #elif df[value].cat.ordered:
         #   order_dic.update({orderName_dic[key]:df[value].cat.categories})
        
        else:
            order_dic.update({value:False})
    #print('Order_dic:', order_dic)
    
    #order_name_ls = [orderName_dic[val] for val in var_dic.keys()]
    #order_dic = {key, value else: key, False for key,value in orderName_dic.items() if value in order_name_ls}
    
    #print('order_name_ls', order_name_ls)
    #vars = [value for value in var_dic.values()]
    #dims_to_order = [key for key in var_dic.keys()]
    #order_name_ls = [value for key, value in orderName_dic.items() if key in dims_to_order]
    #order_name_ls = ['order','hue_order', 'col_order', 'row_order']
    #var_names = ['x','hue','col', 'row']
    #ordered_vars = [x, hue, col, row]
    
    
    #plot_dic = {key:value for key,value in zip(var_names, ordered_vars)}
    #order_dic = {key:value for key,value in zip(order_name_ls, ordered_vars)}
    #plot_dic = {key:value for key,value in zip(dims_to_order, vars)}
    #order_dic = {key:value for key,value in zip(order_name_ls, vars)}
    #order_dic = {key:value for key,value in zip(dims_to_order, orderNames) if }
    
    
    ordered_vars = []
    order_names = []
    orders = []
    for order_name, var in order_dic.items(): 
        order_names.append(order_name)
        if var:
            #print(order_name, var)
            if print_check: print(f'Ordering {var}')
            values = sorted(list(df[var].dropna().unique()))
            order = var_order(var)
            if order:
                order_ls = [n for n in order if n in values]
            else: 
                order_ls = values
            if print_check: print(f'Order: {order_ls}')
            
            #order_names.append(order_name)
            #ordered_vars.append(var)
            orders.append(order_ls)    
        else: 
            #order_names.append(order_name)
            #ordered_vars.append(var)
            orders.append(False)  
            
    order_dic = {order_name:order for order_name, order in zip(order_names, orders) }
    #var_dic.update({key:value for key,value in order_dic.items()})
    if print_check: print('order_dic', order_dic)
    return order_dic

#---------------------------------------------------------------------------
# COUNT NUMBER OF ROWS, COLS, DIMS
def ndims(df, plot_dic):
    dimNames = {'row': 'nrows', 'col': 'ncols'}
    OrderNames = {'row': 'row_order', 'col': 'col_order'}
    dim_dic = {}
    
    #if 'row_order' in plot_dic.keys():
    #    nrows = len(plot_dic['row_order']) if plot_dic['row_order'] else 1
    for key, value in plot_dic.items():
        #if df[value].cat.ordered:
        #    order_dic.update({orderName_dic[key]:df[value].cat.categories})
        if value:
            if key in dimNames.keys():
                    dim_dic[dimNames[key]] = len(df[value].dropna().unique())
                    
                    if df[value].dtype == 'category':
                        print('CATE¤GORY!!!')
                        if df[value].cat.ordered:
                            print('ORDERED!!!')
                            dim_dic[OrderNames[key]] = df[value].cat.categories
                        else:
                            dim_dic[OrderNames[key]] = sorted(list(df[value].dropna().unique()))
                    else:
                        print(df[value].dtype)
                        dim_dic[OrderNames[key]] = sorted(list(df[value].dropna().unique()))
                    
                    
            
    dim_dic.update(**{value: 1 for key, value in dimNames.items() if value not in dim_dic.keys()})
    dim_dic.update(**{value: False for key, value in OrderNames.items() if value not in dim_dic.keys()})
    
    print(dim_dic)
    
    #dim_dic.update(**{ndims: 1 for value in dim_dic.values() if value == 1})
        
    
    
    #else: nrows = 1
    
    #if 'col_order' in plot_dic.keys():
    #    ncols = len(plot_dic['col_order']) if plot_dic['col_order'] else 1
    #else: ncols = 1
    
    if dim_dic['nrows'] > 1 and dim_dic['ncols'] > 1: 
        dim_dic['ndims'] = 2
    
    elif dim_dic['nrows'] > 1 or dim_dic['ncols'] > 1: 
        dim_dic['ndims'] = 1
    else: 
        dim_dic['ndims'] = 0
    
    print('dim_dic:', dim_dic)
    #plot_dic.update({'ndims': ndims})
    return dim_dic#{'nrows': nrows, 'ncols': ncols, 'ndims': ndims}    
    
#---------------------------------------------------------------------------
# Initiate FIGURE MATPLOTLIB SUBPLOTS

def init_fig(#type = 'std', 
             nrows = 1, ncols = 3, 
             sharey = True, sharex = True, squeeze = False, constrained = True,
             height = 1.2, width = 1.4,
             **kws):
        
    
    
    #elif type == 'std':
    #    height = 1.2#1.4 #1.6
    #    width = 1.4#2.4#3#2.2
        
    fig, axs = plt.subplots(nrows, ncols, figsize=(width*ncols,height*nrows), 
                            squeeze = squeeze, constrained_layout = constrained, 
                            sharey = sharey, sharex = sharex)
    
    
    return fig, axs

#---------------------------------------------------------------------------
# Filter df for ax_fig
#def iterate_df_axes(df, row , col, row_order , col_order):

def filter_df(dfz, dim, n_dim, dim_order, 
              print_check = False, 
              **kws):
    if dim: 
        if dim_order:
            value = dim_order[n_dim]
            if print_check: print(f'{dim}_value: {value}')
            df_i = dfz[dfz[dim] == value]#.copy()
        else: 
            df_i = dfz#.copy()
    else: df_i = dfz#.copy()
    return df_i

#---------------------------------------------------------------------------
# Plot on axis

def ax_plot(df, plot_dic, ax, kind, legend, palette = 'tab20', ylims = False, s_size = 3, **kws):
    
    
    #plot_dic.update({'palette': palette})
    if kind == 'bar_strip':
        bar_params = {'errorbar':'se', 'palette' : palette, 'alpha': 0.75 , 
              'capsize': 0.4,
              'err_kws': {"color": ".3", "linewidth": 1.25, 'alpha' : 0.8} }#'capsize' :.4, "linewidth": 0.8, 'zorder': 99, 
        bar_params.update(**plot_dic)
        #'alpha': 0.5 
        # #'lw' : 1.5, 'edgecolor' : ".5", 'facecolor' : (0, 0, 0, 0)
        #'capsize' :.4, 'err_kws': {"color": ".5", "linewidth": 1.5} 
        
        strip_params = {'palette' : palette, 'dodge' : True, 'jitter': 0.2, 'alpha' : 0.5, 'size' : s_size, }#
        #'edgecolor' : 'grey', 'linewidth':0.2  
        strip_params.update(**plot_dic)
        
        sns.barplot(data = df, **bar_params, ax= ax, legend = legend)#legend=False
        sns.stripplot(data = df, **strip_params, ax= ax, legend = False)
        #ax = sns.barplot(data = df, **bar_params, legend = legend)#legend=False
        #ax = sns.stripplot(data = df, **strip_params, legend = False)
        if ylims:
            ax.set_ylim(ylims)
    
    elif kind == 'bar_strip h':
        bar_params = {'errorbar':'se', 'palette' : palette, 
              'capsize': 0.4, 'linewidth' : 1.5, 'edgecolor' : ".5", 'gap': 0.1,#'width': 0.5,
              'err_kws': {"linewidth": 0.8,} }#"color": ".3", 'capsize' :.4, "linewidth": 0.8, 'zorder': 99, 
        bar_params.update(**plot_dic)
        #'alpha': 0.5 
        # #'lw' : 1.5, 'edgecolor' : ".5", 'facecolor' : (0, 0, 0, 0)
        #'capsize' :.4, 'err_kws': {"color": ".5", "linewidth": 1.5} 
        
        strip_params = {'palette' : palette, 'dodge' : 0.3, 'jitter': 0.2, 'size' : s_size, 'alpha' : 0.7, }# 
        #'edgecolor' : 'grey', 'linewidth':0.2  
        strip_params.update(**plot_dic)
        
        sns.barplot(data = df, **bar_params, ax= ax, legend = legend)#legend=False
        sns.stripplot(data = df, **strip_params, ax= ax, legend = False)
        
        for patch in ax.patches:
            clr = patch.get_facecolor()
            patch.set_edgecolor(clr)
            patch.set_facecolor('none')
    
    elif kind == 'stacked_hbar':
        print('stacked_hbar')
        #'right'

#---------------------------------------------------------------------------
# Make composite figure using matplotlib
def ax_fig(dfg, x, y, row, col, hue, 
             #x = 't_lim', 
             #y = 'count',
             #dims_to_order= ['col', 'row', 'hue'],
             kind = 'bar_strip', 
             palette = 'PuBu', 
             sharey= False, 
             print_check = False,
             height = 1.2, 
             width = 1.4,
             #ylims = False,
             **kws):
    
    
    fig_dic = dict(x = x, y = y, row = row, col = col, hue = hue, sharey =sharey, palette = palette)
    #fig_dic = {key:value for key, value in fig_dic.items() if value}
    
    #ordering_dic = {key:value for key,value in fig_dic.items() if key in dims_to_order}
    
    #fig_dic.update(**order_vars3(dfg, ordering_dic))#dims_to_order, vars= ['quadrant', False, 'inh']
    #fig_dic.update({'x': 't_lim', 'y': 'count'})
    fig_dic.update(**kws)
    fig_dic.update(**ndims(dfg, fig_dic))
    
    
    if print_check: print('Fig_dic:', fig_dic)

    #ax_params = ['x', 'y', 'hue', 'order', 'hue_order', 'palette']
    ax_params = ['x', 'y', 'hue', 'palette']
    ax_params = [var for var in ax_params if var]

    plot_dic = {key:value for key, value in fig_dic.items() if value and key in ax_params}
    if print_check: print('plot_dic', plot_dic)

    fig, axs= init_fig(height = height, width = width, **fig_dic)
    #legend_axs = fig.add_axes([.91, .3, .03, .4])

    for n_row in range(fig_dic['nrows']):
        #print(fig_dic['row'], fig_dic['row_order'][n_row])
        dfi = filter_df(dfg, fig_dic['row'], n_row, fig_dic['row_order'])
        #ax1 = axs[n_row, :]
        #axs = axis[n_row]
        for n_col in range(fig_dic['ncols']):
            #ax = ax1[n_col]
            ax = axs[n_row, n_col]
            #print(fig_dic['col'], fig_dic['col_order'][n_col])
            dfii = filter_df(dfi, fig_dic['col'], n_col, fig_dic['col_order'])
            
            #display(dfi.head())
            if n_row == fig_dic['nrows']-1 and n_col == fig_dic['ncols']-1:
                ax_plot(dfii, plot_dic, ax, kind, legend = True, palette = palette, **kws)
                
            else: 
                ax_plot(dfii, plot_dic, ax, kind, legend = False, palette = palette, **kws)
            
            if fig_dic["col_order"] and n_row == 0 :
                ax.set_title(f'{fig_dic["col_order"][n_col]}')#.title()
            if fig_dic["row_order"] and n_col == 0:
                ax.set_ylabel(f'{fig_dic["row_order"][n_row]}')#.title()
                
    #plt.xticks(rotation=45)  
    for ax in axs.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('center')#label.set_ha('right')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #if ylims:
        #for ax in axs.flatten():
        #    ax.set_ylim(ylims)
        #plt.ylim(ylims)
    fig_name = f'{kind} x_{x} y_{y} hue_{hue} row_{row} col_{col}'
    if 'xtra' in kws.keys():
        print('xtra:', kws['xtra'])
    else: 
        print('no xtra')
    fig_name += f' {kws["xtra"]}'
    dfc.save_fig(fig_name)


#---------------------------------------------------------------------------
# SET TITLES ON SEABORN FIGURE LEVEL PLOT

def p_titles(g, col, row):
    if col:
        if row:
            g.set_titles("{col_name}\n:{row_name}")
        else: 
            g.set_titles("{col_name}")
    elif row:
        g.set_titles("{row_name}")
    else:
        g.set_titles("")
    return g
    
#---------------------------------------------------------------------------


def semantic_params(dfi, col = 'inside_injury', row = False, hue = 'inh', palette = 'Dark2'):#, style = False
    
    
    inh_order = dfc.inh_order
    params = {}
    dim_keys = ['col','row','hue']#, 'style'
    dim_labels = [col, row, hue]#, style
    params.update({ key:label for key,label in zip(dim_keys,dim_labels)}) #if label
    
    #print(params)
    dims = 0
    pms = {}
    for key,label in params.items():
        
        if label:
           # print(dfi.columns, params[key])
            dim_values = sorted(dfi[params[key]].unique().tolist(),reverse=False)
            if label == 'inh':
                vals = [inh for inh in dfc.inh_order if inh in dim_values]
                if key == 'hue':
                    palette = palette
                
            elif label in cfg.var_order.keys():
                vals = [val for val in cfg.var_order[label] if val in dim_values]
            
            else: 
                vals = dim_values    
            
            pms.update({f'{key}_order' : vals})
        
           # if key in ['col','row']:
            #    dims += 1
    
    params.update(pms)
    params.update({'palette': palette}) 
    return params

#---------------------------------------------------------------------------

    #for var,var_str in zip([params['row'], params['col'], params['hue']],['row','col','hue']):
    #    if var == 'inh':
    #        params.update({f'{var_str}_order' : dfc.inh_order})
    #    else:
    #        if var in cfg.var_order.keys():
    #            params.update({f'{var_str}_order': cfg.var_order[var]})
    
    
    
def params_choice(choice = 'line'):
    #if choice == 'line':
    #    params.update({'ndims' : dims,
    #               'x_var':x_var, 
    #               'palette': palette, 
    #               'facet_kws': {'sharey': sharey},
    #               'errorbar': 'se',
    #               'height': 2.0})  
    #global pms
    if choice == 'line':
       pms=dict(x='sec',y='roll',hue='inh',errorbar = 'se', kind="line", hue_order=dfc.inh_order,
                     height=3,aspect=1.25, palette = pal1)
    
    if choice == 'line_size':
       pms=dict(x='sec',y='roll',hue='size',errorbar = 'se', kind="line", hue_order=dfc.inh_order,
                    height=3,aspect=1.25, palette = pal1)
        
    elif choice == 'line_indexp':
       pms=dict(x='sec', y='roll',hue='path',kind='line',row='inh', row_order=dfc.inh_order, 
                    height=3, aspect=1.4,legend='full')
    elif choice == 'point':
        pms=dict(x="minute",hue='inh',hue_order=dfc.inh_order, ci=70, kind="point", height=6,aspect=1.25, palette = pal1)
    elif choice == 'heat':
        pms=dict(orient='horizontal',groups=dfc.inh_order,var='ca_corr', smooth='gauss',count_thr=0, 
                xticklabels=16,yticklabels=14, cmap='seq',vmax=100,vmin=0)#yticklabels=2
    elif choice == 'line_indexp2':
        pms=dict(x='sec', y='roll',kind='line',col='inh',row='inh_exp_id',col_order=dfc.inh_order, 
                    height=3, aspect=1.4,legend='full')

#---------------------------------------------------------------------------
                    
def plot_params(x_var = 'sec', y_var = 'roll', 
                sharey = False, kind = 'line', 
                old_pms = {}):#, select_vars=False,select_keys=False
    #params_dict = dict(
    params = {
        'x': x_var,
        'y': y_var,
        'errorbar': 'se',
        'height': 2.0,
        'aspect' : 1.0, #1.25, 
        #'y' : 'roll', 
        #'margin_titles' : True,
    }
    
    pms = dict(
    line = { 
        #'ndims' : dims,
        'facet_kws': {'sharey': sharey},
        'kind' : 'line'
    },
    
    )
    
    if kind in pms.keys():
        params.update(pms[kind]) 
    params.update({key:label for key,label in old_pms.items() if label})
    
    return params


#---------------------------------------------------------------------------
# Functions for colormap normalization etc.  
#---------------------------------------------------------------------------
    
def colormap_midpoint(c_map, values, midpoint = 38, round = 2, **kws):
        #COLORMAP PROCESSING!!
        #levels = 30
        #midpoint = 38
        cmap = cm.get_cmap(c_map)
        colormap = c_map if isinstance(c_map, str) else c_map.name
        
        
        vmin, vmax = min(values), max(values) 
        levels = np.round(values, round)#np.linspace(vmin, vmax, num_levels)
        levels[0] = m.floor(values[0])
        levels[-1] = m.ceil(values[-1])
        #print(len(levels))
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        #vals = np.concatenate(([0],vals,[1]))
        #print(len(midp))
        #print(len(vals))                  
        colors = cmap(vals)#np.arange(cmap.N)
        #colors = plt.cm.seismic(vals)
        cmap, norm = from_levels_and_colors(levels, colors, )#extend = 'max'#extend = 'both'
        
        return cmap, norm, colormap
        
#--------------------------------------------------------------------------------

def colormap_quartiles(c_map, values, low = 25, mid = 37.5, high = 75, round = 2, **kws):
            #--------------------------------------------------------------------------------
        #COLORMAP PROCESSING!!
        #levels = 30
        #midpoint = 38
        cmap = cm.get_cmap(c_map)
        colormap = c_map if isinstance(c_map, str) else c_map.name
        
        
        vmin, vmax = min(values), max(values) 
        levels = np.round(values,round)#np.linspace(vmin, vmax, num_levels)
        levels[0] = m.floor(values[0])
        levels[-1] = m.ceil(values[-1])
        #print(len(levels))
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, low, mid, high, vmax], [0, 0.25, 0.5, 0.75, 1])
        #vals = np.concatenate(([0],vals,[1]))
        #print(len(midp))
        #print(len(vals))
        colors = cmap(vals)#np.arange(cmap.N)
        #colors = plt.cm.seismic(vals)
        cmap, norm = from_levels_and_colors(levels, colors, )#extend = 'max'#extend = 'both'
        
        return cmap, norm, colormap
#--------------------------------------------------------------------------------
      
def colormap_edges(c_map, values, round = 0, **kws):
           
        cmap = cm.get_cmap(c_map)
        colormap = c_map if isinstance(c_map, str) else c_map.name
        
        #bounds = np.round(values,round)
        bounds = np.ceil(values)
        
        norm = colors.BoundaryNorm(bounds, cmap.N, clip = True)#clip = True, extend = 'both'
        #norm = colors.BoundaryNorm(bounds, cmap.N)#, extend='both'
        #cmap = cm.ScalarMappable(norm=norm,cmap=cmap) # IF WORKING WITH SUBPLOTS TEST LINE BELOW INSTEAD INSIDE PLOT
        
        #plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=axs[n])
        
        return cmap, norm, colormap
        
        #bounds = np.round(iso_hues,0)
        #norm = colors.BoundaryNorm(bounds, cmap.N)#, extend='both'
        #cmap_d=cm.ScalarMappable(norm=norm,cmap=cmap)

#--------------------------------------------------------------------------------
     
class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self,vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint=midpoint
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(np.log(value), x, y))
        #--------------------------------------------------------------------------------




# -----------------------------------------------------
                  #SEABORN FIGURE LEVEL PLOTS
                  
def do_sns_fig(data, y, 
               query = False,  
             x = 'a_bin', hue = 'inh', col = 'tri_sec' , row = 'quadrant', 
             height = 1.0,  aspect = 1, errorbar = 'se', kind ='line', palette = 'torch', lw = 1,
             facet_kws=dict(margin_titles=True), 
             **kws):
    if query:
        data = data.query(query) 
    #plot_dic = dict(data = data, y = y)
    plot_dic = {}
    ax_dic = dict(x = x, y = y, hue = hue, col = col, row = row)
    plot_dic.update(ax_dic)
    ae_dic = dict(height = height,  errorbar = errorbar, kind = kind, palette = palette, facet_kws = facet_kws, lw = lw)
    plot_dic.update(ae_dic)
    order_dic = order_vars(**ax_dic)
    plot_dic.update(order_dic)
    plot_dic = {key:value for key, value in plot_dic.items() if value}
    plot_dic.update({'data':data})
    kws.update({'axes': ax_dic,
                  'ae': ae_dic,
                  'order': order_dic,
                  'plot': plot_dic, 
                  }
    )
    
    g = draw_sns_fig(plot_dic, **kws )
    
    return g
    

def draw_sns_fig(plot_dic, style = 'ticks', type = 'rel', log = (False, False), despine = False, compact = False, **kws):
    
    
    dim_dic = {}
    
    if 'row_order' in plot_dic.keys():
        nrows = len(plot_dic['row_order']) if plot_dic['row_order'] else 0
    else: nrows = 0
    
    if 'col_order' in plot_dic.keys():
        ncols = len(plot_dic['col_order']) if plot_dic['col_order'] else 0
    else: ncols = 0
    
    
    if nrows > 1 and ncols > 1: 
        ndims = 2
    
    elif nrows > 1 or ncols > 1: 
        ndims = 1
    else: 
        ndims = 0
    
    
    with sns.axes_style(style):
    
        if type == 'rel':
            g = sns.relplot(**plot_dic)
            if log[0]:
                g.set(xscale="log")
            if log[1]:
                g.set(yscale="log")
                
        elif type == 'dis':
            bad_vars = ['errorbar']
            plot_dic = {key:value for key, value in plot_dic.items() if key not in bad_vars}
            disVars = ['stat', 'complementary', 'weights']
            plot_dic.update(**{key:value for key,value in kws.items() if key in disVars})
            plot_dic.update(**{'log_scale':log})
            plot_params = [f'{key}: {value}' for key, value in plot_dic.items() if key != 'data']
            for par in plot_params:
                print(par)
            g = sns.displot(**plot_dic)
    
    if ndims == 2:
        g.set_titles("{col_name}\n{row_name}")
    elif nrows > 1:
        g.set_titles("{row_name}")
    elif ncols > 1:
        g.set_titles("{row_name}")
        
    
    if compact: g.figure.subplots_adjust(wspace=0.1, hspace=0.1)
    if despine:
        for (row_val, col_val), ax in g.axes_dict.items():
        #for ax in axs.flat:
            ax.spines.bottom.set_visible(True)
            ax.spines.left.set_visible(True)
            ax.yaxis.reset_ticks()
            
            #ax.xaxis.set_ticks_position('default')
            
        #ax.tick_params(reset = True)
            #if row_val != plot_dic['row_order'][-1]:
            #    ax.spines.bottom.set_visible(False)
                #ax.xaxis.set_ticks_position('none')
            if col_val != plot_dic['col_order'][0]:
                ax.spines.left.set_visible(False)
                ax.yaxis.set_ticks_position('none')
    
    return g
        

#------------------------------------------------------
#LINEPLOTS COUNTS/FRACTIONS
#------------------------------------------------------
#,grouping_var='inside_injury',x_var='sec'
def relplot(dfi, x_var = 'sec', y_var = 'roll', 
            row = False, col = 'inside_injury', hue = 'inh', style = 'ticks',
            kind = 'line', sharey = False,
            palette = 'Dark2'):
    #print(palette)
    start_params = semantic_params(dfi, col = col, row = row, hue = hue, palette = palette)#style = 'ticks'#, sharey = False
    #print(start_params['palette'])
    
    if y_var == 'roll':
        roll_hue = []
        for var in ['row', 'col', 'hue']:#, 'style']
            if start_params[var]:
                roll_hue.append(start_params[var])
        
        dfg=dfc.rolling_count(dfi,hue_var = roll_hue)
    else:
        #dfg=dfc.rolling_count(dfi,hue_var = roll_hue)
        dfg = dfi
    
    
    
    params = plot_params(x_var, y_var, sharey = sharey, kind = kind, old_pms = start_params)
    
    params.update({'data' : dfg})
    
    
    params.update({'facet_kws' : {'sharey':sharey}, 'height': 1.2})#'style' : style, 
    #print(params)
    
    #with sns.plotting_context("paper") and sns.axes_style(style):#"whitegrid""notebook"
        #print(params['palette'])
    g = sns.relplot(**params)
    if col:
        if row:
            g.set_titles("{col_var}:{col_name}\n{row_var}:{row_name}")
        else: 
                g.set_titles("{col_var}:{col_name}")
    elif row:
        g.set_titles("{row_var}:{row_name}")
    else:
            g.set_titles("")
    
    #if sharey:
    #    if sharey == 'row':
    #        facet_kws={'sharey': 'row', 'sharex': True}
    #    elif sharey == 'col':
            
    g.set_ylabels('Platelet count')
    g.set_xlabels(f'Time ({x_var})')
    g._legend.set_title(leg_titles[params['hue']] if params['hue'] in leg_titles else hue)
    dfc.save_fig(f'{kind}plot {y_var} col_{col} row_{row} hue_{hue}')
    plt.show()
    return params
    



    
#------------------------------------------------------
#LINEPLOTS MEANS OF VARIABLES
#------------------------------------------------------
def run_meantime_plots(df,col_var):
    df=dfc.scale_vars(df)
    df['nrtracks']*=3.1
    lineplots_meantime=dict(
    stab={
        'y_var':'stab',
        'sharey': False,
        'sup':'Stability',
        'axhline':False,
        'y_lab':r"Distance to closest \nneighbour in next frame ($\mu$m)",
    },
    
    #cont_tot={
    #    'y_var':'cont_tot',
    #    'sharey': False,
    #    'axhline':False,
    #    'sup':'Mean total platelet contraction',
    #    'y_lab':"Total platelet contraction ($\mu$m)",
    #},
    nba_d_10={
        'y_var':'nba_d_10',
        'sharey': False,
        'axhline':False,
        'sup':'Average distance to 10 closest platelets',
        'y_lab':r"Average distance ($\mu$m)",
    },
    ys={
        'y_var':'ys',
        'sharey': False,
        'axhline':False,
        'sup':'Thrombus center of gravity, flow axis',
        'y_lab':"Mean position",
    },
    ca_corr={
        'y_var':'ca_corr',
        'sharey': False,
        'axhline':False,
        'sup':'Platelet corrected calcium levels',
        'y_lab':"Corrected CAL520 fluorescence (AU)",
    },
        nrtracks={
        'y_var':'nrtracks',
        'sharey': False,
        'axhline':False,
        'sup':'Average tracking time per platelet',
        'y_lab':"Tracking time (s)",
    },
    cont_s={
        'y_var':'cont_s',
        'sharey': True,
        'axhline':True,
        'sup':'Mean platelet contraction',
        'y_lab':"Contraction (nm/s)",
    },
        dvz_s={
        'y_var':'dvz_s',
        'sharey': False,
        'axhline':True,
        'sup':'Average axial movement',
        'y_lab':"Movement in z (nm/s)",
    },
         cont_xy_s={
        'y_var':'cont_xy_s',
        'sharey': True,
        'axhline':True,
        'sup':'Mean platelet contraction in XY-plane',
        'y_lab':"Contraction (nm/s)",
        },

        cont_cum={
        'y_var':'cont_cum',
        'sharey': True,
        'axhline':True,
        'sup':'Cumulative platelet contraction (all dims)',
        'y_lab':r"Contraction ($\mu$m)",
    },
        cont_xy_cum={
        'y_var':'cont_xy_cum',
        'sharey': True,
        'axhline':True,
        'sup':'Cumulative platelet contraction in XY-plane',
        'y_lab':r"Contraction ($\mu$m)",
    },
        dvz_cum={
        'y_var':'dvz_cum',
        'sharey': True,
        'axhline':True,
        'sup':'Cumulative platelet movement in axial plane',
        'y_lab':r"Movement ($\mu$m)",
    },
   
    )
    columns_=df.columns.tolist()
    params=dict(col=col_var, col_order=cfg.var_order[col_var])
    y_vars=list(lineplots_meantime.keys())
    for y_var in y_vars:
        if '_s' in y_var:
            if y_var not in columns_:
                if y_var.removesuffix('_s') in columns_:
                    y=dfc.scale_var(df,y_var.removesuffix('_s'))
    dfg = dfc.rolling_means(df,col_var,y_vars,'sec')
    for key,dic in lineplots_meantime.items():
        if key in columns_:
            print(f'Plotting {key}')
            params.update(dict(data=dfg,y=dic['y_var'],facet_kws={'sharey': dic['sharey']}))#data=df.dropna(subset=[dic['y_var']])
            g=lineplot_meantime(**params)
            g.set_ylabels(dic['y_lab'])
            if dic['axhline']:
                g.map(plt.axhline, y=0, ls="-", c=".5")
            g.set_xlabels("Time (s)")
            plt.subplots_adjust(top=0.85)
            g.fig.suptitle(dic['sup'],fontsize=20) 
            #if params['col']:
            #    g._legend.set_title(leg_titles[params['col']])
            g._legend.set_title(leg_titles['inh'])
            y_var=dic['y_var']
            dfc.save_fig(f'lineplot meantime {y_var}')
            plt.show()
        else:
            print(f'Missing variable {key}')

# TIME ON X-AXIS 
#------------------------------------------------------
def lineplot_meantime(**xparams):# Generic function for lineplots showing means over time
    params={'hue':'inh','hue_order':dfc.inh_order,'col':'inside_injury','x':'sec','ci':70, #ci=95
            'kind':'line','y':'stab','facet_kws':{'sharey': False},'aspect':1,'height':3}#'legend':False
    params.update(xparams)
    return sns.relplot(**params)


def mirror_lineplot(dfgc, cmap = 'cividis'):

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6,2.5))

    for ax, inh, invert in zip(axes.ravel(), dfc.inh_order, [False, True]):
        sns.lineplot(data = dfgc[dfgc.inh == inh], y = 'c_per', x = 'sec', estimator = None, units='rank', hue = 'rank', ax = ax, palette = cmap , legend = False )#palette = 'viridis', palette = plot.pal1, style = 'size'
        ax.axhline(100, ls='--', c = 'grey')#c='green'black
        
        if invert:
            ax.invert_xaxis()
            ax.spines['left'].set_visible(False)
            #ax.set_yticks([])
            #ax.spines['right'].set_position('zero')
            ax.yaxis.set_ticks_position('none') 
        else:
            ax.spines['right'].set_visible(False)
        
        
    plt.subplots_adjust(wspace=0)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    dfc.save_fig(f'mirror lineplot {cmap}')


# OTHER VARIABLES ON X-AXIS
#------------------------------------------------------

# Generic function for lineplots showing means of two varibles
def lineplot_meanvar(df,x_var,y_var,**xparams):
    params={'hue':'inh','hue_order':dfc.inh_order,'col':'phase','x':x_var,'ci':95,
            'kind':'line','y':y_var}#'facet_kws':{'sharey': False},'legend':False
    params.update(xparams)
    return sns.relplot(data=df, aspect=1.25, height=6,**params)



#Countplots
#------------------------------------------------------

def lineplot_count_isovol(df): #LINEPLOT WITH FRACTION NEW, UNSTABLE & NET DIFF PLATELETS
    df=df[(df.iso_vol<100)]
    params=dict(x='iso_vol',y='roll',hue='inh',hue_order=dfc.inh_order,ci=70, kind="line",  
                col='phase',col_order=cfg.phase_order,height=4,aspect=1.25,legend=True)
    x_var=params['x']
    grouping_var=['phase']
    dfg=dfc.rolling_timecount(df,grouping_var,[x_var])
    params.update({'data':dfg,'col':'phase','col_order':cfg.var_order['phase']})
    g=sns.relplot(**params)
    g.map(plt.axvline, x=37.5, ls="--", c=".5")
    g.set_ylabels('Platelet count')
    g.set_xlabels("Isovolumetric outer radius")
    #g.set(xlim=[0,100])#125
    g._legend.set_title(leg_titles[params['hue']])
 #   plt.subplots_adjust(top=0.92)
    dfc.save_fig(f'Count isovol',f'lineplot')
    plt.show()

def lineplot_countmov_isovol(df,grouping_var='movement'): #LINEPLOT WITH FRACTION NEW, UNSTABLE & NET DIFF PLATELETS
    df=df[(df.iso_vol<100)]
    params=dict(x='iso_vol',y='roll',hue=grouping_var,ci=70, kind="line", col='inh',col_order=dfc.inh_order, 
                row='phase',row_order=cfg.phase_order,height=4,aspect=1.25,legend=True)
    x_var=params['x']
    dfg=dfc.rolling_timecount(df,[grouping_var,'phase'],[x_var])
    params.update({'data':dfg})#,'col':'phase','col_order':cfg.var_order['phase']
    g=sns.relplot(**params)
    g.set_ylabels('Platelet count')
    g.set_xlabels("Distance from center")
    g._legend.set_title(leg_titles[grouping_var])
 #   plt.subplots_adjust(top=0.92)
    dfc.save_fig(f'Count {grouping_var}',f'lineplot')
    plt.show()
    
#Lineplots with individual thrombi & mean
#------------------------------------------------------
    
def lineplots_ind_mean(dfgc, cmap = 'cividis', y = 'c_per', x = 'sec', rank = 'rank_auc', x_lines = [100, 300], y_lines = [100,]):
    palette= dict(zip([0, 0.25, 0.5, 0.75], sns.color_palette("flare", 4)))
    g = sns.relplot(data = dfgc, y = y, x = x, lw = 2.0, height = 1.2, alpha = 1.0, zorder = 10, col = 'size_quart', hue = 'size_quart', palette = palette , 
                    row = 'inh', kind = 'line', errorbar = None , legend = False )
    
    for (row_val, col_val), ax in g.axes_dict.items():
        print(row_val, col_val)
        sns.lineplot(data = dfgc[(dfgc.inh == row_val) & (dfgc.size_quart == col_val)], y = y, x = x, lw = 1.0, alpha = 0.5, hue = 'size_quart', 
                estimator = None, units= rank, ax = ax, palette = palette ,
                legend = False )

        ax.spines['left'].set_visible(False)
        #ax.axhline(100, ls='--', c = 'grey')#c='green'black
        #    ax.invert_xaxis()
        #ax.spines['right'].set_position('zero')
        #ax.set_yticks([])
        ax.yaxis.set_ticks_position('none')
        ax.spines['right'].set_visible(False)
        
    if x_lines:
            for x in x_lines:
                g.map(plt.axvline, x=x, ls="--", c=".5")
    
    if y_lines:
            for y in y_lines:
                g.map(plt.axhline, y=y, ls="--", c=".5")
        
        
    plt.subplots_adjust(wspace=0)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    dfc.save_fig(f'ind_lineplot {cmap} {y} {rank}')
    
# Heatmaps with individual thrombi & mean
#------------------------------------------------------

from scipy import ndimage
def heatmap_filter(dfg1,smooth):
    #print('filter:')
    dfg_n=dfg1.to_numpy()
    #print(dfg_n)
    if smooth == 'uniform':
        dfg_n=ndimage.uniform_filter(dfg_n, size=2)
    elif smooth == 'gauss':
        dfg_n=ndimage.gaussian_filter(dfg_n, sigma=2, mode = 'nearest' )#sigma=3)
    elif smooth == 'gauss_s1':
        dfg_n=ndimage.gaussian_filter(dfg_n, sigma=1)
    elif smooth == 'gauss1D':
        dfg_n=ndimage.gaussian_filter1d(dfg_n, sigma=2)
    dfg1.loc[:,:] = dfg_n
    #print('OK')
    #dfg1=dfg1.fillna(0)
    return dfg1

def heatmap_ind_exp(df_a, rank = 'rank_auc', values = 'diff', 
                    center = 0, v1 = (-100, 300), v2 = (-40, 150), 
                    c_map = 'RdBu_r'):
    norm = dict(vmin = v1[0], vmax = v1[1])
    gauss = dict(vmin = v2[0], vmax = v2[1])
    settings = dict(norm = norm, gauss = gauss)
    


    fig, axs = plt.subplots(2,2, figsize = (8,4))

    for row, s in enumerate(['norm', 'gauss']):
        for col, q in enumerate(dfc.inh_order):
            print(row, col, q)
            ax = axs[row, col]
            test = df_a[df_a['inh'] == q].pivot(index = rank, columns = '10_sec', values = values).fillna(0)
            if s == 'gauss':
                test = heatmap_filter(test,s)
            if values == 'diff':
                sns.heatmap(test, cmap = c_map, norm = TwoSlopeNorm(center, **settings[s]), square = True, ax = ax)
            elif values == 'count':
                sns.heatmap(test, cmap = c_map, vmin = settings[s]['vmin'], vmax = settings[s]['vmax'], square = True, ax = ax)

            #ax.set_title(norm)


    #test = df_a.query(query).pivot(index = 'rank', columns = '10_sec', values = 'diff').fillna(0)#[(df_a.inh == 'MIPS')]
    #test = heatmap_filter(test,'uniform')
    #sns.heatmap(test, cmap = 'RdBu_r', norm = TwoSlopeNorm(center, vmax = vmax, vmin=vmin, ), square = True)
    dfc.save_fig(f'Heatmap ind exp MIPS diff {values} {rank}')
    
#------------------------------------------------------
# PLOTS WITH CATEGORICAL X-AXIS
#------------------------------------------------------

def catplot_meantime(df,**xparams):# General function for catplots showing means over time
    params={'hue':'inside_injury','hue_order':cfg.bol_order,'col':'inh','x':'minute','col_order':dfc.inh_order,'col_wrap':3,'ci':95,
            'kind':'point','y':'dvz_s'}
    params.update(xparams)
    return sns.catplot(data=df, height=4,**params)


def catplot(df, hue = 'inh', y = 'area', x = 'stability', col = 'quadrant', row = False,
            height = 1.3, aspect = 1, errorbar = 'se', cond = False, palette = 'torch', title = 'count', 
            kind = 'bar', sharey = 'row', style = 'whitegrid', **xtra):
    print('Starting catplot...')
    tick_params = dict(
        width = 0.5,
        labelsize = 6,
        labelfontfamily = 'arial',
        grid_linewidth = 0.5,
        )
    
    palette = sns.color_palette(palette, n_colors=len(df[hue].unique()))

    func_dic = dict(height = height, aspect = aspect, errorbar = errorbar, palette = palette, kind = kind, y = y, sharey = sharey)
    
    plot_dic = {key:value for key, value in func_dic.items() if value}
    #order_vars = order_vars(x, hue, col, row)
    plot_dic.update(**order_vars(x, hue, col, row))
    #plot_dic.update(**xtra)
    print('plot_dic', plot_dic)
    if kind == 'point':
        plot_dic.update({'markersize': 3})
    
    with sns.axes_style(style):
    
        g = sns.catplot(data = df, linewidth = 1.5, **plot_dic)#facet_kws=dict(margin_titles=True), 
        #g.tick_params(axis='both', **tick_params)
        
        g = p_titles(g, col, row)
        #g.set_titles("{col_name} {row_name}")
        

    # iterate through the axes
        if cond == 'per_ctrl':
            g.set_axis_labels("", f'{title} ratio')
            axes = g.axes.flatten()
            for i, ax in enumerate(axes):
                ax.axhline(1, ls='--', c='grey')
                #ax.axvline(la[i][1], ls='--', c='purple')
            #g.map(ax.axhline(y = 1, ls = '--', c = 'grey' ))
            g.fig.suptitle(title,fontsize=14)
            #if col:
            #    if row:
            #        g.set_titles("{col_name}\n:{row_name}")
            #    else: 
            #        g.set_titles("{col_name}")
            #elif row:
            #    g.set_titles("{row_name}")
            #else:
            #    g.set_titles("")
            #g.tight_layout()
            for n, axis in enumerate(g.axes.flat):
                print(n)
                if n > 0:
                    axis.tick_params(left=False, length=0)
                    axis.tick_params(bottom=True, length=3), 
                    #ax.set_yticks(np.arange(0, 1, 0.05), minor=True)
                    #axis.set_yticks([])
                    #plt1.tick_params(labelleft=False, length=0)
            #        sns.despine(ax = axis, left = True, bottom = True)
                    #ax.spines['left'].set_visible(False)
                    #ax.ticks['left'].set_visible(False)
                    #ax.despine() 
                    #ax.despine()
            #   else: 
            #       sns.despine(ax = axis, left = False, bottom = True)
    g.tight_layout()
    #g.se

    #dfc.save_fig(f'{kind}plot {x} stability {plot_dic['col']}')

    return g, plot_dic

def displot(df, x= 'count', hue = 'inh', col = 'quadrant', row = False, y = False, 
            kind = 'ecdf', stat = 'proportion', comp = False, log= (True, False), 
            weights = False, lw = 1,
            height = 1.4, aspect = 0.8, palette = 'torch', **kws):
    
    func_dic = dict(y = y ,stat = stat, kind = kind, row = row, col = col, complementary = comp, log_scale = log,
                    weights = weights, lw = lw,
                    height = height, aspect = aspect, palette = palette, )
    
    plot_dic = {key:value for key, value in func_dic.items() if value}
    if kind == 'ecdf':
        plot_dic = {key:value for key, value in plot_dic.items() if key != 'y'}
    #order_vars(x, hue, col, row)
    plot_dic.update(**order_vars(x, hue, col, row, **kws))
    
    for key, value in kws.items():
        if key == 'sharey':
            plot_dic.update({'facet_kws' : {'sharey': value}})
                
    g = sns.displot(data = df, **plot_dic)
    
    g = p_titles(g, col, row)
    #g.set_titles("{col_name}")
    #g.map(plt.axvline, x=l_count, ls="--", c=".5")
    #g.map(plt.axvline, x=h_count, ls="--", c=".5")
    return g



def pointplot_triphase(dfgi, xtra = '', x="tri_sec", hue = 'inside_injury',col="size", row = 'condition', palette = 'crest', **kws ):#y = 'count'

    #with sns.plotting_context("paper") and sns.axes_style("ticks"):
    #var_ls = [x, hue, col, row]
    #var_name_ls = ['x','hue','col', 'row']# 
    #order_name_dic = {'order':x,'col_order':col, 'row_order':row}#'hue_order':hue, 
    #order_name_ls = ['order','hue_order', 'col_order', 'row_order']
    #order_ls = [var_order(var) for var in var_ls if var]
    
    #plot_dic = {key:value for key,value in zip(var_name_ls, var_ls) if value}
    #plot_dic.update({key:var_order(value) for key, value in order_name_dic.items() if value})
    plot_dic = order_vars(x, hue, col, row)
    print(plot_dic)

    for var in dfgi['variable'].unique():
        g = sns.catplot(
            data=dfgi[dfgi.variable == var], y='value', 
            #x = x, order = var_order(x), 
            #hue = hue, hue_order = var_order(hue), 
            #col = col, col_order = var_order(col), 
            #row = row, row_order = var_order(row), 
            errorbar = 'se', kind="point", height=1.6, aspect=0.8, dodge = True, palette = palette, **plot_dic
        )
        g.set_xticklabels(rotation=45)
        if col:
            if row:
                g.set_titles("{col_name}\n:{row_name}")
            else: 
                 g.set_titles("{col_name}")
        elif row:
            g.set_titles("{row_name}")
        else:
             g.set_titles("")
        g.map(plt.axhline, y=1, ls="--", c=".5")
        #plt.subplots_adjust(top=0.90)
        g.fig.suptitle(var,fontsize=8) 
        #g.set_yticklabels(labels=[100000])#, **kwargs
        #g.ax.xaxis.set_major_locator(ticker.MultipleLocator(100000)
        #g.ax..xaxis.set_major_formatter(ticker.ScalarFormatter())
        fig_name = f'Pointplot per_ctrl {var} {hue} {col} {row}'
        fig_name += xtra
        dfc.save_fig(fig_name)
            #order = ['0-100', '100-300', '300-600']
            #hue_order = [True, False]
            #col_order = ['small', 'large']

#---------------------------------------------------------------------------
# lineplots of means with isovolumes as hue/color
#---------------------------------------------------------------------------

def plot_ax(dfgi, axis, iso_hues, var, 
            x = 'sec', hue = 'inh', labels = [], cmap = 'viridis', norm = [], nrow = 1, n_row = 1, ncol = 1, n_col = 1, 
               print_check = False, marker = 'black', count_thr = 1, lw = 1, midpoint = 37.5, cond = False,
               **params):
    
    #max = dfgi[var].max()
    #min = dfgi[var].min()
    #print('min', min, 'max', max)
    #iso_hues1 = [iso_hue for iso_hue in iso_hues if iso_hue in dfgi.columns]
    if midpoint:
        #print(params)
        iso_hues2 = iso_hues -37.5*np.ones(len(iso_hues))
        for val, diff in zip(iso_hues, iso_hues2):
            if diff == min(iso_hues2):    
                iso_hues.append(iso_hues.pop(iso_hues.index(val)))

    
    # PLOT INDIVIDUAL HUES AND TREATMENTS 
    #---------------------------------------------
    for cn,iso in enumerate(iso_hues):
        #for n,inh in enumerate([inh for inh in inhs if inh in dfg_i.inh.unique()]):
        #    dfgi=dfg_i[(dfg_i.inh==inh)]
        dfgi_iso=dfgi.loc[dfgi[hue]==iso,[x,var]].dropna()
        

        if dfgi.loc[dfgi[hue]==iso,'count'].mean(axis=0) > count_thr or 'count' in var :#== 'count_ALL'
            
            #print(dfgi_iso[x_var])
            if marker:
                if iso < 37 or iso > 39:
                    axis.plot(dfgi_iso[x],dfgi_iso[var], c=cmap(norm(iso)),linewidth= lw)
                else:
                    axis.plot(dfgi_iso[x],dfgi_iso[var], c=marker, linewidth= lw, zorder = 100)
            else:
                axis.plot(dfgi_iso[x],dfgi_iso[var], c=cmap(norm(iso)),linewidth= lw)
    
    # FORMAT SUBPLOT
    #---------------------------------------------
    if params['axhline']:
        #axs[n].axhline(0, color='0.7', linestyle='--')
        axis.axhline(0, color='0.7', linestyle='--')
        #plt.axvline(45, alpha = 0.4, color = 'grey')
    #if cond:
        #if cond == 'per_ctrl':
        #axs[n].axhline(0, color='0.7', linestyle='--')
            #axis.axhline(1, alpha = 0.5, color = 'grey', zorder = 100, linestyle='--')
            #ticks = []
            
            #if min < 0.25:
                #axis.axhline(0.25, alpha = 0.5, color = 'grey', zorder = 100, linestyle='--')
            #    ticks.append(0.25)
            #if min <= 0.5:
                #axis.axhline(0.5, alpha = 0.5, color = 'grey', zorder = 100, linestyle='--')
            #    ticks.append(0.5)   
            #ticks.append(1.0)  
            #if max >= 1.5:
            #    ticks.append(1.5)
                #axis.axhline(1.5, alpha = 0.5, color = 'grey', zorder = 100, linestyle='--')
            #if max >= 2.0:
            #    ticks.append(2.0)
                #axis.axhline(2.0, alpha = 0.5, color = 'grey', zorder = 100, linestyle='--')
            #if len(ticks) == 1:
            #    ticks.append(np.round(max, 0))
            #if len(ticks) > 1:
            #    axis.set_yticks(ticks, labels = ticks)
            #axis.set_yticks(ticks, labels = ticks)
            
                
    if n_row == 0:
        if ncol > 1:
            axis.set_title(f'{params["col_val"]}')
    
    if n_col == 0:
        if params['y_lab']:
            axis.set_ylabel(params['y_lab'])
        else:
            axis.set_ylabel(f'{var}')
    else:
        axis.set_ylabel('')
        axis.spines.left.set_visible(False)
        axis.yaxis.set_ticks_position('none')
        #axs[n].set_yticks([])
        
    if params['row']:
        if ncol - n_col == 1: 
            axis.annotate(f'{params["row_val"]}'.title(), xy=(0.85, 0.85), xycoords='axes fraction',horizontalalignment='right',fontsize=8 )
    #axis.set_title(inh)
    axis.spines.right.set_visible(False)
    axis.spines.top.set_visible(False)
    
    
    if x in ['rho_bin', 'rho']:
        axis.set_xlim(-92, 92)
        plt.xticks([-90, 0, 90])
        axis.axvline(- 45, alpha = 0.1, ls = ':', color = 'lightgrey', )#zorder = 0
        axis.axvline(45, alpha = 0.1, ls = ':', color = 'lightgrey', )#zorder = 0
        
    if x in ['sec', '10_sec']:
        axis.set_xlim(0, 600)
        #plt.xticks([0, 250, 500])
        axis.set_xticks([0, 250, 500])
        #axis.axvline(250, alpha = 0.5, ls = ':', color = 'grey', zorder = 100)#zorder = 0
        #axis.axvline(300, alpha = 0.1, ls = ':', color = 'lightgrey', zorder = 0)#zorder = 0
            
    if print_check: print('ROW:', nrow, n_row)
    if print_check: print('COL:', ncol, n_col)



    
        
    # SELECT VARIABLES AND TREATMENTS TO PLOT 
    #---------------------------------------------
def lines_cbar_mean(df1, x_var = 'rho_bin', col = 'inh', row = 'tri_sec', hue = 'iso_A', 
                    c_map= 'twilight', midpoint = 37.5,
                    #count_thr = 10, #marker = 'black',
                    print_check = False, roll_windows =  (5, 3, 5), #window_type = False,'blackman' #'triang'#'bartlett'
                    mean_vars = 'std',
                    #['elong', 'ca_corr', 'track_time', 'stab', 'dv_s','dvy_s', 'nba_d_5', ], #'dv_s','dvy_s', 'stab', 'nba_d_5', 'zs', 
                    #mean_vars = ['elong', 'ca_corr', 'track_time', 'stab', 'dv_s','sliding', 'nba_d_5', ], 
                    #hue_bin = 'aa', #'set',
                    #hue_nr = 20, #hue_range = 'linear'
                    xtra = "", 
                    save = False, 
                    height = 1.0, 
                    width = 1.5,
                    marker = 'black',
                    iso_max = False,
                    cond = False,
                    **kws):

                    
    if mean_vars == 'std':
        if cond == 'per_ctrl':
            mean_vars = ['elong', 'ca_corr', 'track_time', 'stab', 'sliding', 'nba_d_5', ]
        else:
            mean_vars = ['elong', 'ca_corr', 'track_time', 'stab', 'dvy_s', 'nba_d_5', ]
    
    plot_dic = order_vars(x = x_var, hue = hue, col = col, row = row)
    var_dic = {key:value for key,value in plot_dic.items() if 'order' not in key}
    id_vars = [var for var in var_dic.values()]

    mean_vars=[m_var for m_var in mean_vars if m_var in df1.columns.to_list()]

    all_vars = id_vars + mean_vars + ['pid']
    if 'path' not in all_vars:
        all_vars.append('path')
    if cond:
        if cond == 'per_ctrl':
            all_vars.append('inh')
        else:
            all_vars = all_vars + ['nrtracks', 'tracknr']
    df1 = df1[all_vars]

    #if hue_bin == 'set':
    #    df1, edges, labels, round = dfc.set_ranges(df1, hue, hue_nr, **kws)#iso_max = iso_max, 
    
    #    if print_check: print(f'Unique hue values: {sorted(df1.loc[:,hue].unique())}')

    roll_vars = [x_var, hue, x_var]
    #wins=(5,3)
    roll_seq = [(r_var,win) for r_var, win in zip(roll_vars, roll_windows)]
    if print_check: print(f'Roll seq: {roll_seq}')

    agg_dic = dict(agg_var = False, mean_vars = mean_vars, print_check = print_check, 
                roll = True, roll_seq = roll_seq, melt = False, cond = cond)#window_type = window_type
    if cond == 'per_ctrl':
        agg_dic.update(**{'plot_break_var' : 'inh'}) 
    agg_dic.update(**var_dic)
    print('agg_dic', agg_dic)    
    dfg, kws = dfc.agg_plot(df1, **agg_dic)

    plot_vars = [c_var for c_var in dfg.columns.to_list() if c_var not in id_vars + ['pid', 'path', 'inh', 'nrtracks', 'tracknr', 'count']]
    #plot_vars = dfg.loc[:, ~(id_vars)]

    p_d = sort_var_values(dfg, col, row)
    p_d.update(**var_dic)

    for dim, a_var in zip(['col', 'row'], [col, row]):
        if a_var:
            p_d.update({f'n{dim}':len(p_d[f'{a_var}_values'])})
            p_d.update({f'{dim}s':p_d[f'{a_var}_values']})
            p_d.update({f'{dim}':a_var})
            
        else:
            p_d.update({f'n{dim}':1})
            p_d.update({f'{dim}':False})
            p_d.update({f'{dim}s':False})
        if print_check: print('p_d:',p_d)
        
        #inhs=dfc.inh_order
    for p_var in plot_vars:
        print('Variable:', p_var)
        params_iso = std_params_plot()
        params = params_iso['std']
        if p_var in params_iso.keys(): 
                params.update(**{key: value for key, value in params_iso[p_var].items()})
        params.update(**p_d)
        params.update(**{'marker':marker})
        if iso_max:
            if print_check: print('iso_max:', iso_max)
            params.update(**{'iso_max': iso_max})
        if cond:
            params.update(**{'cond': cond})
            #if hue_bin == 'set':
            #    labels = [level for level in labels if level <= iso_max[1] and level >= iso_max[0]]
            
        

        #params.update(**p_d)
        if print_check: print('params updated with p_id:', params)
        
    # SELECT RELEVANT DATA IN DATAFRAME
        #---------------------------------------------
        if p_var == 'count':
            dfg_i = dfg.loc[(dfg[hue] > params['iso_max'][0]) & (dfg[hue] <= params['iso_max'][1]), id_vars + ['count']]
        else:
            dfg_i = dfg.loc[(dfg[hue] > params['iso_max'][0]) & (dfg[hue] <= params['iso_max'][1]), id_vars + ['count', p_var]]
        
        if print_check: 
            print('hue levels: ', dfg_i[hue].unique())#display(dfg_i)
            #print('labels: ', labels)
            #print('edges: ', labels)
        
        #n_inh=len(dfg_i.inh.unique())#len(dfc.inh_order)
        # SET FIGURE PARAMETERS & INITIATE FIGURE
        #---------------------------------------------


        #rows = plot_dic['row_order'] if plot_dic['row_order'] else dfg[row].unique().to_list()
        #cols = plot_dic['col_order'] if plot_dic['col_order'] else dfg[col].unique().to_list()
        #height = 1.0#1.4 #1.6
        #width = 1.5#2.4#3#2.2
        width1 = width +1 if params['ncol'] ==1 else width
        fig, axs = plt.subplots(params['nrow'], params['ncol'], figsize=(width1*params['ncol'],height*params['nrow']), 
                                squeeze = False,
                                constrained_layout = True, sharey = True, sharex = True)
        #fig, axs = plt.subplots(, n_inh, figsize=(1.2*n_inh,1.2), constrained_layout=True,sharey=True, sharex = True)
        
        #if hue_bin == 'set':
        #    iso_hues= labels
        #else:
        iso_hues= sorted(list(dfg_i.loc[dfg_i[p_var].notna(), hue].unique()))
        #if hue in cfg.varBinDic.keys():
        #    labels = np.round(iso_hues, cfg.varBinDic[hue]['round'])
        
        plt.ylim(dfg_i[p_var].min(),dfg_i[p_var].max())
        if not cond: 
            if params['logscale']: 
            #axs[n].set_yscale('log')
                plt.yscale('log')
            
        
            if params['ylim']:
                if (params['ylim'][0]):
                    plt.ylim(bottom = params['ylim'][0])
                if params['ylim'][1]:
                    plt.ylim(top = params['ylim'][1])
                
        
        
        #print(dfg_i, n_inh)
        
            # CALL COLORMAP FUNCTION
        #---------------------------------------------
        #cmap, norm, colormap = colormap_midpoint(c_map, iso_hues, round = 0)
        #cmap, norm, colormap = colormap_edges(c_map, iso_hues)
        #if hue_bin == 'set':
            #if low:
            #    cmap, norm, colormap = colormap_quartiles(c_map, iso_hues, low = low, mid = midpoint, high = high, **kws)
        
        #    if midpoint:
        #        cmap, norm, colormap = colormap_midpoint(c_map, iso_hues, midpoint = midpoint, **kws)
        #    else:
        #        cmap, norm, colormap = colormap_edges(c_map, iso_hues, **kws)
        
        
        #else:
        if midpoint:
            cmap, norm, colormap = colormap_midpoint(c_map, iso_hues, midpoint = midpoint, **kws)
        else:
            cmap, norm, colormap = colormap_edges(c_map, iso_hues, **kws)
                #cmap, norm, colormap = colormap_midpoint(c_map, iso_hues, round = 0)
        
        params.update({'cmap': cmap, 'norm': norm, 'colormap': colormap, 'midpoint': midpoint})
        
        if params['nrow'] > 1 and params['ncol'] > 1: 
            ndims = 2
        elif params['nrow'] > 1 or params['ncol'] > 1: 
            ndims = 1
        else: 
            ndims = 0
        params_iso['ndims'] = ndims
        
        if ndims == 2:
            for n_row,row in enumerate(params['rows']):
                if print_check:
                    print(f'Row: {n_row}, ', end = '')
                    print(f'Column:', end = '')
                params.update(**{'n_row':n_row,'row_val':row})
                for n_col,col in enumerate(params['cols']):
                    if print_check: print(f'{n_col}, ', end = '')
                    
                    params.update(**{'n_col':n_col,'col_val':col})                
                    dfi=dfg_i.loc[(dfg_i[params['col']] == col) & (dfg_i[params['row']] == row)].copy()
                    axis = axs[n_row, n_col]
                    
                    plot_ax(dfi, axis, iso_hues, p_var, **params)
                    #labels1 = [label for label in labels if label in dfi.columns]
                    #axs[n_row, n_col].stackplot(dfi[x], dfi[labels1].fillna(0).T, colors = cmap(norm(labels1)), **plot_p)
                    #axs[n_row, n_col].set_title(f'')
        
        elif ndims == 1:
            if params['row']:
                if print_check: print(f'Row: ', end = '')
                for n_row,row in enumerate(params['rows']):
                    if print_check: print(f'{n_row}, ', end = '')
                    params.update({'n_row':n_row,'row_val':row,'n_col':False,})
                    dfi = dfg_i.loc[(dfg_i[params['row']]==row)].copy()
                    axis = axs[n_row, 0]
                    plot_ax(dfi, axis, iso_hues, p_var, **params)
                    #labels1 = [label for label in labels if label in dfi.columns]
                    #axs[n_row, 0].stackplot(dfi[x], dfi[labels1].fillna(0).T, colors = cmap(norm(labels1)), **plot_p)
                    
            else:
                if print_check: print(f'Col: ', end = '')
                for n_col,col in enumerate(params['cols']):
                    if print_check: print(f'{n_col}, ', end = '')
                    params.update({'n_col':n_col,'col_val':col,'n_row':False})            
                    dfi=dfg_i.loc[(dfg_i[params['col']] == col)].copy()
                    axis = axs[0, n_col]
                    plot_ax(dfi, axis, iso_hues, p_var, **params)
                    #labels1 = [label for label in labels if label in dfi.columns]
                    #axs[0, n_col].stackplot(dfi[x], dfi[labels1].fillna(0).T, colors = cmap(norm(labels1)), **plot_p)
                    
                print('')    
        else:
            dfi=dfg_i.copy()
            axis = axs[0, 0]
            plot_ax(dfi, axis, iso_hues, p_var, **params)
        
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax= axis )#fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[n])#
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label(hue, rotation=270)
        
        if save:
            file_name = f'lines_cbar_{hue} {x_var} {p_var}'
            if params['col']:
                file_name += f' {params['col']}'
            if params['row']:
                file_name += f' {params['row']}'
            file_name += f' {colormap}'
            if xtra:
                file_name += f' {xtra}'
            if cond:
                file_name += f' {cond}'
                
            dfc.save_fig(file_name,transparent=False)
            plt.show()


def timemean_huevar3(dfg,hue='iso_vol',c_map='Spectral',x_var='sec', count_thr = 4, marker = 'red', save_ind_iso = False, xtra = False):
    
    # SELECT VARIABLES AND TREATMENTS TO PLOT 
    #---------------------------------------------
    vars=[var for var in list(dfg.columns) if var not in ['inh',hue, x_var]]
    inhs=dfc.inh_order
    

    # LOOP THROUGH VARIABLES 
    #---------------------------------------------
    for var in vars:
        print('Variable',var)
        
        # SET STANDARD PARAMETERS 
        #---------------------------------------------
        params_iso= std_params_plot()
        params = params_iso['std']
        #params.update(**{key: value for key, value in params_iso['std'].items()})
        if var in params_iso.keys(): 
            params.update(**{key: value for key, value in params_iso[var].items()})
         
        #print(params)
        
        # SELECT RELEVANT DATA IN DATAFRAME
        #---------------------------------------------
        if var == 'count_ALL':
            dfg_i=dfg.loc[(dfg[hue] > params['iso_max'][0]) & (dfg[hue] < params['iso_max'][1]), ['inh', hue, x_var, 'count_ALL']]
        else:
            dfg_i=dfg.loc[(dfg[hue] > params['iso_max'][0]) & (dfg[hue] < params['iso_max'][1]), ['inh', hue, x_var, 'count_ALL', var]]
        
        n_inh=len(dfg_i.inh.unique())#len(dfc.inh_order)
        
        
        # SET FIGURE PARAMETERS & INITIATE FIGURE
        #---------------------------------------------
        fig, axs = plt.subplots(1, n_inh, figsize=(1.2*n_inh,1.2), constrained_layout=True,sharey=True, sharex = True)
        
        iso_hues= sorted(list(dfg_i.loc[dfg_i[var].notna(), hue].unique()))
        #print(dfg_i, n_inh)
        
        # CALL COLORMAP FUNCTION
        #---------------------------------------------
        cmap, norm, colormap = colormap_midpoint(c_map, iso_hues, round = 0)
        #cmap, norm, colormap = colormap_edges(c_map, iso_hues)

        
        plt.ylim(dfg_i[var].min(),dfg_i[var].max())
        if params['ylim']:
            if (params['ylim'][0]):
                plt.ylim(bottom = params['ylim'][0])
            if params['ylim'][1]:
                plt.ylim(top = params['ylim'][1])
            #else:
             #   plt.ylim(dfg[var].min(),dfg[var].max())
        
        
        #if params['sup']:
        #    plt.suptitle(params['sup'])
        if params['logscale']: 
            #axs[n].set_yscale('log')
            plt.yscale('log')
            
        #plt.xticks(rotation=90)
        #cbar=plt.colorbar(cmap_d)#plt.colorbar(cmap_d)#, ax=axs[n]#cm.ScalarMappable(norm=norm,cmap=cmap))#
        #cbar.ax.get_yaxis().labelpad = 15
        #cbar.ax.set_ylabel('Outer radius of isovolume ($\mu$m)', rotation=270)
        iso_hues2 = iso_hues -37.5*np.ones(len(iso_hues))
        for val, diff in zip(iso_hues, iso_hues2):
            if diff == min(iso_hues2):    
                iso_hues.append(iso_hues.pop(iso_hues.index(val)))
        
        # PLOT INDIVIDUAL HUES AND TREATMENTS 
        #---------------------------------------------
        for cn,iso in enumerate(iso_hues):
            for n,inh in enumerate([inh for inh in inhs if inh in dfg_i.inh.unique()]):
                dfgi=dfg_i[(dfg_i.inh==inh)]
                dfgi_iso=dfgi.loc[dfgi[hue]==iso,[x_var,var]].dropna()

                if dfgi.loc[dfgi[hue]==iso,'count_ALL'].mean(axis=0) > count_thr or 'count' in var :#== 'count_ALL'
                    
                    #print(dfgi_iso[x_var])
                    if iso < 37 or iso > 39:
                        axs[n].plot(dfgi_iso[x_var],dfgi_iso[var], c=cmap(norm(iso)),linewidth=1)
                    else:
                        axs[n].plot(dfgi_iso[x_var],dfgi_iso[var], c=marker, linewidth=1, zorder = 100)#ls = ':'
                    #if iso > 37 and iso < 39:
                    #    axs[n].plot(dfgi_iso[x_var],dfgi_iso[var],c='grey',linewidth=4,alpha=0.4)
                 
                # FORMAT SUBPLOT
                #---------------------------------------------
                if params['axhline']:
                    #axs[n].axhline(0, color='0.7', linestyle='--')
                    axs[n].axhline(0, color='0.7', linestyle='--')
                    #plt.axvline(45, alpha = 0.4, color = 'grey')
                    
                if n == 0:
                    if params['y_lab']:
                        axs[n].set_ylabel(params['y_lab'])
                    else:
                        axs[n].set_ylabel(f'{var}')
                else:
                    axs[n].set_ylabel('')
                    axs[n].spines.left.set_visible(False)
                    #axs[n].set_yticks([])
                    
                    
                axs[n].set_title(inh)
                axs[n].spines.right.set_visible(False)
                axs[n].spines.top.set_visible(False)
                
                if x_var in ['rho_bin', 'rho']:
                    axs[n].set_xlim(-92, 92)
                    plt.xticks([-90, 0, 90])
                    axs[n].axvline(- 45, alpha = 0.1, ls = ':', color = 'lightgrey', )#zorder = 0
                    axs[n].axvline(45, alpha = 0.1, ls = ':', color = 'lightgrey', )#zorder = 0
            
            # SAVES PLOTS FOR INDIVIDUAL ISO LEVELS IF save_ind_iso == TRUE
            #---------------------------------------------
            if save_ind_iso:
                dfc.save_fig(f'lineplot iso {iso} var_{var} cmap_{colormap}',transparent=False)
               
            
        plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=axs[n])
        #plt.suptitle(inh)
        if not save_ind_iso:
            file_name = f'lineplot iso {x_var} var_{var} cmap_{colormap}'
            if xtra:
                file_name += xtra
                
            dfc.save_fig(file_name,transparent=False)
        plt.show()

#---------------------------------------------------------------------------
# Stackplot with colormap, use with df_stackplot in data_functions
#---------------------------------------------------------------------------
def sort_var_values(dfi, *args):
    #dim_vars =  ['col_var', 'row_var']
    kws = {}
    for dim in args:
        
        if dim:
            
            if dim == 'inh':
                vals = dfc.inh_order
                values = [val for val in vals if val in dfi[dim].unique().tolist()]
                kws.update({dim +'_values' : values}) 
                #params.update({dim +'_values' : dfc.inh_order}) 
            
            elif dim in cfg.var_order.keys():
                vals = cfg.var_order[dim] 
                values = [val for val in vals if val in dfi[dim].unique().tolist()]
                kws.update({dim +'_values' : values}) 

            else:
                #print(df[params[dim]].unique())
                #print(sorted(df[params[dim]].unique().tolist(),reverse=False))
                kws.update({dim +'_values' : sorted(dfi[dim].unique().tolist(),reverse=False)}) 
        
        #else: 
        #    kws[dim + '_values'] = False
            
    return kws


def stackplot(df1, 
              edges = [],
              labels = [],
              stackVar = 'ca_corr', 
              x = 'sec', 
              cmap = 'coolwarm', 
              #midpoint = 50,
              midpoint = False,
              xtra = "",
              baseline = 'zero',
              order = 'low_first',
              col = 'inh',
              row = False,
              low = False, 
              high = False,
              ylog = False,
              print_check = False,
              title = False,
              
              **kws
              ):
    
    if title: title = xtra#.copy()
    
    if order != 'low_first':
        labels = sorted(labels, reverse = True)
    
    if baseline != 'zero':
        xtra += f'{baseline}'
    if row:
        xtra += f' {row}'
    if col != 'inh':
        xtra += f' {col}'
    
    id_kws = [value for value in [col, row] if value]
    id_cols = [x]
    id_cols += id_kws
    
    pd = sort_var_values(df1, col, row)
    
    if print_check: print(pd)
    for dim, var in zip(['col', 'row'], [col, row]):
        if var:
            pd.update({f'n{dim}':len(pd[f'{var}_values'])})
            pd.update({f'{dim}s':pd[f'{var}_values']})
            pd.update({f'{dim}':var})
            
        else:
            pd.update({f'n{dim}':1})
            pd.update({f'{dim}':False})
            pd.update({f'{dim}s':False})
    if print_check: print(pd)
    #if stackVar == 'iso_vol':
    #    df1 = df1.astype({'iso_vol': str})
    piv = df1.pivot_table(index = id_cols, columns = stackVar, values = 'count').reset_index()
    
    
    n_inh = len(dfc.inh_order)
    
    if low:
        cmap, norm, colormap = colormap_quartiles(cmap, edges, low = low, mid = midpoint, high = high, **kws)
    
    elif midpoint:
        cmap, norm, colormap = colormap_midpoint(cmap, edges, midpoint = midpoint, **kws)
    else:
        cmap, norm, colormap = colormap_edges(cmap, edges, **kws)
    
    plot_p = dict(lw = 0, baseline = baseline, )
    pd.update({'plot_p': plot_p, 'cmap' : cmap, 'norm' : norm, 'labels' : labels, 'x': x, 'print_check': print_check, })
    
    
    height = 1.2#1.4 #1.6
    width = 1.6#2.4#3#2.2
    fig, axs = plt.subplots(pd['nrow'], pd['ncol'], figsize=(width*pd['ncol'],height*pd['nrow']), 
                            squeeze = False,
                            constrained_layout = True, sharey = True, sharex = True)
    
    if pd['nrow'] > 1 and pd['ncol'] > 1: 
        ndims = 2
    elif pd['nrow'] > 1 or pd['ncol'] > 1: 
        ndims = 1
    else: 
        ndims = 0
    pd['ndims'] = ndims
    
    if ndims == 2:
        for n_row,row in enumerate(pd['rows']):
            if print_check:
                print(f'Row: {n_row}, ', end = '')
                print(f'Column:', end = '')
            pd.update(**{'n_row':n_row,'row_val':row})
            for n_col,col in enumerate(pd['cols']):
                if print_check: print(f'{n_col}, ', end = '')
                
                pd.update(**{'n_col':n_col,'col_val':col})                
                dfi=piv.loc[(piv[pd['col']] == col) & (piv[pd['row']] == row)].copy()
                df_all = df1[(df1[pd['col']] == col) & (df1[pd['row']] == row)].copy()
                
                max = df_all.groupby(x)['count'].sum().max()
                idx_max = df_all.groupby(x)['count'].sum().idxmax()
                
                axis = axs[n_row, n_col]
                
                plot_stack(dfi, axis, max = max, idx_max = idx_max, **pd)
                #labels1 = [label for label in labels if label in dfi.columns]
                #axs[n_row, n_col].stackplot(dfi[x], dfi[labels1].fillna(0).T, colors = cmap(norm(labels1)), **plot_p)
                #axs[n_row, n_col].set_title(f'')
    
    elif ndims == 1:
        if pd['row']:
            if print_check: print(f'Row: ', end = '')
            for n_row,row in enumerate(pd['rows']):
                if print_check: print(f'{n_row}, ', end = '')
                pd.update({'n_row':n_row,'row_val':row,'n_col':False,})
                dfi = piv.loc[(piv[pd['row']]==row)].copy()
                df_all = df1[(df1[pd['row']] == row)].copy()
                
                max = df_all.groupby(x)['count'].sum().max()
                idx_max = df_all.groupby(x)['count'].sum().idxmax()
                
                axis = axs[n_row, 0]
                plot_stack(dfi, axis, max = max, idx_max = idx_max, **pd)
                #labels1 = [label for label in labels if label in dfi.columns]
                #axs[n_row, 0].stackplot(dfi[x], dfi[labels1].fillna(0).T, colors = cmap(norm(labels1)), **plot_p)
                
        else:
            if print_check: print(f'Col: ', end = '')
            for n_col,col in enumerate(pd['cols']):
                if print_check: print(f'{n_col}, ', end = '')
                pd.update({'n_col':n_col,'col_val':col,'n_row':False})            
                dfi=piv.loc[(piv[pd['col']] == col)].copy()
                df_all = df1[(df1[pd['col']] == col)].copy()
                
                max = df_all.groupby(x)['count'].sum().max()
                idx_max = df_all.groupby(x)['count'].sum().idxmax()
                
                axis = axs[0, n_col]
                plot_stack(dfi, axis, max = max, idx_max = idx_max, **pd)
                #labels1 = [label for label in labels if label in dfi.columns]
                #axs[0, n_col].stackplot(dfi[x], dfi[labels1].fillna(0).T, colors = cmap(norm(labels1)), **plot_p)
                
            print('')    
    else:
        dfi=piv.copy()
        axs[0, 0].stackplot(dfi[x], dfi[labels].fillna(0).T, colors = cmap(norm(labels)), **plot_p)
        
    #for n,inh in enumerate(dfc.inh_order):
    #    dfi = piv.loc[piv.inh == inh]
    #    labels1 = [label for label in labels if label in dfi.columns]
        #print(labels1)
    #    axs[n].stackplot(dfi[x], dfi[labels1].fillna(0).T, colors = cmap(norm(labels1)), lw=0, baseline = baseline)#, labels = bin_labels,
        #axs[n].set_ylabel(f'{var}')
        
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax= axis )#fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[n])#
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label(stackVar, rotation=270)
    
    for ax in axs.flatten():
        if ylog:
            print('ylog')
            #plt.yscale('log')
            ax.set_yscale('log')
            
    if title:
        #plt.title(title)
        plt.subplots_adjust(top=0.90)
        fig.suptitle(title,fontsize=12) 

    file_name = f'Areaplot {stackVar} x_{x} cmap_{colormap} {xtra}'
    #        file_name += xtra
        
    dfc.save_fig(file_name,transparent=False)
    
    return piv, pd

def plot_stack(dfi, axis, max = False, idx_max = False, x = 'sec', 
               labels = [], cmap = 'viridis', norm = [], nrow = 1, n_row = 1, ncol = 1, n_col = 1, 
               print_check = False,
               max_line = True,
               **kws):
    labels1 = [label for label in labels if label in dfi.columns]
    plot_params = kws['plot_p']
    
    axis.stackplot(dfi[x], dfi[labels1].fillna(0).T, colors = cmap(norm(labels1)), **plot_params)
    if print_check: print('col_val:', kws["col_val"])
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    
    #axis.spines['bottom'].set_visible(False)
    if n_row == 0:
        axis.set_title(f'{kws["col_val"]}')
    #else:
        
    if n_col == 0:
        axis.set_ylabel('Platelets')
    else:
        axis.spines['left'].set_visible(False)
        axis.yaxis.set_ticks_position('none')
    if kws['row']:
        if ncol - n_col == 1: 
            axis.annotate(f'{kws["row_val"]}', xy=(0.85, 0.85), xycoords='axes fraction',horizontalalignment='right',fontsize=8 )#.title()
    
    if max:
        if plot_params['baseline'] == 'zero':
            axis.axhline(max, ls = '--')  
        else: 
            axis.axhline(max/2, ls = '--')
            axis.axhline(-max/2, ls = '--')
            axis.axhline(0, ls = ':', color = 'black')
    
    if idx_max:
        axis.axvline(idx_max, ls = '--')
    
    
    if print_check: print('ROW:', nrow, n_row)
    if print_check: print('COL:', ncol, n_col)
    #if
    
#---------------------------------------------------------------------------
# Mapping platelet positions
#---------------------------------------------------------------------------
def plt_map(df_obj,col_var,x_var,vmin,vmax): #Map of platelets at different time points coloured with name variable
    #plt.rcParams['image.cmap'] = 'viridis'#'coolwarm'#"turbo"    #plt.rcParams['image.cmap'] = 'jet_r'
    sns.set_style("white")
    #Set boundaries of plots #params={'col':'path','row':'c','hue':'c',}
    lims=['x_s', 'ys', 'zs']
    limsv=dict(x_s=(-100,100),ys=(-120,80),zs=(0,100))
    #limsv={}
    #for l in lims:
    #    limsv[l]=df_obj[l].min(), df_obj[l].max()   
    #Pick frames for visualization
    frames=[10,20,30,50,90,180]#pd.unique(df_obj.frame)[::20]+10
    ncols=3
    nrows=len(frames)
    #Set figure size  
    plt.figure(figsize=(ncols*4,nrows*4))
    #Choose plotting dimensions in graphs
    cols=[('x_s', 'ys','zs'), ('x_s', 'zs','ys'), ('ys', 'zs','x_s')]
    ### Set color variable name='cld'#name='stab'#name='c'#name='depth' #colorv=[1,2,4,8] #name='c2_max'
    #vmin=0 #vmax=30#vmax=10 #vmax=400
    for r, f in enumerate(frames):
        #sel_f=df_obj[df_obj.frame==f]
        sel_f=df_obj[df_obj.frame.isin(range(f-2,f+2))]
        for c, xy in enumerate(cols):
            sel_f.sort_values(by=xy[2])
            ax=plt.subplot2grid((nrows, ncols), (r, c))
            ax.scatter(sel_f[xy[0]], sel_f[xy[1]], alpha=0.7, c=sel_f[col_var],vmin=vmin,vmax=vmax,s=30, linewidth=0.1,cmap='coolwarm')#'bwr' 'coolwarm'TOG BORT , vmin=vmin, vmax=vmax,)
            ax.set_title('Time (sec): '+ str(np.round(sel_f[x_var].mean())),fontsize=12)#sel_f.time.mean()
            ax.set_ylim(limsv[xy[1]])
            ax.set_xlim(limsv[xy[0]])
            plt.xticks([])#-37.5,37.5
            plt.yticks([])#-37.5,37.5
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelsize=12)
            if xy[1]=='ys':
                circle=plt.Circle((0, 0), 37.5, alpha=0.4,fc='grey',edgecolor='black')#edgecolor='black',linewidth=7,fill='black'
                ax.add_patch(circle)
            else:
                ax.hlines(y=0, xmin=-38, xmax=38, linewidth=8, color='grey',alpha=0.8)
            sns.despine(top=True, right=True, left=True, bottom=True)
            #ax.ticklabel_format()
            #ax.set_axis_bgcolor('black')
    plt.tight_layout()
    dfc.save_fig(col_var,'plt_map')  

def plt_map2(df_obj,col_var,x_var,vmin,vmax): #Map of platelets at different time points coloured with name variable
    #plt.rcParams['image.cmap'] = 'viridis'#'coolwarm'#"turbo"    #plt.rcParams['image.cmap'] = 'jet_r'
    sns.set_style("white")
    #Set boundaries of plots #params={'col':'path','row':'c','hue':'c',}
    lims=['x_s', 'ys', 'zs']
    limsv=dict(x_s=(-100,100),ys=(-120,80),zs=(0,100))
    #limsv={}
    #for l in lims:
     #   limsv[l]=df_obj[l].min()-1, df_obj[l].max()+1   
    #Pick frames for visualization
    frames=[10,20,30,50,90,180]#pd.unique(df_obj.frame)[::20]+10
    ncols=len(lims)
    nrows=len(frames)
    size=ncols*nrows
    #Set figure size  
    #Choose plotting dimensions in graphs
    cols=[('x_s', 'ys','zs'), ('x_s', 'zs','ys'), ('ys', 'zs','x_s')]
    ### Set color variable name='cld'#name='stab'#name='c'#name='depth' #colorv=[1,2,4,8] #name='c2_max'
    #vmin=0 #vmax=30#vmax=10 #vmax=400
    for c, xy in enumerate(cols,0):
        fig=plt.figure(figsize=(6,nrows*6))#figsize=(4,30)
        #fig.set_title(inhibitor)
        gs=GridSpec(nrows,1)
        plot_nr=0
        for r, f in enumerate(frames):
            sel_f=df_obj[df_obj.frame.isin(range(f-2,f+2))]
            #sel_f=df_obj[df_obj.frame==f]
            sel_f.sort_values(by=xy[2])
            ax=fig.add_subplot(gs[plot_nr])
            ax.scatter(sel_f[xy[0]], sel_f[xy[1]], alpha=0.7, c=sel_f[col_var],vmin=vmin,vmax=vmax,s=60, linewidth=0.1,cmap='coolwarm' )#TOG BORT , vmin=vmin, vmax=vmax,)
            ax.set_title('Time (sec): '+ str(np.round(sel_f[x_var].mean())),fontsize=12)#sel_f.time.mean()
            ax.set_ylim(limsv[xy[1]])
            ax.set_xlim(limsv[xy[0]])
            plt.xticks([])#-37.5,37.5
            plt.yticks([])#-37.5,37.5
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelsize=12)
            if xy[1]=='ys':
                circle=plt.Circle((0, 0), 37.5, alpha=0.4,fc='grey',edgecolor='black')#edgecolor='black',linewidth=7,fill='black'
                ax.add_patch(circle)
            else:
                ax.hlines(y=0, xmin=-38, xmax=38, linewidth=8, color='grey',alpha=0.8)
            sns.despine(top=True, right=True, left=True, bottom=True)
            plot_nr+=1
                #ax.ticklabel_format()
                #ax.set_axis_bgcolor('black')
        gs.tight_layout(fig)
    #plt.tight_layout()
    dfc.save_fig(col_var,'plt_map')  

#---------------------------------------------------------------------------
# Mapping platelet trajectories
#---------------------------------------------------------------------------



#OBS! MÅSTE FIXA VARIABELN time SÅ ATT DEN FUNGERAR I FUNKTIONEN INNAN DENNA FUNKAR ATT KÖRA!!!"
def t_traj_mov(df,c_var='ca_corr',**xtra_params):#vmin,vmax
    df=df.sort_values(by=['pid'])
    sns.set_style("white")#sns.set_style("dark")# sns.set_style("white")
    plt.rcParams['image.cmap'] = 'coolwarm'
    plt.rcParams.update({'font.size': 22})
    params={'cols':3,'nrows':3,'hue':cfg.mov_class_order1,'vmin':0,'vmax':70,'time_bins':'phase','c_var':'tracknr'}
    params.update(xtra_params)
    for c, inhibitor in enumerate(pd.unique(df.inh),0):
        fig=plt.figure(figsize=(12,14))#figsize=(4,30)
        #fig.set_title(inhibitor)
        gs=GridSpec(params['nrows'],params['cols'])
        plot_nr=0
        for time in df['time_bin'].unique():
            for pop in params['hue']:
                plt_pop=df[(df.mov_phase==pop)&(df['time_bin']==time)&(df.inh==inhibitor)].copy()
                #pop_part=plt_pop.particle.unique()
                ax=fig.add_subplot(gs[plot_nr])
                
                plt_pop.sort_values(params['c_var'],ascending=False)
                plt.scatter(x=plt_pop.x_s, y=plt_pop.ys , c=plt_pop[c_var], s=10, alpha=0.7, cmap='viridis',vmax=params['vmax'], vmin=params['vmin'], linewidth=0)#'coolwarm'
                #plt.scatter(x=0, y=0, s=375,c='none', alpha=0.5, linewidth=40,edgecolor='black')#c='black',     
                circle=plt.Circle((0, 0), 37.5, alpha=0.4,fc='grey',edgecolor='black')#edgecolor='black',linewidth=7,fill='black'
                ax.add_patch(circle)
                plt.axis([-100, 100, -100, 100])
                plt.xlim(-100,100)  
                plt.ylim(-125,100)
                plt.xticks([])#-37.5,37.5
                plt.yticks([])#-37.5,37.5
                if pop==params['hue'][0]:
                    ax.set_ylabel(f'Time range: {np.round(time.left,-1)}-{np.round(time.right,-1)} s',fontsize=14)
                if plot_nr<3:
                    ax.set_title(pop)
                #if pop==plot[2]:
                #    plt.colorbar()
                plot_nr+=1
        sns.despine(top=True, right=True, left=True, bottom=True)
        fig.suptitle(inhibitor, fontsize=16)
        treatment=cfg.longtoshort_dic[inhibitor]
        dfc.save_fig(f'traj_map_{treatment}_','mov_class')
        plt.show()


#---------------------------------------------------------------------------
# Outlier detection plots
#---------------------------------------------------------------------------
def outliers_nrtracks(pc):
    test_var='nrtracks'
    pc_test=pc
    dfg=pc_test.groupby(['inh','path']).mean()[[test_var]].reset_index()
    outliers=[]
    for inh in dfg.inh.unique():
        dfg_inh=dfg[(dfg.inh==inh)].copy()
        outliers.append(dfg_inh[(np.abs(stats.zscore(dfg_inh[test_var])) > 1)]['path'])#['inh_id'])#Changed from 2!!!
    df_outliers=pd.concat(outliers,axis=0)
    dfg['outlier']=dfg.path.isin(df_outliers)
    dfg['value']=dfg[test_var]
    g=sns.catplot(data=dfg,y='value',x='inh',hue='outlier',height=5,aspect=3,kind='swarm',legend=False)
    g.set_xticklabels(rotation=45)
    dfc.save_fig('all variables','outliers nrtracks')        
    plt.show()
    
    return dfg[(dfg.outlier==True)],dfg[(dfg.outlier==False)]

def outliers_count(df):
    hue_order=['True','False','Both']
    df['tracked']=df.nrtracks>1
    dfg_count=df.groupby(['inh','exp_id','tracked']).count()[['pid']].reset_index()
    dfg_count1=df.groupby(['inh','exp_id']).count()[['pid']].reset_index().set_index(
        ['inh','pid']).sort_index().reset_index()
    dfg_count1['tracked']='Both'
    dfg=pd.concat([dfg_count,dfg_count1],axis=0)
    outliers=[]
    for inh in dfg.inh.unique():
        dfg_inh=dfg[(dfg.inh==inh)&(dfg.tracked=='True')].copy()
        outliers.append(dfg_inh[(np.abs(stats.zscore(dfg_inh.pid)) > 2)]['exp_id'])
    df_outliers=pd.concat(outliers,axis=0)
    dfg['outlier']=dfg.exp_id.isin(df_outliers)
    dfg['value']=dfg['pid']
    g=sns.catplot(data=dfg,y="value",x='tracked',col='inh',hue='outlier',col_wrap=3,height=4,kind='swarm')
    dfc.save_fig('all variables','outliers count')      
    plt.show()
    return dfg[(dfg.outlier==True)&(dfg.tracked=='True')]

def outliers_count1(df):
    dfg=df.groupby(['inh','path']).count()[['pid']].reset_index()#.set_index(#,'mouse','inj',
    outliers=[]
    for inh in dfg.inh.unique():
        dfg_inh=dfg[(dfg.inh==inh)].copy()
        outliers.append(dfg_inh[(np.abs(stats.zscore(dfg_inh.pid)) > 2)]['path'])
    df_outliers=pd.concat(outliers,axis=0)
    dfg['outlier']=dfg.path.isin(df_outliers)
    dfg['value']=dfg['pid']
    g=sns.catplot(data=dfg,y="value",x='inh',hue='outlier',height=5,aspect=3,kind='swarm')
    g.set_xticklabels(rotation=45)
    dfc.save_fig('all variables','outliers count') 
    plt.show()
    return dfg[(dfg.outlier==True)]

def outliers_fluo(pc):
    pc_test=pc[(pc.tracked==True)]#(pc.inside_injury==True)&(pc.height=='bottom')&
    dfg_fluo=pc_test.groupby(['inh','path']).mean()[['c0_mean','c0_max','c2_mean','c1_mean','c1_max']].reset_index()#.set_index(
    id_cols=['inh','path',]#'mouse','inj',
    melt_vars=['c0_mean','c1_mean','c2_mean',]#'c0_max','c1_max'
    #melt_vars=['c0_mean','c0_max','c2_mean','c1_mean','c1_max']
    dfg_fluo_long=dfg_fluo.melt(id_vars=id_cols,value_vars=melt_vars,var_name='Measure',value_name='Fluorescence')
    outliers=[]
    for inh in dfg_fluo_long.inh.unique():
            for measure in dfg_fluo_long.Measure.unique():
                dfg_=dfg_fluo_long[(dfg_fluo_long.inh==inh)&(dfg_fluo_long.Measure==measure)].copy()
                dfg_['outlier']=(np.abs(stats.zscore(dfg_.Fluorescence)) > 2)
                outliers.append(dfg_)
    try:
        df_outliers=pd.concat(outliers,axis=0)
    except ValueError: 
        print(pc_test)
    g=sns.catplot(data=df_outliers,y="Fluorescence",x='inh',hue='outlier',row='Measure',height=5,aspect=3,kind='swarm',sharey=False)
    g.set_xticklabels(rotation=45)
    plt.show()
    df_outliers['value']=df_outliers['Fluorescence']
    dfc.save_fig('all variables','outliers fluo')      
    return df_outliers[df_outliers.outlier==True]

#---------------------------------------------------------------------------
# HEATMAP FOR REGIO MEASURE
#---------------------------------------------------------------------------

# TEST FUNCTION WHERE WE EXPRESS RESULTS AS PERCENT OF CONTROL  (INHIBITED - CONTROL)/CONTROL   INSTEAD OF ABSOLUTE VALUES  

def heatmap_regio(df1,  
                      center = 25,
                      vmin = 10,
                      vmax = 50,
                      c_map = cmr.chroma,#cmr.chroma#cmr.torch#'plasma'#'RdBu_r'
                      var = 'ca_corr',
                      filter = False,#'gauss_s1'
                      #mean_vars = ['ca_corr', 'nba_d_5', 'stab', 'dv', 'dvy', 'track_time'],
                      area_var = 'regio',
                      sec_bin = 20, 
                      row_thr = 150,
                      pixel_thr = 5,
                      xtra = False, 
                      agg_type = 'mean_tot',
                      norm = False, 
                      orientation = 'vertical',
                      kind = 'per_ctrl',
                      rolling = True,
                      
                      
                      ):
    import copy
    import itertools
    regio_order = [r[0] + r[1] for r in itertools.product(cfg.quadrant3_order, cfg.region_order[::-1])]
    #if orientation == 'vertical':
    #    x_var = 'regio'
    #    y_var = f'{sec_bin}_sec'
    
    #else:
    #    y_var = 'regio'
    #    x_var = f'{sec_bin}_sec'
    #time_var = f'{sec_bin}_sec'
    
    
    #if time_var not in df1.columns:
    #    df1 = dfc.bin_time_var(df1, bin_size = sec_bin)
    
    
    # CALCULATE COUNTS
    #-----------------------------------------------------------------
    #counts_path = df1.groupby(['inh', 'path', area_var, time_var, 'frame', ], 
    #                          observed = True)['pid'].count().fillna(0).groupby(level = [0,1,2,3], observed = True).mean().rename('count')
    #counts_mean = counts_path.groupby(level = [0,2,3], observed = True).mean()
    
    counts_mean = df1.groupby(['inh', 'path', area_var, 'sec', ], observed = True)['pid'].count().fillna(0).groupby(
        level = [0,2,3], observed = True).mean().rename('count')
    
    #-----------------------------------------------------------------
    #if var != 'count':
    #    
    #else:
        
    # CALCULATE MEANS
    #-----------------------------------------------------------------
    if var != 'count':
        columns_ = [var] + ['count']
        if agg_type == 'mean_tot':
            #means = df1.groupby(['inh', x_var, y_var, 'frame'], observed = True)[mean_vars].agg('mean').groupby(level = [0,1,2], observed = True).mean()
            #means = df1.groupby(['inh', area_var, time_var, 'frame'], observed = True)[[var]].agg('mean').groupby(level = [0,1,2], observed = True).mean()
            means = df1.groupby(['inh', area_var, 'sec'], observed = True)[[var]].agg('mean')#.groupby(level = [0,1,2], observed = True).mean()
            
        else: 
            #means = df1.groupby(['inh', 'path',area_var, time_var, 'frame'], observed = True)[[var]].agg('mean').groupby(level = [0,1,2,3], observed = True).rolling(window = 3, min_periods = 1).mean().droplevel([0,1])
            means = df1.groupby(['inh', 'path',area_var,  'sec'], observed = True)[[var]].agg('mean')#.groupby(level = [0,1,2,3], observed = True)#.rolling(window = 3, min_periods = 1).mean().droplevel([0,1])
            
            means = means.groupby(level = [0,2,3], observed = True).mean()
            #means = df1.groupby(['inh', 'path',x_var, y_var, 'frame'], observed = True)[mean_vars].agg('mean').groupby(
            #    level = [0,1,2,3], observed = True).mean().groupby(level = [0,2,3], observed = True).mean()
    #dfg = means.copy()#.reset_index()
        
        means['count'] = counts_mean
        
    else:
        means = counts_mean.to_frame()
        columns_ = ['count']
    
    means = means.groupby(level = [0,1], observed = True).rolling(window = 3, min_periods = 1).mean().droplevel([0,1]).reset_index()
        
    time_var = f'{sec_bin}_sec'
    
    
    if time_var not in means.columns:
        means = dfc.bin_time_var(means, bin_size = sec_bin)
        
    means = means.groupby(['inh', area_var, time_var], observed = True).mean()
    #if rolling == True:
     #       means = means.groupby(level = [0,1], observed = True).rolling(window = 3, min_periods = 1).mean().droplevel([0,1])
    #-----------------------------------------------------------------
        
    
    # FIX COLORMAP AND SET BAD VALUES TO BLACK
    #-----------------------------------------------------------------
    colormap = c_map if isinstance(c_map, str) else c_map.name
    c_map = copy.copy(plt.get_cmap(c_map))
    c_map.set_bad(color= 'black' if c_map.name in ['cmc.bilbao', 'cmc.devon_r', 'gist_heat_r', 'ocean_r', 'cmc.oslo_r'] else 'black')
    #params['cmap']=cmap
    print(f'Colormap chosen: {colormap}')
    #-----------------------------------------------------------------


    # Censor regio's with less than pixel_thr observations and areas with on average less than row_thr observations per experiment
    #-----------------------------------------------------------------

    means.where(means['count'] > pixel_thr, np.nan, inplace = True)
    
    #display(means.groupby(level = [0,1], observed = True)['count'].sum().head())
    
    count_mask = means.groupby(level = [0,1], observed = True)['count'].sum() > row_thr
    
    means.where(count_mask, np.nan, inplace = True)
    #-----------------------------------------------------------------
    
    
    #-----------------------------------------------------------------
    if kind == 'per_ctrl':
        label = f'{var}, percent of vehicle'
        means.loc[:,'condition'] = 'treatment'
        means = means.reset_index()
        means.loc[means['inh'] == dfc.inh_order[0],'condition'] = 'vehicle'
        #display(means.head())
        means = means.set_index([time_var, area_var])
        s_inh = means[means.condition == 'treatment']['inh']
        #dfgi.loc[:,'pop'] = n
        #columns_ = mean_vars + ['count']
        dfg = means[means.condition == 'treatment'][columns_].div(means[means.condition == 'vehicle'][columns_])
        dfg = dfg.mul(100)#.reset_index()#.drop(columns = 'condition')
        #display(s_inh)
        #display(dfg.head())
        dfg['inh'] = s_inh
        axes = dfc.inh_order[1:]
        
    #-----------------------------------------------------------------
    else:
        axes = dfc.inh_order
        dfg = means[columns_].reset_index().set_index([time_var, area_var])
        label = var
        
        
        
    
    #display(dfg.head())

    # Melt dataframe 
    dfg = dfg.melt(value_vars = columns_, ignore_index = False, var_name = 'var', value_name = 'value', id_vars = ['inh'])#.reset_index()
    #display(dfg.head())
    tick_dic = {
                '20_sec': [0, 100, 200, 300, 400, 500, 600], #[0, 60, 120, 240, 360, 480],
            
                '30_sec': [0, 60, 120, 240, 360, 480], 
                '25_sec': [0, 100, 200, 300, 400, 500, 600],
                '10_fsec': [0, 100, 200, 300, 400, 500, 600],
                }

    dfgi = dfg[(dfg['var'] == var) ].reset_index()


    if orientation == 'vertical':
        fig, axs = plt.subplots(1,len(axes), figsize = (10,3), squeeze = False, constrained_layout = True)
        #col = 0
    else:
        #fig, axs = plt.subplots(len(axes),1, figsize = (5,4*len(axes)), squeeze = False, constrained_layout = True)
        fig, axs = plt.subplots(1, len(axes), figsize = (3.5*len(axes), 4), squeeze = False, constrained_layout = True)
        
   
    #cbar_ax = fig.add_axes([.91, .3, .02, .4])
    
    #fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8,4))

    for num, inh in enumerate(axes):
        print(num, inh)
    #    ax = axs[0, col]
        
        #display(dfgi.head())
        if orientation == 'vertical':
            ax = axs[0, num]
            dfg1 = dfgi[dfgi['inh'] == inh].pivot(index = time_var, columns = area_var, values = 'value')#.fillna(center)
            cbar_dict = dict(cbar = False, cbar_kws = {'label': label, 'shrink': 0.7, 'pad':0.03})#'width':"5%", 'height':"50%",'fraction': 0.1, 
        else:
            #ax = axs[num, 0]
            ax = axs[0, num]
            dfg1 = dfgi[dfgi['inh'] == inh].pivot(index = area_var, columns = time_var, values = 'value')#.sort_index(by = regio_order)
            cbar_dict = dict(yticklabels = True, cbar = False, cbar_kws = {'label': label, 'shrink': 0.3, 'pad':0.05},)#'width':"5%", 'height':"50%",'fraction': 0.1, 
        if filter:
            test = hfs.heatmap_filter(dfg1,filter)
            #if q == dfc.inh_order[-1]:
        #cbar_dict = dict(cbar = True, cbar_ax = cbar_ax, cbar_kws = {'label': var})
        
            #else: cbar_dict = dict(cbar = False)
        if inh == axes[-1]:
            cbar_dict.update({'cbar' : True})
        if norm:
            sns.heatmap(dfg1, cmap = c_map, norm = norm, square = True, ax = ax, **cbar_dict)
        else:
            sns.heatmap(dfg1, cmap = c_map, vmin = vmin, vmax = vmax, square = True, ax = ax, **cbar_dict)#rasterized=True, 
        #sns.heatmap(dfg1, cmap = c_map, norm = TwoSlopeNorm(center, vmin, vmax), square = True, ax = ax, **cbar_dict)#rasterized=True, 
        #ax.set_title(inh)
        ax.set(xlabel="", ylabel="", title = f'{var} percent of ctrl, {xtra}' if kind == 'per_ctrl' else inh)
        
        if orientation == 'vertical':
            
            ax.xaxis.tick_top()
            #ax.set_xticklabels(rotation=45)
            plt.xticks(rotation=90)

            y_vals = dfg1.index
            ytick_idx = [y_vals.get_loc(tick) for tick in tick_dic[time_var] if tick in y_vals]
            #ytick_idx = [y_vals.get_loc(tick) for tick in yticks]
            yticklabels = [y_vals[idx] for idx in ytick_idx]
            ax.set_yticks(ytick_idx , labels = yticklabels, va="center")
        else:
            
            
            x_vals = dfg1.columns
            xtick_idx = [x_vals.get_loc(tick) for tick in tick_dic[time_var] if tick in x_vals]
            #ytick_idx = [y_vals.get_loc(tick) for tick in yticks]
            xticklabels = [x_vals[idx] for idx in xtick_idx]
            ax.set_xticks(xtick_idx , labels = xticklabels, va="center")
            #ax.invert_yaxis()
        
        #ax.set_title(f'{var} percent of ctrl, {xtra}')
    cmap_name = c_map if isinstance(c_map, str) else c_map.name
    fig_name = f'Heatmap {area_var} {kind} {var} {cmap_name} {time_var} {orientation}'
    if xtra:
        fig_name += f' {xtra}'
    dfc.save_fig(fig_name)
    
    return dfgi


#---------------------------------------------------------------------------
# STANDARD PARAMS FUNCTION
#---------------------------------------------------------------------------

def std_params_plot(select_keys = False, select_vars = False):
    
    std=dict(
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
        'iso_max': (0,100),
        'ylim': (False, 60),#80,
        'axhline':False,
        'sup':'Platelet corrected calcium',
        'y_lab':r"Calcium (AU)",
        #'y_lab':r"Corrected CAL520 fluorescence (AU)",
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
        'y_lab':r"Cluster distance ($\mu$m)",
        
    },
    nba_d_10={
        'iso_max': (0,84),#50,
        'ylim': (False, 20),
        'axhline':False,
        'sup':'Average distance to 10 closest platelets',
        'y_lab':r"Cluster distance ($\mu$m)",
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
    track_time ={
        'iso_max':(0, 90),
        'ylim': (False, 600),#600,
        'axhline':False,
        'sup':'Cohesion',
        'y_lab':r"Plt cohesion (s)",
        'logscale' : True,
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
        'iso_max':(0, 70),
        #'iso_max':(0, 62),
        'ylim': (-250, 100),
        'axhline':True,
        'sup':'Movement, y axis',
        'y_lab':f"Δy (nm/s)",
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
    azi_diff={
        'ylim': (False, 0.5),
        'iso_max':(0, 70),
        'axhline':True,
        'sup':'Angular velocity',
        'y_lab':"Angular velocity (deg/s)",
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
    
    #params = std['std']
    if select_vars:
        std = {key: value for key,value in std.items() if key in select_vars}
    if select_keys:
        #std1={}
        for var,dic_ in std.items():
            dic_n = {key: value for key,value in dic_.items() if key in select_keys}
        std[var]=dic_n

    return std


