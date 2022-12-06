#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os 
import pandas as pd
import math

def paint(eps, ls_or_dis, file_name, all_or_global, smp):
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    ja10, ja20, ja30 = [], [], []

    if ls_or_dis == 'loss':
        for it in eps:
            if it == 0:
                ja10.append( df[ (df['atk_elp']==eps[1]) & (df['model_type']=='JCBgma10') & (df['smooth_sample']==smp)]['ord_loss'].values)
                ja20.append( df[ (df['atk_elp']==eps[1]) & (df['model_type']=='JCBgma20') & (df['smooth_sample']==smp)]['ord_loss'].values)
                ja30.append( df[ (df['atk_elp']==eps[1]) & (df['model_type']=='JCBgma30') & (df['smooth_sample']==smp)]['ord_loss'].values)
            else:
                ja10.append( df[ (df['atk_elp']==it) & (df['model_type']=='JCBgma10') & (df['smooth_sample']==smp)]['atk_loss'].values)
                ja20.append( df[ (df['atk_elp']==it) & (df['model_type']=='JCBgma20') & (df['smooth_sample']==smp)]['atk_loss'].values)
                ja30.append( df[ (df['atk_elp']==it) & (df['model_type']=='JCBgma30') & (df['smooth_sample']==smp)]['atk_loss'].values)
    elif ls_or_dis == 'discrepancy':
        for it in eps:
            ja10.append( df[ (df['atk_elp']==it) & (df['model_type']=='JCBgma10') & (df['smooth_sample']==smp)]['atk_dis'].values[0])
            ja20.append( df[ (df['atk_elp']==it) & (df['model_type']=='JCBgma20') & (df['smooth_sample']==smp)]['atk_dis'].values[0])
            ja30.append( df[ (df['atk_elp']==it) & (df['model_type']=='JCBgma30') & (df['smooth_sample']==smp)]['atk_dis'].values[0])

    plt.figure(figsize=(16,8))   
    plot_title = f'{all_or_global}_{ls_or_dis}'
    
    plt.plot(eps, ja10, color='olive', marker='s', label = 'Jacobian_10', alpha=1, linewidth=2.0, markersize=3)##### 
    plt.plot(eps, ja20, color='forestgreen', marker='s', label = 'Jacobian_20', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, ja30, color='lawngreen', marker='s', label = 'Jacobian_30', alpha=1, linewidth=2.0, markersize=3)#####

    plt.xlabel('Adversarial PGA attack norm bound', fontsize=15)
    plt.xticks(np.array(eps), fontsize=15)
    plt.ylabel(ls_or_dis, fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(plot_title, fontsize=15)
    plt.legend(prop={'size': 13})

    epss = '_'.join( str(eps).split('.') )
    sth = 1 if smp else 0
    img_name = f'plots/{all_or_global}_{ls_or_dis}_smooth_{sth}_Jacobian_PGA_attack_comparisons.jpg'
    plt.savefig(img_name)


def dis2(eps, ls_or_dis, file_name, name):
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    org_bc, spc_bc, adv_bc, smt_bc, advs_bc, smta_bc, org, spc, adv, smt, advs, smta = [], [], [], [], [], [], [], [], [], [], [], []

    if ls_or_dis == 'discrepancy':
        for atk in eps:
            org_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ISTA') ]['atk_dis'].values[0])
            spc_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_dis'].values[0])
            adv_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_dis'].values[0])
            advs_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_dis'].values[0])
            smt_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_dis'].values[0])
            # smta_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_dis'].values[0])

            org.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ISTA') ]['atk_dis'].values[0])
            spc.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_dis'].values[0])
            adv.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_dis'].values[0])
            smt.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_dis'].values[0])
            advs.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_dis'].values[0])
            # smta.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_dis'].values[0])
    
    elif ls_or_dis == 'loss':        
        for atk in eps:
            org_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ISTA') ]['atk_loss'].values[0])
            spc_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_loss'].values[0])
            adv_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_loss'].values[0])
            advs_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_loss'].values[0])
            smt_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_loss'].values[0])
            # smta_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_loss'].values[0])

            org.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ISTA') ]['atk_loss'].values[0])
            spc.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_loss'].values[0])
            adv.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_loss'].values[0])
            advs.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_loss'].values[0])
            smt.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_loss'].values[0])
            # smta.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_loss'].values[0])
    
    elif ls_or_dis == 'psnr':        
        for atk in eps:
            org_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ISTA') ]['atk_psnr'].values[0])
            spc_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_psnr'].values[0])
            adv_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_psnr'].values[0])
            advs_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_psnr'].values[0])
            smt_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_psnr'].values[0])
            # smta_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_psnr'].values[0])

            org.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ISTA') ]['atk_psnr'].values[0])
            spc.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_psnr'].values[0])
            adv.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_psnr'].values[0])
            advs.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_psnr'].values[0])
            smt.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_psnr'].values[0])
            # smta.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_psnr'].values[0])

    plt.figure(figsize=(8,6.5))   
    plot_title = f'({ls_or_dis}) Adversarial attack comparisons'


    plt.plot(eps, org_bc, color='hotpink', marker='s', label = 'Ord', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, spc_bc, color='gold', marker='s', label = 'Jcb', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, adv_bc, color='yellowgreen', marker='s', label = 'Adv', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, advs_bc, color='deepskyblue', marker='s', label = 'Smt-Adv', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, smt_bc, color='orangered', marker='s', label = 'Smt-Grad', alpha=1, linewidth=2.0, markersize=3)#####
    # plt.plot(eps, smta_bc, color='lightgrey', marker='s', label = 'smt_grad_adv', alpha=1, linewidth=2.0, markersize=3)#####
    
    plt.plot(eps, org, color='purple', marker='s', label = 'Ord + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, spc, color='darkorange', marker='s', label = 'Jcb + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, adv, color='forestgreen', marker='s', label = 'Adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, advs, color='navy', marker='s', label = 'Smt-Adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, smt, color='darkred', marker='s', label = 'Smt-Grad + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    # plt.plot(eps, smta, color='dimgrey', marker='s', label = 'smt_grad_adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    
    plt.xlabel('Attack Radius', fontsize=15)
    plt.xticks(np.array(eps), fontsize=15)
    plt.ylabel(ls_or_dis, fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(plot_title, fontsize=15)
    plt.legend(prop={'size': 15})

    epss = '_'.join( str(eps).split('.') )
    img_name = f'plots/{name}_({ls_or_dis}).jpg'
    plt.savefig(img_name)



if __name__ == '__main__':

    # #Jcb
    # paint(eps=[0,0.1,0.5,1,5,12,18,21,25,30], ls_or_dis='loss', file_name='itr10alp800_final', all_or_global='all', smp=0)
    # paint(eps=[0.1,0.5,1,5,12,18,21,25,30], ls_or_dis='discrepancy', file_name='itr10alp800_final', all_or_global='all', smp=0)
    # paint(eps=[0,0.1,0.5,1,5,12,18,21,25,30], ls_or_dis='loss', file_name='itr10alp800_final', all_or_global='all', smp=100)
    # paint(eps=[0.1,0.5,1,5,12,18,21,25,30], ls_or_dis='discrepancy', file_name='itr10alp800_final', all_or_global='all', smp=100)


    # oct best
    # dis2([1,5,12,18,25,30,36], 'discrepancy', 'test_oct27', 'nov28_800')
    # dis2([1,5,12,18,25,30,36], 'loss', 'test_oct27', 'nov28_800')
    # dis2([1,5,12,18,25,30,36], 'psnr', 'test_oct27', 'nov28_800')


    #below nov try
    # dis2([1, 5, 10, 20, 30, 50], 'discrepancy', 'test_nov28_1500', 'nov28_1500')
    # dis2([1, 5, 10, 20, 30, 50], 'loss', 'test_nov28_1500', 'nov28_1500')
    # dis2([1, 5, 10, 20, 30, 50], 'psnr', 'test_nov28_1500', 'nov28_1500')
    # # dis2([1, 5, 10, 20, 30, 50], 'discrepancy', 'test_nov27', 'nov26')
    # # dis2([1, 5, 10, 20, 30, 50], 'loss', 'test_nov27', 'nov26')
    # # dis2([1, 5, 10, 20, 30, 50], 'psnr', 'test_nov27', 'nov26')
    # dis2([1, 5, 10, 20, 30, 50], 'discrepancy', 'test_nov28_1000', 'nov28_1000')
    # dis2([1, 5, 10, 20, 30, 50], 'loss', 'test_nov28_1000', 'nov28_1000')
    # dis2([1, 5, 10, 20, 30, 50], 'psnr', 'test_nov28_1000', 'nov28_1000')

    # dis2([1, 5, 10, 15, 20], 'discrepancy', 'test_nov28_500', 'nov28_500')
    # dis2([1, 5, 10, 15, 20], 'loss', 'test_nov28_500', 'nov28_500')
    # dis2([1, 5, 10, 15, 20], 'psnr', 'test_nov28_500', 'nov28_500')

    # dis2([1, 5, 10, 20, 30, 50], 'discrepancy', 'test_nov28_200', 'nov28_200')
    # dis2([1, 5, 10, 20, 30, 50], 'loss', 'test_nov28_200', 'nov28_200')
    # dis2([1, 5, 10, 20, 30, 50], 'psnr', 'test_nov28_200', 'nov28_200')


    # dis2([1, 5, 10, 20], 'discrepancy', 'test_nov28_500', 'nov28_500_part')
    # dis2([1, 5, 10, 20], 'loss', 'test_nov28_500', 'nov28_500_part')
    # dis2([1, 5, 10, 20], 'psnr', 'test_nov28_500', 'nov28_500_part')


    def c(n):
        a =  (5+3*math.sqrt(5))/10 * math.pow(((1+math.sqrt(5))/2),n) + (5-3*math.sqrt(5))/10 * math.pow(((1-math.sqrt(5))/2),n)
        return  math.log(a,2) / n

    def c1(n):
        a =  (5+math.sqrt(5))/10 * math.pow(((1+math.sqrt(5))/2),n) + (5-math.sqrt(5))/10 * math.pow(((1-math.sqrt(5))/2),n)
        return  math.log(a,2) / n

    rst = []
    b_rst = []
    y_rst = []
    for i in range(1,101):
        rst.append(c(i))
        b_rst.append(c1(i))
        y_rst.append(math.log((1+math.sqrt(5))/2,2))



    plt.figure(figsize=(8,6.5))   
    plt.plot([i for i in range(1,101)], rst, color='red', marker='s', label = r'$\frac{\log \alpha_n}{n}$', alpha=1, linewidth=1.0, markersize=1)#####
    plt.plot([i for i in range(1,101)], y_rst, color='green', marker='s', label = r'$\log \frac{1+\sqrt{5}}{2}$', alpha=1, linewidth=1.0, markersize=1)#####
    plt.plot([i for i in range(1,101)], b_rst, color='blue', marker='s', label = r'$\frac{\log \beta_n}{n}$', alpha=1, linewidth=1.0, markersize=1)#####
    plt.xlabel('n', fontsize=15)
    plt.xticks(np.array([i for i in range(1,101)]), fontsize=15)
    plt.locator_params(axis='x', nbins=5)


    plt.yticks(fontsize=15)
    plt.legend(prop={'size': 17})
    img_name = f'plots/test.jpg'
    plt.savefig(img_name)

    