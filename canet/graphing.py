#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os 
import pandas as pd

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


def dis2(eps, ls_or_dis, file_name, name, num_pixels=321*481):
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    org_bc, spc_bc, adv_bc, smt_bc, advs_bc, smta_bc, org, spc, adv, smt, advs, smta = [], [], [], [], [], [], [], [], [], [], [], []

    if ls_or_dis == 'discrepancy':
        for atk in eps:
            org_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ordinary') ]['atk_dis'].values[0]/num_pixels)
            spc_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_dis'].values[0]/num_pixels)
            adv_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_dis'].values[0]/num_pixels)
            advs_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_dis'].values[0]/num_pixels)
            smt_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_dis'].values[0]/num_pixels)

            org.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ordinary') ]['atk_dis'].values[0]/num_pixels)
            spc.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_dis'].values[0]/num_pixels)
            adv.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_dis'].values[0]/num_pixels)
            smt.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_dis'].values[0]/num_pixels)
            advs.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_dis'].values[0]/num_pixels)
    
    elif ls_or_dis == 'loss':        
        for atk in eps:
            org_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ordinary') ]['atk_loss'].values[0]/num_pixels)
            spc_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_loss'].values[0]/num_pixels)
            adv_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_loss'].values[0]/num_pixels)
            advs_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_loss'].values[0]/num_pixels)
            smt_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_loss'].values[0]/num_pixels)
            # smta_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_loss'].values[0])

            org.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ordinary') ]['atk_loss'].values[0]/num_pixels)
            spc.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_loss'].values[0]/num_pixels)
            adv.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_loss'].values[0]/num_pixels)
            advs.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_loss'].values[0]/num_pixels)
            smt.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_loss'].values[0]/num_pixels)
            # smta.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_loss'].values[0])
    
    elif ls_or_dis == 'psnr':        
        for atk in eps:
            org_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ordinary') ]['atk_psnr'].values[0])
            spc_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_psnr'].values[0])
            adv_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_psnr'].values[0])
            advs_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_psnr'].values[0])
            smt_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_psnr'].values[0])
            # smta_bc.append( df[ (df['smooth_sample']==0) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_psnr'].values[0])

            org.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ordinary') ]['atk_psnr'].values[0])
            spc.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='JCBgma20') ]['atk_psnr'].values[0])
            adv.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='ADV') ]['atk_psnr'].values[0])
            advs.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='smtADV') ]['atk_psnr'].values[0])
            smt.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH') ]['atk_psnr'].values[0])
            # smta.append( df[ (df['smooth_sample']==100) & (df['atk_elp']==atk) & (df['model_type']=='STH_itr6_alp0.5_eps0.05_pga') ]['atk_psnr'].values[0])

    plt.figure(figsize=(8,6.5))   
    plot_title = f'({ls_or_dis}) Adversarial attack comparisons'

    # if name == 'all':
    plt.plot(eps, org_bc, color='hotpink', marker='s', label = 'Ord', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, spc_bc, color='gold', marker='s', label = 'Jcb', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, adv_bc, color='yellowgreen', marker='s', label = 'Adv', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, advs_bc, color='deepskyblue', marker='s', label = 'Smt-Adv', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, smt_bc, color='orangered', marker='s', label = 'Smt-Grad', alpha=1, linewidth=2.0, markersize=3)#####
    # plt.plot(eps, smta_bc, color='lightgrey', marker='s', label = 'smt_grad_adv', alpha=1, linewidth=2.0, markersize=3)#####
    
    # if name == 'all':
    plt.plot(eps, org, color='purple', marker='s', label = 'Ord + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, spc, color='darkorange', marker='s', label = 'Jcb + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, adv, color='forestgreen', marker='s', label = 'Adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, advs, color='navy', marker='s', label = 'Smt-Adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, smt, color='darkred', marker='s', label = 'Smt-Grad + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    # plt.plot(eps, smta, color='dimgrey', marker='s', label = 'smt_grad_adv_S', alpha=1, linewidth=2.0, markersize=3)#####
    
    plt.xlabel('Attack Radius', fontsize=15)
    plt.xticks(np.array(eps), fontsize=15)
    plt.ylabel(ls_or_dis, fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(plot_title, fontsize=15)
    plt.legend(prop={'size': 15})

    epss = '_'.join( str(eps).split('.') )
    img_name = f'plots/({ls_or_dis})_{name}.jpg'
    plt.savefig(img_name)







def noise(stds, ls_or_dis, file_name, name, num_pixels=321*481):
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    org_bc, spc_bc, adv_bc, smt_bc, advs_bc, smta_bc, org, spc, adv, smt, advs, smta = [], [], [], [], [], [], [], [], [], [], [], []

    if ls_or_dis == 'discrepancy':
        for std in stds:

            org.append( df[ (df['smooth_std']==std) & (df['model_type']=='ordinary') ]['atk_dis'].values[0])
            spc.append( df[ (df['smooth_std']==std) & (df['model_type']=='JCBgma20') ]['atk_dis'].values[0])
            adv.append( df[ (df['smooth_std']==std) & (df['model_type']=='ADV') ]['atk_dis'].values[0])
            smt.append( df[ (df['smooth_std']==std) & (df['model_type']=='STH') ]['atk_dis'].values[0])
            advs.append( df[ (df['smooth_std']==std) & (df['model_type']=='smtADV') ]['atk_dis'].values[0])
    
    elif ls_or_dis == 'loss':        
        for std in stds:

            org.append( df[ (df['smooth_std']==std) & (df['model_type']=='ordinary') ]['atk_loss'].values[0]/num_pixels)
            spc.append( df[ (df['smooth_std']==std) & (df['model_type']=='JCBgma20') ]['atk_loss'].values[0]/num_pixels)
            adv.append( df[ (df['smooth_std']==std) & (df['model_type']=='ADV') ]['atk_loss'].values[0]/num_pixels)
            advs.append( df[ (df['smooth_std']==std) & (df['model_type']=='smtADV') ]['atk_loss'].values[0]/num_pixels)
            smt.append( df[ (df['smooth_std']==std) & (df['model_type']=='STH') ]['atk_loss'].values[0]/num_pixels)
    
    elif ls_or_dis == 'psnr':        
        for std in stds:

            org.append( df[ (df['smooth_std']==std) & (df['model_type']=='ordinary') ]['atk_psnr'].values[0])
            spc.append( df[ (df['smooth_std']==std) & (df['model_type']=='JCBgma20') ]['atk_psnr'].values[0])
            adv.append( df[ (df['smooth_std']==std) & (df['model_type']=='ADV') ]['atk_psnr'].values[0])
            advs.append( df[ (df['smooth_std']==std) & (df['model_type']=='smtADV') ]['atk_psnr'].values[0])
            smt.append( df[ (df['smooth_std']==std) & (df['model_type']=='STH') ]['atk_psnr'].values[0])

    plt.figure(figsize=(16,8))   
    plot_title = f'({ls_or_dis}) Adversarial attack comparisons'

    plt.plot(stds, org, color='purple', marker='s', label = 'Ord + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(stds, spc, color='darkorange', marker='s', label = 'Jcb + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(stds, adv, color='forestgreen', marker='s', label = 'Adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(stds, advs, color='navy', marker='s', label = 'Smt-Adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(stds, smt, color='darkred', marker='s', label = 'Smt-Grad + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    
    plt.xlabel('smoothing noise', fontsize=14)
    plt.xticks(np.array(stds), rotation=45, fontsize=13)
    plt.ylabel(ls_or_dis, fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(plot_title, fontsize=15)
    plt.legend(prop={'size': 14})

    img_name = f'plots/{name}_({ls_or_dis})_nov21.jpg'
    plt.savefig(img_name)





if __name__ == '__main__':

    # #Jcb
    # paint(eps=[0,0.1,0.5,1,5,10,15,20], ls_or_dis='loss', file_name='casnet_sep30', all_or_global='all', smp=0)
    # paint(eps=[0.1,0.5,1,5,10,15,20], ls_or_dis='discrepancy', file_name='casnet_sep30', all_or_global='all', smp=0)
    # paint(eps=[0,0.1,0.5,1,5,10,15,20], ls_or_dis='loss', file_name='casnet_sep30', all_or_global='all', smp=250)
    # paint(eps=[0.1,0.5,1,5,10,15,20], ls_or_dis='discrepancy', file_name='casnet_sep30', all_or_global='all', smp=250)


    # #pga
    # dis2([0.1,0.5,1,5,10,15], 'discrepancy', 'casnet_sep30_all', 'nov21')
    # dis2([0.1,0.5,1,5,10,15], 'loss', 'casnet_sep30_all', 'nov21')
    # dis2([0.1,0.5,1,5,10,15], 'psnr', 'casnet_sep30_all', 'nov21')

    # dis2([0.1, 0.5, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50], 'discrepancy', 'casnet_oct25', 'nov24')
    # dis2([0.1, 0.5, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50], 'loss', 'casnet_oct25', 'nov24')
    # dis2([0.1, 0.5, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50], 'psnr', 'casnet_oct25', 'nov24')
    
    # dis2([10, 20, 50, 90], 'discrepancy', 'casnet_nov25_new', 'nov25')
    # dis2([10, 20, 50, 90], 'loss', 'casnet_nov25_new', 'nov25')
    # dis2([10, 20, 50, 90], 'psnr', 'casnet_nov25_new', 'nov25')
    dis2([10, 25, 50, 100, 150], 'discrepancy', 'casnet_nov26_new', 'nov26')
    dis2([10, 25, 50, 100, 150], 'loss', 'casnet_nov26_new', 'nov26')
    dis2([10, 25, 50, 100, 150], 'psnr', 'casnet_nov26_new', 'nov26')


    # noise([0, 0.01, 0.1, 0.5, 2, 5], 'discrepancy', 'casnet_oct26', 'nov16')
    # noise([0, 0.01, 0.1, 0.5, 2, 5], 'loss', 'casnet_oct26', 'nov16')
    # noise([0, 0.01, 0.1, 0.5, 2, 5], 'psnr', 'casnet_oct26', 'nov16')


    # noise([0, 0.01, 0.1], 'discrepancy', 'casnet_oct27', 'nov17')
    # noise([0, 0.01, 0.1], 'loss', 'casnet_oct27', 'nov17')
    # noise([0, 0.01, 0.1], 'psnr', 'casnet_oct27', 'nov17')
