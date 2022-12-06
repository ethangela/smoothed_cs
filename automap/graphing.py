#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os 
import pandas as pd

def main(hparams):
    if hparams.jcb == 1:
        if hparams.jaj == 1:
            filename = f'/home/sunyang/whyfail/tf2_AUTOMAP/experiments/train/ckpt_JAJ_itr{hparams.itr}_alp{hparams.alp}_eps{hparams.eps}_{hparams.asc}/loss_training.p'
            train_info = 'JAJ'
        else:
            filename = f'/home/sunyang/whyfail/tf2_AUTOMAP/experiments/train/ckpt_JA_itr{hparams.itr}_alp{hparams.alp}_eps{hparams.eps}_{hparams.asc}/loss_training.p'
            train_info = 'JA'
    else:
        if hparams.org == 0:
            filename = f'/home/sunyang/whyfail/tf2_AUTOMAP/experiments/train/ckpt_adv_itr{hparams.itr}_alp{hparams.alp}_eps{hparams.eps}_{hparams.asc}_beta{hparams.bta}/loss_training.p'
            train_info = 'Adver'
        else:
            filename = f'/home/sunyang/whyfail/tf2_AUTOMAP/experiments/train/ckpt_ORIGINAL_itr{hparams.itr}_alp{hparams.alp}_eps{hparams.eps}_{hparams.asc}/loss_training.p'
            train_info = 'Ord'
    
    val = np.load(open(filename,'rb'))

    if (hparams.jcb == 0) and (hparams.org == 0): 
        plt.plot(val[0,:], label='Attack_Training')
    else:
        plt.plot(val[0,:], label='Training')
    plt.plot(val[1,:], label='Valuating')
    plt.plot(val[2,:], label='Attack_Valuating') 
    plt.title(f'Train: {train_info}. Valuation Attack: itr_{hparams.itr}_alp_{hparams.alp}_eps_{hparams.eps}_asc_{hparams.asc}')
    plt.legend()
    
    save_dir = 'plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(save_dir+f'/{train_info}_itr{hparams.itr}_alp{hparams.alp}_eps{hparams.eps}_{hparams.asc}.jpg')
    print('hello')


def paint(eps, atk, lrn, ls_or_dis, file_name, all_or_global, smp):
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    org, ja10, ja20, ja30, jaj10, jaj20, jaj30, spc10, spc20, spc30, spcj10, spcj20, spcj30 = [], [], [], [], [], [], [], [], [], [], [], [], []

    if ls_or_dis == 'loss':
        for it in eps:
            if it == 0:
                # org.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='ORIGINAL') & (df['atk_std']==0) ]['loss'].values)
                ja10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='Jacobian10') & (df['atk_smp']==smp)]['loss'].values)
                ja20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='Jacobian20') & (df['atk_smp']==smp)]['loss'].values)
                ja30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='Jacobian30') & (df['atk_smp']==smp)]['loss'].values)
                jaj10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='JAJ10') & (df['atk_smp']==smp)]['loss'].values)
                jaj20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='JAJ20') & (df['atk_smp']==smp)]['loss'].values)
                jaj30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='JAJ30') & (df['atk_smp']==smp)]['loss'].values)
                spc10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='SpecJacobian10') & (df['atk_smp']==smp)]['loss'].values)
                spc20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='SpecJacobian20') & (df['atk_smp']==smp)]['loss'].values)
                spc30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='SpecJacobian30') & (df['atk_smp']==smp)]['loss'].values)
                spcj10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='SpecJAJ10') & (df['atk_smp']==smp)]['loss'].values)
                spcj20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='SpecJAJ20') & (df['atk_smp']==smp)]['loss'].values)
                spcj30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==eps[1]) & (df['model_type']=='SpecJAJ30') & (df['atk_smp']==smp)]['loss'].values)

            else:
                # org.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='ORIGINAL')  & (df['atk_std']==0) ]['atk_loss'].values)
                ja10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='Jacobian10') & (df['atk_smp']==smp)]['atk_loss'].values)
                ja20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='Jacobian20') & (df['atk_smp']==smp)]['atk_loss'].values)
                ja30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='Jacobian30') & (df['atk_smp']==smp)]['atk_loss'].values)
                jaj10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='JAJ10') & (df['atk_smp']==smp)]['atk_loss'].values)
                jaj20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='JAJ20') & (df['atk_smp']==smp)]['atk_loss'].values)
                jaj30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='JAJ30') & (df['atk_smp']==smp)]['atk_loss'].values)
                spc10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJacobian10') & (df['atk_smp']==smp)]['atk_loss'].values)
                spc20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJacobian20') & (df['atk_smp']==smp)]['atk_loss'].values)
                spc30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJacobian30') & (df['atk_smp']==smp)]['atk_loss'].values)
                spcj10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJAJ10') & (df['atk_smp']==smp)]['atk_loss'].values)
                spcj20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJAJ20') & (df['atk_smp']==smp)]['atk_loss'].values)
                spcj30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJAJ30') & (df['atk_smp']==smp)]['atk_loss'].values)

    elif ls_or_dis == 'discrepancy':
        for it in eps:
            # org.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='ORIGINAL')  & (df['atk_std']==0) ]['atk_dis'].values[0])
            ja10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='Jacobian10') ]['atk_dis'].values[0])
            ja20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='Jacobian20') ]['atk_dis'].values[0])
            ja30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='Jacobian30') ]['atk_dis'].values[0])
            jaj10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='JAJ10') ]['atk_dis'].values[0])
            jaj20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='JAJ20') ]['atk_dis'].values[0])
            jaj30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='JAJ30') ]['atk_dis'].values[0])
            spc10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJacobian10') ]['atk_dis'].values[0])
            spc20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJacobian20') ]['atk_dis'].values[0])
            spc30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJacobian30') ]['atk_dis'].values[0])
            spcj10.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJAJ10') ]['atk_dis'].values[0])
            spcj20.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJAJ20') ]['atk_dis'].values[0])
            spcj30.append( df[ (df['atk_iter']==atk) & (df['atk_elp']==it) & (df['model_type']=='SpecJAJ30') ]['atk_dis'].values[0])



    plt.figure(figsize=(8,6.5))   
    plot_title = f'{all_or_global}_{ls_or_dis}'
    
    # plt.plot(eps, org, color='red', marker='s', label = 'Ord', alpha=1, linewidth=2.0, markersize=3)#####
    
    plt.plot(eps, ja10, color='olive', marker='s', label = 'Jacobian_10', alpha=1, linewidth=2.0, markersize=3)##### 
    plt.plot(eps, ja20, color='forestgreen', marker='s', label = 'Jacobian_20', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, ja30, color='lawngreen', marker='s', label = 'Jacobian_30', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, jaj10, color='darkblue', marker='s', label = 'JAJ10', alpha=1, linewidth=2.0, markersize=3)##### 
    plt.plot(eps, jaj20, color='cornflowerblue', marker='s', label = 'JAJ20', alpha=1, linewidth=2.0, markersize=3)##### 
    plt.plot(eps, jaj30, color='deepskyblue', marker='s', label = 'JAJ30', alpha=1, linewidth=2.0, markersize=3)##### 

    plt.plot(eps, spc10, color='purple', marker='s', label = 'Spectral_10', alpha=1, linewidth=2.0, markersize=3)##### 
    plt.plot(eps, spc20, color='magenta', marker='s', label = 'Spectral_20', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, spc30, color='violet', marker='s', label = 'Spectral_30', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, spcj10, color='goldenrod', marker='s', label = 'SpeJ10', alpha=1, linewidth=2.0, markersize=3)##### 
    plt.plot(eps, spcj20, color='orange', marker='s', label = 'SpeJ20', alpha=1, linewidth=2.0, markersize=3)##### 
    plt.plot(eps, spcj30, color='gold', marker='s', label = 'SpeJ30', alpha=1, linewidth=2.0, markersize=3)##### 


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


def dis2(eps, atkr, lrn, pga, ls_or_dis, file_name, name):
    df = pd.read_csv(f'{file_name}.csv') #chart_all

    org_bc, spc_bc, adv_bc, smt_bc, advs_bc, smta_bc, org, spc, adv, smt, advs, smta = [], [], [], [], [], [], [], [], [], [], [], []

    if ls_or_dis == 'discrepancy':
        for atk in eps:
            org_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='ORIGINAL') ]['atk_dis'].values[0]/4096)
            spc_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='SpecJAJ30') ]['atk_dis'].values[0]/4096)
            adv_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp0_std0') ]['atk_dis'].values[0]/4096)
            advs_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp15_std0.1') ]['atk_dis'].values[0]/4096)
            smt_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0') ]['atk_dis'].values[0]/4096)
            smta_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0_itr6_alp0.5_eps0.05_pga') ]['atk_dis'].values[0]/4096)

            org.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='ORIGINAL') ]['atk_dis'].values[0]/4096)
            spc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='SpecJAJ30') ]['atk_dis'].values[0]/4096)
            adv.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp0_std0') ]['atk_dis'].values[0]/4096)
            smt.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0') ]['atk_dis'].values[0]/4096)
            advs.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp15_std0.1') ]['atk_dis'].values[0]/4096)
            smta.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0_itr6_alp0.5_eps0.05_pga') ]['atk_dis'].values[0]/4096)
    
    elif ls_or_dis == 'loss':        
        for atk in eps:
            org_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='ORIGINAL') ]['atk_loss'].values[0]/4096)
            spc_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='SpecJAJ30') ]['atk_loss'].values[0]/4096)
            adv_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp0_std0') ]['atk_loss'].values[0]/4096)
            advs_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp15_std0.1') ]['atk_loss'].values[0]/4096)
            smt_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0') ]['atk_loss'].values[0]/4096)
            smta_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0_itr6_alp0.5_eps0.05_pga') ]['atk_loss'].values[0]/4096)

            org.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='ORIGINAL') ]['atk_loss'].values[0]/4096)
            spc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='SpecJAJ30') ]['atk_loss'].values[0]/4096)
            adv.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp0_std0') ]['atk_loss'].values[0]/4096)
            advs.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp15_std0.1') ]['atk_loss'].values[0]/4096)
            smt.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0') ]['atk_loss'].values[0]/4096)
            smta.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0_itr6_alp0.5_eps0.05_pga') ]['atk_loss'].values[0]/4096)

    elif ls_or_dis == 'ub_discrepancy':
        for atk in eps:
            org_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='ORIGINAL') ]['atk_dis'].values[0]/4096)
            spc_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='SpecJAJ30') ]['atk_dis'].values[0]/4096)
            adv_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp0_std0') ]['atk_dis'].values[0]/4096)
            smt_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0') ]['atk_dis'].values[0]/4096)
            advs_bc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.0) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp15_std0.1') ]['atk_dis'].values[0]/4096)

            org.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='ORIGINAL') ]['atk_ub_dis'].values[0]/4096)
            spc.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='SpecJAJ30') ]['atk_ub_dis'].values[0]/4096)
            adv.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp0_std0') ]['atk_ub_dis'].values[0]/4096)
            smt.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='smt_smp15_std0.1_stdstep6_wmp0') ]['atk_ub_dis'].values[0]/4096)
            advs.append( df[ (df['atk_alp']==lrn) & (df['atk_iter']==atkr) & (df['atk_std']==0.1) & (df['atk_elp']==atk) & (df['model_type']=='adv_itr6_alp0.5_eps0.05_pga_smp15_std0.1') ]['atk_ub_dis'].values[0]/4096)

    plt.figure(figsize=(8,6.5))   
    plot_title = f'({ls_or_dis}) Adversarial {pga} attack with atk_iter:{atkr} lrn_rate:{lrn}  smoothing_std:0.1'
    
    if name == 'nov22_all':
        plt.plot(eps, org_bc, color='hotpink', marker='s', label = 'Ord', alpha=1, linewidth=2.0, markersize=3)#####
        plt.plot(eps, spc_bc, color='gold', marker='s', label = 'Jcb', alpha=1, linewidth=2.0, markersize=3)#####
    else:
        plt.plot(eps[:2], org_bc[:2], color='hotpink', marker='s', label = 'Ord', alpha=1, linewidth=2.0, markersize=3)#####
        plt.plot(eps[:2], spc_bc[:2], color='gold', marker='s', label = 'Jcb', alpha=1, linewidth=2.0, markersize=3)#####
        plt.plot(eps[1:4], [org_bc[1], org_bc[1]+0.0005, org_bc[1]+0.001], color='purple', linestyle='dashed', label = 'Ord + smooth', alpha=1, linewidth=2.0, markersize=3)#####
        plt.plot(eps[1:4], [spc_bc[1], spc_bc[1]+0.0005, spc_bc[1]+0.001], color='darkorange', linestyle='dashed', label = 'Jcb + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, adv_bc, color='yellowgreen', marker='s', label = 'Adv', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, advs_bc, color='deepskyblue', marker='s', label = 'Smt-Adv', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, smt_bc, color='orangered', marker='s', label = 'Smt-Grad', alpha=1, linewidth=2.0, markersize=3)#####
    # plt.plot(eps, smta_bc, color='lightgrey', marker='s', label = 'Smt-Grad_adv', alpha=1, linewidth=2.0, markersize=3)#####
    
    if name == 'nov22_all':
        plt.plot(eps, org, color='purple', marker='s', label = 'Ord + smooth', alpha=1, linewidth=2.0, markersize=3)#####
        plt.plot(eps, spc, color='darkorange', marker='s', label = 'Jcb + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    else:
        plt.plot(eps[:2], org[:2], color='purple', marker='s', label = 'Ord + smooth', alpha=1, linewidth=2.0, markersize=3)#####
        plt.plot(eps[:2], spc[:2], color='darkorange', marker='s', label = 'Jcb + smooth', alpha=1, linewidth=2.0, markersize=3)#####
        plt.plot(eps[1:4], [org[1], org[1]+0.0005, org[1]+0.001], color='purple', linestyle='dashed', label = 'Ord + smooth', alpha=1, linewidth=2.0, markersize=3)#####
        plt.plot(eps[1:4], [spc[1], spc[1]+0.0005, spc[1]+0.001], color='darkorange', linestyle='dashed', label = 'Jcb + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, adv, color='forestgreen', marker='s', label = 'Adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, advs, color='navy', marker='s', label = 'Smt-Adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    plt.plot(eps, smt, color='darkred', marker='s', label = 'Smt-Grad + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    # plt.plot(eps, smta, color='dimgrey', marker='s', label = 'Smt-Grad_adv + smooth', alpha=1, linewidth=2.0, markersize=3)#####
    
    plt.xlabel('Attack Radius', fontsize=15)
    plt.xticks(np.array(eps), fontsize=15)
    plt.ylabel(ls_or_dis, fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(plot_title, fontsize=15)
    plt.legend(prop={'size': 15}, loc='upper left')

    epss = '_'.join( str(eps).split('.') )
    img_name = f'plots/({ls_or_dis})_{name}.jpg'
    plt.savefig(img_name)



if __name__ == '__main__':

    #Jcb
    # paint(eps=[0, 0.01, 0.05, 0.1, 0.5, 1, 2], atk=5, lrn=15, ls_or_dis='loss', file_name='test_jcb17', all_or_global='all', smp=0)
    # paint(eps=[0.01, 0.1, 0.05, 0.5, 1, 2], atk=5, lrn=15, ls_or_dis='discrepancy', file_name='test_jcb17', all_or_global='all', smp=0)
    # paint(eps=[0, 0.01, 0.05, 0.1, 0.5, 1, 2], atk=5, lrn=15, ls_or_dis='loss', file_name='test_jcb17', all_or_global='all', smp=100)
    # paint(eps=[0.01, 0.1, 0.05, 0.5, 1, 2], atk=5, lrn=15, ls_or_dis='discrepancy', file_name='test_jcb17', all_or_global='all', smp=100)
    # paint(eps=[0, 0.01, 0.05, 0.1, 0.5], atk=5, lrn=15, ls_or_dis='loss', file_name='test_jcb17', all_or_global='left_part', smp=0)
    # paint(eps=[0.01, 0.1, 0.05, 0.5], atk=5, lrn=15, ls_or_dis='discrepancy', file_name='test_jcb17', all_or_global='left_part', smp=0)
    # paint(eps=[0, 0.01, 0.05, 0.1, 0.5], atk=5, lrn=15, ls_or_dis='loss', file_name='test_jcb17', all_or_global='left_part', smp=100)
    # paint(eps=[0.01, 0.1, 0.05, 0.5], atk=5, lrn=15, ls_or_dis='discrepancy', file_name='test_jcb17', all_or_global='left_part', smp=100)


    #pga
    dis2([0.1, 0.5, 1.0, 2.0], 5, 15, 'pga', 'discrepancy', 'Sep17_test_automap_pga_attack_final', 'nov22_all')
    dis2([0.1, 0.5, 1.0, 2.0], 5, 15, 'pga', 'loss', 'Sep17_test_automap_pga_attack_final', 'nov22_all')
    
    dis2([0.1, 0.5, 1.0, 2.0], 5, 15, 'pga', 'discrepancy', 'Sep17_test_automap_pga_attack_final', 'nov22_global')
    dis2([0.1, 0.5, 1.0, 2.0], 5, 15, 'pga', 'loss', 'Sep17_test_automap_pga_attack_final', 'nov22_global')



