import tensorflow as tf

from data_loader.automap_inference_data_generator import InferenceDataGenerator
from trainers.automap_inferencer import AUTOMAP_Inferencer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from argparse import ArgumentParser
import pickle as pkl
import pandas as pd
import os


def main(hparams):

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[hparams.gpu], 'GPU')

    try:
        #args = get_args()
        config = process_config(hparams.cfg, hparams)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create_dirs([config.summary_dir, config.checkpoint_dir])
    data = InferenceDataGenerator(config)
    model = tf.keras.models.load_model(config.checkpoint_dir)
    
    #inferencer = AUTOMAP_Inferencer(model, data, config)
    #inferencer.inference()
    config.pickle_file_path = 'test_{}.pkl'.format(hparams.pkl)
    inferencer = AUTOMAP_Inferencer(model, data, config, 
        attack_iter=hparams.a_itr, alpha=hparams.a_alp, eps=hparams.a_eps, ascent=hparams.a_asc, beta=hparams.bta, 
        vis=hparams.vis,
        atk=hparams.atk, atxs=hparams.atxs, 
        denoise=hparams.dno, sample=hparams.a_smp, std=hparams.a_std, m_m=hparams.smm)
    inferencer.inference_attack()

    with open(config.pickle_file_path, "rb") as f:
        object = pkl.load(f)
    df = pd.DataFrame(object)
    df.to_csv(f'test_{hparams.pkl}.csv', index=False)


if __name__ == '__main__':
    
    PARSER = ArgumentParser()
    PARSER.add_argument('--cfg', type=str, default='configs/inference_64x64_ex.json', help='config file')
    PARSER.add_argument('--adv', type=int, default=0, help='adv model')
    PARSER.add_argument('--org', type=int, default=0, help='original model')
    PARSER.add_argument('--jcb', type=int, default=0, help='jacobian=1')
    PARSER.add_argument('--spc', type=int, default=0, help='spectral=1')
    PARSER.add_argument('--jaj', type=int, default=0, help='jaj=1')
    PARSER.add_argument('--smt', type=int, default=0, help='smt=1')
    PARSER.add_argument('--gma', type=int, default=20, help='jacobian beta')

    PARSER.add_argument('--itr', type=int, default=0, help='iterations')
    PARSER.add_argument('--alp', type=float, default=0., help='alpha')
    PARSER.add_argument('--eps', type=float, default=0., help='epsilon')
    PARSER.add_argument('--asc', type=str, default='pga', help='fgsm or pga')
    PARSER.add_argument('--bta', type=float, default=0., help='beta')

    PARSER.add_argument('--atk', type=int, default=0, help='atk mode')
    PARSER.add_argument('--atxs', type=int, default=0, help='atk mode + smooth version')
    PARSER.add_argument('--a_itr', type=int, default=0, help='iterations')
    PARSER.add_argument('--a_alp', type=float, default=0, help='alpha')
    PARSER.add_argument('--a_eps', type=float, default=0, help='epsilon')
    PARSER.add_argument('--a_asc', type=str, default='pga', help='fgsm or pga')
    
    PARSER.add_argument('--smp', type=int, default=0, help='samples')
    PARSER.add_argument('--std', type=float, default=0, help='std')
    PARSER.add_argument('--sts', type=int, default=0, help='std step')
    PARSER.add_argument('--wmp', type=int, default=0, help='warmup epochs')
    PARSER.add_argument('--a_smp', type=int, default=0, help='samples')
    PARSER.add_argument('--a_std', type=float, default=0, help='std')
    PARSER.add_argument('--smm', type=str, default='median', help='median or mean')
    PARSER.add_argument('--dno', type=int, default=0, help='smooth denoise')

    PARSER.add_argument('--pkl', type=str, default='pkile', help='pickle file')
    PARSER.add_argument('--vis', type=int, default=0, help='visualize')

    PARSER.add_argument('--gpu', type=int, default=0, help='gpu index')



    HPARAMS = PARSER.parse_args()

    main(HPARAMS)
