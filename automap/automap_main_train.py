import tensorflow as tf

from data_loader.automap_data_generator import DataGenerator, ValDataGenerator
from models.automap_model import AUTOMAP_Basic_Model 
from trainers.automap_trainer import AUTOMAP_Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

from argparse import ArgumentParser




def main(hparams):

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[hparams.gpu], 'GPU')

    try:
        # args = get_args()
        config = process_config(hparams.cfg, hparams)

    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])
    data = DataGenerator(config)
    valdata = ValDataGenerator(config)

    if config.resume ==0:
        model = AUTOMAP_Basic_Model(config)
    elif config.resume == 1:
        model = tf.keras.models.load_model(config.loadmodel_dir)
    model.summary()

    
    if hparams.jcb:
        # jacobian regularzation
        if hparams.jaj:
            trainer = AUTOMAP_Trainer(model=model, data=data, valdata=valdata, config=config, jacobian=1, jaj=1, jcb_gamma=hparams.gma)
            trainer.jcb_train()
        else:
            trainer = AUTOMAP_Trainer(model=model, data=data, valdata=valdata, config=config, jacobian=1, jcb_gamma=hparams.gma)
            trainer.jcb_train()
    elif hparams.spc:
        # Spectral norm of jacobian regularzation
        if hparams.jaj:
            trainer = AUTOMAP_Trainer(model=model, data=data, valdata=valdata, config=config, spectral=1, jaj=1, jcb_gamma=hparams.gma)
            trainer.jcb_train()
        else:
            trainer = AUTOMAP_Trainer(model=model, data=data, valdata=valdata, config=config, spectral=1, jcb_gamma=hparams.gma)
            trainer.jcb_train()
    elif hparams.smt:
        #smooth training
        trainer = AUTOMAP_Trainer(model=model, data=data, valdata=valdata, config=config, sample=hparams.smp, std=hparams.std, std_step=hparams.sts, wmstart=hparams.wmp, attack_iter=hparams.itr, alpha=hparams.alp, eps=hparams.eps, sample_smt=hparams.sample_smt)
        trainer.smooth_train()
    else:
        # adversiary training / ordinary training 
        trainer = AUTOMAP_Trainer(model, data, valdata, config, attack_iter=hparams.itr, alpha=hparams.alp, eps=hparams.eps, ascent=hparams.asc, original=hparams.org, bta=hparams.bta, sample=hparams.smp, std=hparams.std)
        trainer.attack_train()

if __name__ == '__main__':
    
    PARSER = ArgumentParser()
    PARSER.add_argument('--cfg', type=str, default='configs/train_64x64_ex.json', help='config file')
    
    PARSER.add_argument('--adv', type=int, default=0, help='adv=1')
    PARSER.add_argument('--itr', type=int, default=0, help='iterations')
    PARSER.add_argument('--alp', type=float, default=0, help='alpha')
    PARSER.add_argument('--eps', type=float, default=0, help='epsilon')
    PARSER.add_argument('--asc', type=str, default='pga', help='fgsm or pga')
    PARSER.add_argument('--bta', type=float, default=0, help='beta')
    
    PARSER.add_argument('--org', type=int, default=0, help='original=1')
    
    PARSER.add_argument('--jcb', type=int, default=0, help='jacobian=1')
    PARSER.add_argument('--spc', type=int, default=0, help='spectral=1')
    PARSER.add_argument('--jaj', type=int, default=0, help='jaj=1')
    PARSER.add_argument('--gma', type=int, default=20, help='jacobian beta')
    
    PARSER.add_argument('--smt', type=int, default=0, help='smooth=1')
    PARSER.add_argument('--smp', type=int, default=0, help='samples')
    PARSER.add_argument('--std', type=float, default=0, help='std')
    PARSER.add_argument('--sts', type=int, default=0, help='std step')
    PARSER.add_argument('--wmp', type=int, default=0, help='warmup epochs')
    PARSER.add_argument('--sample_smt', type=int, default=0, help='extreme samples')

    PARSER.add_argument('--gpu', type=int, default=0, help='gpu index')

    HPARAMS = PARSER.parse_args()

    main(HPARAMS)
