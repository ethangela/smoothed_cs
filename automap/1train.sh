# python automap_main_train_1.py --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --smp 15 --std 0.1
# python automap_main_train_1.py --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga


# python automap_main_train_1.py --spc 1 --gma 10
# python automap_main_train_1.py --spc 1 --gma 20
# python automap_main_train_1.py --spc 1 --gma 30
# python automap_main_train_1.py --spc 1 --jaj 1 --gma 10
# python automap_main_train_1.py --spc 1 --jaj 1 --gma 20
# python automap_main_train_1.py --spc 1 --jaj 1 --gma 30

# python automap_main_train_1.py --jcb 1 --gma 10
# python automap_main_train_1.py --jcb 1 --gma 20
# python automap_main_train_1.py --jcb 1 --gma 30
# python automap_main_train_1.py --jcb 1 --jaj 1 --gma 10
# python automap_main_train_1.py --jcb 1 --jaj 1 --gma 20
# python automap_main_train_1.py --jcb 1 --jaj 1 --gma 30

python automap_main_train.py --gpu 1 --smt 1 --smp 15 --sts 6 --std 0.1 --wmp 0 --itr 6 --alp 0.5 --eps 0.05 --asc pga --sample_smt 15
