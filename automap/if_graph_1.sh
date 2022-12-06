## Attack ##
for i in 0.05 0.5 2.0
do
    #ordinary
    python automap_main_inference.py --gpu 1 --pkl Sep17 --org 1 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --spc 1 --jaj 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --smp 15 --std 0.1 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 1 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 6 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 10 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 6 --wmp 3 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
    # python automap_main_inference.py --gpu 1 --pkl Sep22 --smt 1 --smp 15 --std 0.1 --sts 6 --wmp 0 --itr 6 --alp 0.5 --eps 0.05 --asc pga --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga

    #smoothing 
    python automap_main_inference.py --gpu 1 --pkl Sep17 --org 1 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --spc 1 --jaj 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --smp 15 --std 0.1 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 1 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 6 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 10 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
    python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 6 --wmp 3 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
    # python automap_main_inference.py --gpu 1 --pkl Sep22 --smt 1 --smp 15 --std 0.1 --sts 6 --wmp 0 --itr 6 --alp 0.5 --eps 0.05 --asc pga --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
done




## visual ##
# for i in 0.01 0.1 1.0 2.0
# do
#     # # #ordinary
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --org 1 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 6 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --smp 15 --std 0.1 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --spc 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 1 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --vis 1

#     # #smoothing 
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --org 1 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --smt 1 --smp 15 --std 0.1 --sts 6 --wmp 0 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --adv 1 --itr 6 --alp 0.5 --eps 0.05 --asc pga --smp 15 --std 0.1 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     # python automap_main_inference.py --gpu 1 --pkl Sep17 --spc 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
# done



# ## jacobian ##
# for i in 0.01 0.05 0.1 0.5 1.0 2.0
# do  
#     #ordinary
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --gma 10 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --jaj 1 --gma 10 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --jaj 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --jaj 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --gma 10 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --jaj 1 --gma 10 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --jaj 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --jaj 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga

#     #smth
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --gma 10 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --jaj 1 --gma 10 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --jaj 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --jcb 1 --jaj 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --gma 10 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --jaj 1 --gma 10 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --jaj 1 --gma 20 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
#     python automap_main_inference.py --gpu 1 --pkl tableaug21 --spc 1 --jaj 1 --gma 30 --atk 1 --a_itr 5 --a_alp 15.0 --a_eps $i --a_asc pga --atxs 1 --a_smp 100 --a_std 0.1
# done



