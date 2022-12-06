#python train.py --gpu 1 --itr 5 --alp 500 --eps 1.0
python train.py --gpu 1 --atk 1 --itr 5 --alp 500 --eps 1.0 --smp 10 --std 0.04

# python train.py --gpu_idx 0 --jcb 1 --gma 10
# python train.py --gpu_idx 0 --jcb 1 --gma 20
# python train.py --gpu_idx 0 --jcb 1 --gma 30

# python train.py --gpu 0 --smt 1 --smp 10 --std 0.05 --stp 6 --itr 10 --alp 200 --eps 0.01 #algo3