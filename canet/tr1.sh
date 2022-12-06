# python train.py --gpu_idx 1 --atk 1 --itr 2 --alp 5000 --eps 5
python train.py --gpu_idx 0,1 --atk 1 --itr 2 --alp 5000 --eps 5 --smp 10 --std 0.1

# python train.py --gpu_idx 1
# python train.py --gpu_idx 1 --jcb 1 --gma 10
# python train.py --gpu_idx 1 --jcb 1 --gma 20
# python train.py --gpu_idx 1 --jcb 1 --gma 30
# python train.py --gpu_idx 1 --smt 1 --smp 10 --std 0.1 --stp 1 #algo3S