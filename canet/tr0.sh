# python train.py --itr 10 --alp 200 --eps 0.01
# python train.py --itr 10 --alp 200 --eps 0.01 --smp 10 --std 0.05

# python train.py --gpu_idx 0
# python train.py --gpu_idx 0 --jcb 1 --gma 10
# python train.py --gpu_idx 0 --jcb 1 --gma 20
# python train.py --gpu_idx 0 --jcb 1 --gma 30
python train.py --gpu_idx 0,1 --smt 1 --smp 10 --std 0.1 --stp 6  #algo3
