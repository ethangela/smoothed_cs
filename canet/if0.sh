


# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.01
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.02
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.03
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.04
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.05
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.06
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.07
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.08
# python test.py --gpu 0 --pkl sep27 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.09



# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.01
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.02
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.03
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.04
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.05
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.06
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.07
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.08
# python test.py --gpu 0 --pkl sep27 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 5 --talp 500 --teps 1.0 --tsmt 1 --tsmp 100 --tstd 0.09







# for i in 0.1 0.5 1.0 5.0 10.0 15.0
# for i in 0.1 0.5 1.0 2.0 5.0 7.5 10.0 12.0 15.0 20.0
# for i in 0.1 0.5 1.0 1.5 2.0 3.0
# for i in 5.0 7.5 10.0 15.0
# do 
# #only atk
# python test.py --gpu 0 --pkl oct25 --tatk 1 --titr 5 --talp 10000 --teps $i
# python test.py --gpu 0 --pkl oct25 --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 10000 --teps $i
# python test.py --gpu 0 --pkl oct25 --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 10000 --teps $i
# python test.py --gpu 0,1 --pkl oct25 --atk 1 --itr 2 --alp 5000 --eps 5 --smp 10 --std 0.1 --tatk 1 --titr 5 --talp 10000 --teps $i
# python test.py --gpu 0 --pkl oct25 --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 10000 --teps $i

# #atk +smt
# python test.py --gpu 0 --pkl oct25 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# python test.py --gpu 0 --pkl oct25 --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# python test.py --gpu 0 --pkl oct25 --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# python test.py --gpu 0,1 --pkl oct25 --atk 1 --itr 2 --alp 5000 --eps 5 --smp 10 --std 0.1 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# python test.py --gpu 0 --pkl oct25 --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# done

# for i in 0.01 0.5 2 5
# do 
# python test.py --gpu 0,1 --pkl oct27 --tatk 1 --titr 5 --talp 10000 --teps 7.5 --tsmt 1 --tsmp 250 --tstd $i
# python test.py --gpu 0,1 --pkl oct27 --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 10000 --teps 7.5 --tsmt 1 --tsmp 250 --tstd $i
# python test.py --gpu 0,1 --pkl oct27 --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 10000 --teps 7.5 --tsmt 1 --tsmp 250 --tstd $i
# python test.py --gpu 0,1 --pkl oct27 --atk 1 --itr 2 --alp 5000 --eps 5 --smp 10 --std 0.1 --tatk 1 --titr 5 --talp 10000 --teps 7.5 --tsmt 1 --tsmp 250 --tstd $i
# python test.py --gpu 0,1 --pkl oct27 --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 10000 --teps 7.5 --tsmt 1 --tsmp 250 --tstd $i
# done


# for i in 20.0 30.0 40.0 50.0
# do 
#only atk
# python test.py --gpu 0 --pkl nov23 --tatk 1 --titr 5 --talp 10000 --teps $i
# python test.py --gpu 0 --pkl nov23 --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 10000 --teps $i
# python test.py --gpu 0 --pkl nov23 --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 10000 --teps $i
# python test.py --gpu 0 --pkl nov23 --atk 1 --itr 2 --alp 5000 --eps 5 --smp 2 --std 0.1 --tatk 1 --titr 5 --talp 10000 --teps $i
# python test.py --gpu 0 --pkl nov23 --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 10000 --teps $i

#atk +smt
# python test.py --gpu 0 --pkl nov23 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# python test.py --gpu 0 --pkl nov23 --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# python test.py --gpu 0 --pkl nov23 --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# python test.py --gpu 0 --pkl nov23 --atk 1 --itr 2 --alp 5000 --eps 5 --smp 2 --std 0.1 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# python test.py --gpu 0 --pkl nov23 --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 10000 --teps $i --tsmt 1 --tsmp 250 --tstd 0.1
# done



# for i in 10 20 50 90
for i in 10 25 50 100 150
do 
##only atk
python test.py --gpu 0 --pkl nov26_new --tatk 1 --titr 5 --talp 80000 --teps $i
python test.py --gpu 0 --pkl nov26_new --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 80000 --teps $i
python test.py --gpu 0 --pkl nov26_new --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 80000 --teps $i
python test.py --gpu 0 --pkl nov26_new --atk 1 --itr 2 --alp 5000 --eps 5 --smp 2 --std 0.1 --tatk 1 --titr 5 --talp 80000 --teps $i
python test.py --gpu 0 --pkl nov26_new --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 80000 --teps $i

# #atk +smt
python test.py --gpu 0 --pkl nov26_new --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
python test.py --gpu 0 --pkl nov26_new --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
# python test.py --gpu 0 --pkl nov26_new --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
# python test.py --gpu 0 --pkl nov26_new --atk 1 --itr 2 --alp 5000 --eps 5 --smp 2 --std 0.1 --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
# python test.py --gpu 0 --pkl nov26_new --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
done