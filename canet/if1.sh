


# # for i in 5.0 7.5 10.0 15.0
# #for i in 20.0 30.0 40.0 50.0
# for i in 100 500 1000
# do 
# #only atk
# # python test.py --gpu 1 --pkl nov23 --tatk 1 --titr 5 --talp 10000 --teps $i
# # python test.py --gpu 1 --pkl nov23 --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 10000 --teps $i
# # python test.py --gpu 1 --pkl nov23 --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 10000 --teps $i
# # python test.py --gpu 1 --pkl nov23 --atk 1 --itr 2 --alp 5000 --eps 5 --smp 2 --std 0.1 --tatk 1 --titr 5 --talp 10000 --teps $i
# # python test.py --gpu 1 --pkl nov23 --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 10000 --teps $i

# #atk +smt
# python test.py --gpu 1 --pkl nov24 --tatk 1 --titr 5 --talp 10000 --teps 5 --tsmt 1 --tsmp $i --tstd 0.1
# python test.py --gpu 1 --pkl nov24 --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 10000 --teps 5 --tsmt 1 --tsmp $i --tstd 0.1
# python test.py --gpu 1 --pkl nov24 --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 10000 --teps 5 --tsmt 1 --tsmp $i --tstd 0.1
# python test.py --gpu 1 --pkl nov24 --atk 1 --itr 2 --alp 5000 --eps 5 --smp 2 --std 0.1 --tatk 1 --titr 5 --talp 10000 --teps 5 --tsmt 1 --tsmp $i --tstd 0.1
# python test.py --gpu 1 --pkl nov24 --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 10000 --teps 5 --tsmt 1 --tsmp $i --tstd 0.1
# done




for i in 10 25 50 100 150
do 
# ##only atk
# python test.py --gpu 1 --pkl nov26_new --tatk 1 --titr 5 --talp 80000 --teps $i
# python test.py --gpu 1 --pkl nov26_new --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 80000 --teps $i
# python test.py --gpu 1 --pkl nov26_new --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 80000 --teps $i
# python test.py --gpu 1 --pkl nov26_new --atk 1 --itr 2 --alp 5000 --eps 5 --smp 2 --std 0.1 --tatk 1 --titr 5 --talp 80000 --teps $i
# python test.py --gpu 1 --pkl nov26_new --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 80000 --teps $i

# # #atk +smt
# python test.py --gpu 1 --pkl nov26_new --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
# python test.py --gpu 1 --pkl nov26_new --jcb 1 --gma 20 --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
python test.py --gpu 1 --pkl nov26_new --atk 1 --itr 2 --alp 5000 --eps 5 --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
python test.py --gpu 1 --pkl nov26_new --atk 1 --itr 2 --alp 5000 --eps 5 --smp 2 --std 0.1 --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
python test.py --gpu 1 --pkl nov26_new --smt 1 --smp 10 --std 0.1 --stp 1 --tatk 1 --titr 5 --talp 80000 --teps $i --tsmt 1 --tsmp 100 --tstd 0.1
done