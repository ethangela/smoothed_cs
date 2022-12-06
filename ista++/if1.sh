for i in 15
do 
# ##only atk
python test.py --gpu 1 --pkl nov28_500 --tatk 1 --titr 10 --talp 500 --teps $i
python test.py --gpu 1 --pkl nov28_500 --atk 1 --itr 10 --alp 200 --eps 0.01 --tatk 1 --titr 10 --talp 500 --teps $i
python test.py --gpu 1 --pkl nov28_500 --jcb 1 --gma 20 --tatk 1 --titr 10 --talp 500 --teps $i
python test.py --gpu 1 --pkl nov28_500 --atk 1 --itr 10 --alp 200 --eps 0.01 --smp 10 --std 0.05 --tatk 1 --titr 10 --talp 500 --teps $i
python test.py --gpu 1 --pkl nov28_500 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 10 --talp 500 --teps $i

# # #atk +smt
python test.py --gpu 1 --pkl nov28_500 --tatk 1 --titr 10 --talp 500 --teps $i --tsmt 1 --tsmp 100 --tstd 0.04
python test.py --gpu 1 --pkl nov28_500 --atk 1 --itr 10 --alp 200 --eps 0.01 --tatk 1 --titr 10 --talp 500 --teps $i --tsmt 1 --tsmp 100 --tstd 0.04
python test.py --gpu 1 --pkl nov28_500 --jcb 1 --gma 20 --tatk 1 --titr 10 --talp 500 --teps $i --tsmt 1 --tsmp 100 --tstd 0.04
python test.py --gpu 1 --pkl nov28_500 --atk 1 --itr 10 --alp 200 --eps 0.01 --smp 10 --std 0.05 --tatk 1 --titr 10 --talp 500 --teps $i --tsmt 1 --tsmp 100 --tstd 0.04
python test.py --gpu 1 --pkl nov28_500 --smt 1 --smp 10 --std 0.05 --stp 6 --tatk 1 --titr 10 --talp 500 --teps $i --tsmt 1 --tsmp 100 --tstd 0.04


done