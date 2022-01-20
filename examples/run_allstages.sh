exp=5
gpu=2
epoch=5

if [[ $exp -eq 1 ]]
then
    group_num=10
    data=amazonbooks
elif [[ $exp -eq 2 ]]
then
    group_num=100
    data=amazonbooks
elif [[ $exp -eq 3 ]]
then
    group_num=10
    data=amazonElectron
elif [[ $exp -eq 4 ]]
then
    group_num=100
    data=amazonElectron
elif [[ $exp -eq 5 ]]
then
    group_num=10
    data=ml-1M
elif [[ $exp -eq 6 ]]
then
    group_num=100
    data=ml-1M
elif [[ $exp -eq 7 ]]
then
    group_num=10
    data=ali
elif [[ $exp -eq 8 ]]
then
    group_num=100
    data=ali
else
    echo "Not implemented exp number"
fi

python run_din_4096.py -gn $group_num -g $gpu -s 1 -r 1 -d $data -e $epoch
python run_din_4096.py -gn $group_num -g $gpu -s 2 -r 1 -d $data -e $epoch

# stage=3
# for reg in 0 1
# do
#     for i in {0..3}
#     do
#         if [[ $reg -eq 0 ]]
#         then
#             python run_din.py -d ${data} -gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i}  &
#         else
#             python run_din.py -d ${data} -gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i} --reg  &
#         fi
#     done
# done
# i=20
# gpu=1
# python run_din.py -d ${data} -gn ${group_num} -g $gpu -s ${stage} -r ${i} -e $epoch  &
# 
# gpu=2
# python run_din.py -d ${data} -gn ${group_num} -g $gpu -s ${stage} -r ${i} --reg -e $epoch  &









