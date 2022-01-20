

exp=3
gpu=1
epoch=10
i=25
rl=1

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
else
    echo "Not implemented exp number"
fi

stage=3
python run_din.py -d ${data} -gn ${group_num} -g $gpu -s ${stage} -r ${i} -e $epoch -b 4096 -rl $rl
stage=4
python run_din.py -d ${data} -gn ${group_num} -g $gpu -s ${stage} -r ${i} -e $epoch -b 4096










