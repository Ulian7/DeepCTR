ori_date=$(date '+%m%d')
reg_date=$(date '+%m%d')
ori_date=0722
i=1
fd_lambda=0
bd_lambda=0.01
#reg_date=0720


stage=4

exp=5
reg=0

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
else
    echo "Not implemented exp number"
fi

#for i in {0..3}
#do
#    if [[ $reg -eq 0 ]]
#    then
#        python run_din.py -d ${data} -gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i} --date ${ori_date} &
#    else
#        python run_din.py -d ${data} -gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i} --reg --date $reg_date &
#    fi
#done

proxychains4 python run_din_4096.py -d ${data} -gn $gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i} --reg --date ${ori_date} --fd ${fd_lambda} --bd ${bd_lambda}






