i=3
fd_lambda=0.01
bd_lambda=10
ori_date=$(date '+%m%d')

reg_date=$(date '+%m%d')
#ori_date=0731
#reg_date=0731
run_num=0
stage=3

exp=5
reg=0
epoch=5
rl=0.01

if [[ $exp -eq 1 ]]
then
    group_num=10
    data=amazonBooks
    echo "Data:amazonbook && Group Num:10"
elif [[ $exp -eq 2 ]]
then
    group_num=100
    data=amazonbooks
    echo "Data:amazonbook && Group Num:100"
elif [[ $exp -eq 3 ]]
then
    group_num=10
    data=amazonElectron
    rl=0.01
    echo "Data:amazonElectron && Group Num:10"
elif [[ $exp -eq 4 ]]
then
    group_num=100
    data=amazonElectron
    rl=0.001
    echo "Data:amazonElectron && Group Num:100"
elif [[ $exp -eq 5 ]]
then
    group_num=10
    data=ml-1M
    rl=1
    echo "Data:ml-1M && Group Num:10"
elif [[ $exp -eq 6 ]]
then
    group_num=100
    data=ml-1M
    echo "Data:ml-1M && Group Num:100"
    rl=0.001
elif [[ $exp -eq 7 ]]
then
    group_num=10
    data=ali
    rl=0.001
    echo "Data:ali && Group Num:10"
elif [[ $exp -eq 8 ]]
then
    group_num=100
    data=ali
    rl=0.01
    echo "Data:ali && Group Num:100"
else
    echo "Not implemented exp number"
fi


#for i in {0..3}
#do
#    if [[ $reg -eq 0 ]]
#    then
#        python run_din.py -d ${data} -gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i} -e $epoch -b 4096  &
#    else   
#        python run_din.py -d ${data} -gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i} --reg -e $epoch -rl $rl -b 4096 &
#    fi
#done



python run_din.py -d ${data} -gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i} --reg -e $epoch -rl $rl -b 4096 --fd ${fd_lambda} --bd ${bd_lambda}
stage=4
#done
proxychains4 python run_din_4096.py -d ${data} -gn $gn ${group_num} -g $(( ${i} % 4 )) -s ${stage} -r ${i} --reg --date ${ori_date} --fd ${fd_lambda} --bd ${bd_lambda} 





