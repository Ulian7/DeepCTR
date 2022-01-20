echo "第0个参数: $0"
echo "第1个参数: $1"
echo "第2个参数: $2"
echo "第3个参数: $3"
i=$1
cd_lambda=$2
py_exp=$3
train_or_test=${4:-0}
fd_lambda=0
bd_lambda=0
gpu_id=4
# ori_date='1129'
ori_date=$(date '+%m%d')

reg_date=$(date '+%m%d')

run_num=0
stage=3

exp=5
reg=0
epoch=10
rl=0.01

if [[ $exp -eq 1 ]]
then
    group_num=10
    data=amazonBooks
    echo "Data:amazonbook && Group Num:10"
elif [[ $exp -eq 2 ]]
then
    group_num=100
    data=amazonBooks
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


# echo "python run_din.py -d ${data} -gn ${group_num} -g $gpu_id -s ${stage} -r ${i} -e $epoch -rl $rl -b 4096 --fd ${fd_lambda} --bd ${bd_lambda} --exp $py_exp"

if [[ $train_or_test -eq 0 ]]
then
    echo " Start Training & Testing"
fi


if [[ $train_or_test -eq 0 || $train_or_test -eq 1 ]]
then
    echo "Start Training"
    python run_din.py -d ${data} -gn ${group_num} -g $gpu_id -s ${stage} -r ${i} -e $epoch -rl $rl -b 4096 --fd ${fd_lambda} --bd ${bd_lambda} --cd ${cd_lambda} --exp $py_exp
fi

stage=4
# echo "python run_din_4096.py -d ${data} -gn $gn ${group_num} -g $gpu_id -s ${stage} -r ${i} --date ${ori_date} --fd ${fd_lambda} --bd ${bd_lambda} --exp $py_exp"

if [[ $train_or_test -eq 0 || $train_or_test -eq 2 ]]
then
    echo "Start Testing"
    python run_din_4096.py -d ${data} -gn $gn ${group_num} -g $gpu_id -s ${stage} -r ${i} --date ${ori_date} --fd ${fd_lambda} --bd ${bd_lambda} --cd ${cd_lambda} --exp $py_exp
fi





