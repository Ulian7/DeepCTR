i=300  # 4 的倍数
for fd_lambda in 1.0
do
    for bd_lambda in {0,0.01,0.1,1}
    do
    {
    proxychains4 bash auto_run_multi.sh $i $fd_lambda $bd_lambda
    }&
 ((i++))
if [[ $((i % 4)) == 0 ]]
    then
    wait
fi
done
done


    

