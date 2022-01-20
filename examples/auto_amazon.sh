i=27  # 4 的倍数
proxychains4 bash auto_run_multi.sh 1 0 0 &    
proxychains4 bash auto_run_multi.sh 21 10 0 &
proxychains4 bash auto_run_multi.sh 25 10 10 &
proxychains4 bash auto_run_multi.sh 26 0 0.02 &
wait
for fd_lambda in {0}
do
    for bd_lambda in {0.03,0.04,0.05,0.06,0.07,0.08,0.09}
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


    

