i=0  # 4 的倍数
wait
for cd_lambda in {100,50,25,10,5,2,1,0.5,0.2,0.1,0.05,0.02,0.01}
do
{
bash auto_run_multi.sh $i $cd_lambda
}&
 ((i++))
# if [[ $((i % 4)) == 0 ]]
#     then
#     wait
# fi
done


    

