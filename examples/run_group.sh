
gi=$1
echo $gi


# python run_din_4096_group_train.py -d amazonBooks -gn 10 -g 5 -s 2 -r 0 -e 5 -rl 0 --reg -b 4096 --fd 0 --bd 0
python run_din_4096_group.py -d amazonBooks -gn 10 -g 0 -s 4 -r 0 -e 5 -rl 0 --reg -b 4096 --fd 0 --bd 0 --gi $gi



