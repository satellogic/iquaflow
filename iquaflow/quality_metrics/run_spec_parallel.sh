if [ -z "$1" ]
  then
    echo "Must provide folder of .cfg files"
    exit N
  else
   myfolder=$1
fi
if [ -z "$2" ]
  then
    gpu="0"
  else
   gpu=$2
fi

if [ -z "$3" ]
  then
    seed=12345 #$RANDOM
  else
    seed=$3
fi

if [ -z "$4" ]
  then
    BS=4
  else
   BS=$4
fi


path="${myfolder}"/*.cfg
NP=0
for file in $path
do
   python3 regressor.py --cfg_path=$file --cuda --gpus "$gpu" --seed $seed --debug &
   NP=$((NP + 1))
   if [ $NP -ge $BS ]; then
       wait
       echo "New batch of processes"
       NP=0
   fi
done
