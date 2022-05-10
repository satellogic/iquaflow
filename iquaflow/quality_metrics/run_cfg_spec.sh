if [ -z "$1" ]
  then
    echo "Must provide path of .cfg file"
    exit N
  else
   file=$1
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

python3 regressor.py --cfg_path=$file --cuda --gpus "$gpu" --seed $seed --debug;

