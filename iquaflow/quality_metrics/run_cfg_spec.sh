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
    mymail="maildber@gmail.com"
  else
   mymail=$3
fi
if [ -z "$4" ]
  then
    seed=$RANDOM
  else
    seed=$4
fi

python3 regressor.py --cfg_path=$file --cuda --gpus "$gpu" --seed $seed > tmp-/log_$(basename "$file" .cfg) ; 
# echo $(cat tmp-/log_$(basename "$file" .cfg)) | mail -s tmp-/log_$(basename "$file" .cfg) "$mymail";
python3 benchmark.py;
