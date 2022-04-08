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

path="${myfolder}"/*.cfg
for file in $path
do
   echo "Started processing, seed $seed " > tmp-/log_$(basename "$file" .cfg);
   python3 regressor.py --cfg_path=$file --cuda --gpus "$gpu" --seed $seed > tmp-/log_$(basename "$file").txt ; 
   # echo $(cat tmp-/log_$(basename "$file" .cfg)) | mail -s tmp-/log_$(basename "$file" .cfg) "$mymail";
done
python3 benchmark.py;
