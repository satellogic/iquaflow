

# DEFAULT UNIQUE PARAMS
cfg_folder_root="cfgs_data"
#RUN
trainid="data_test" # prefix
#PATHS
trainds="/home/Imatge/projects/satellogic/iquaflow/tests/test_datasets/AerialImageDataset"
traindsinput="/home/Imatge/projects/satellogic/iquaflow/tests/test_datasets/AerialImageDataset/train/images"
valds="/home/Imatge/projects/satellogic/iquaflow/tests/test_datasets/AerialImageDataset"
valdsinput="/home/Imatge/projects/satellogic/iquaflow/tests/test_datasets/AerialImageDataset/test/images"
outputpath="tmp-"
#HYPERPARAMS (UNIQUE)
workers="8"
data_shuffle="True"
#HYPERPARAMS (SINGLE)
epochs="200"
splits="0.8 0.2"
#HYPERPARAMS (ALT)
lr="1e-3"
weight_decay="1e-4"
momentum=0.9
soft_threshold=0.3

###################################################### BLUR (SIGMA) 
cfg_folder_param="${cfg_folder_root}/blur"
num_regs=50
modifier_params='{"sigma": np.linspace(1.0, 2.5, 50)}'
# CROPS 64
cfg_folder="${cfg_folder_param}_200crops64"
num_crops=200
input_size="64 64"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 128
cfg_folder="${cfg_folder_param}_100crops128"
num_crops=100
input_size="128 128"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 256
cfg_folder="${cfg_folder_param}_50crops256"
num_crops=50
input_size="256 256"
batch_size="16"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 512
cfg_folder="${cfg_folder_param}_20crops512"
num_crops=20
input_size="512 512"
batch_size="8"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 1024
cfg_folder="${cfg_folder_param}_10crops1024"
num_crops=10
input_size="1024 1024"
batch_size="4"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

###################################################### BLUR (RER) 
cfg_folder_param="${cfg_folder_root}/rer"
num_regs=40
modifier_params='{"rer": np.around(np.linspace(0.15, 0.55, 40),2), "dataset": "inria-aid"}'
# CROPS 64
cfg_folder="${cfg_folder_param}_200crops64"
num_crops=200
input_size="64 64"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 128
cfg_folder="${cfg_folder_param}_100crops128"
num_crops=100
input_size="128 128"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 256
cfg_folder="${cfg_folder_param}_50crops256"
num_crops=50
input_size="256 256"
batch_size="16"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 512
cfg_folder="${cfg_folder_param}_20crops512"
num_crops=20
input_size="512 512"
batch_size="8"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 1024
cfg_folder="${cfg_folder_param}_10crops1024"
num_crops=10
input_size="1024 1024"
batch_size="4"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

###################################################### SHARPNESS (smooth factor)
cfg_folder_param="${cfg_folder_root}/sharpness"
num_regs=9
modifier_params='{"sharpness": np.linspace(1.0, 10, 9)}'
# CROPS 64
cfg_folder="${cfg_folder_param}_200crops64"
num_crops=200
input_size="64 64"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 128
cfg_folder="${cfg_folder_param}_100crops128"
num_crops=100
input_size="128 128"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 256
cfg_folder="${cfg_folder_param}_50crops256"
num_crops=50
input_size="256 256"
batch_size="16"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 512
cfg_folder="${cfg_folder_param}_20crops512"
num_crops=20
input_size="512 512"
batch_size="8"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 1024
cfg_folder="${cfg_folder_param}_10crops1024"
num_crops=10
input_size="1024 1024"
batch_size="4"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

###################################################### NOISE RATIO (SNR)
cfg_folder_param="${cfg_folder_root}/snr"
num_regs=40
modifier_params='{"snr": np.linspace(15, 55, 40, dtype=int), "dataset": "inria-aid"}'
# CROPS 64
cfg_folder="${cfg_folder_param}_200crops64"
num_crops=200
input_size="64 64"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 128
cfg_folder="${cfg_folder_param}_100crops128"
num_crops=100
input_size="128 128"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 256
cfg_folder="${cfg_folder_param}_50crops256"
num_crops=50
input_size="256 256"
batch_size="16"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 512
cfg_folder="${cfg_folder_param}_20crops512"
num_crops=20
input_size="512 512"
batch_size="8"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 1024
cfg_folder="${cfg_folder_param}_10crops1024"
num_crops=10
input_size="1024 1024"
batch_size="4"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

###################################################### SCALE (GSD)
cfg_folder_param="${cfg_folder_root}/gsd"
num_regs=10
modifier_params='{"scale": np.linspace(1.0, 2.0, 10), "resol": 0.3}'
# CROPS 64
cfg_folder="${cfg_folder_param}_200crops64"
num_crops=200
input_size="64 64"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 128
cfg_folder="${cfg_folder_param}_100crops128"
num_crops=100
input_size="128 128"
batch_size="32"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 256
cfg_folder="${cfg_folder_param}_50crops256"
num_crops=50
input_size="256 256"
batch_size="16"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 512
cfg_folder="${cfg_folder_param}_20crops512"
num_crops=20
input_size="512 512"
batch_size="8"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

# CROPS 1024
cfg_folder="${cfg_folder_param}_10crops1024"
num_crops=10
input_size="1024 1024"
batch_size="4"

python3 cfg_generator.py \
--cfg_folder $cfg_folder \
--trainds $trainds \
--traindsinput $traindsinput \
--valds $valds \
--valdsinput $valdsinput \
--outputpath $outputpath \
--epochs $epochs \
--splits $splits \
--lr $lr \
--weight_decay $weight_decay \
--momentum $momentum \
--soft_threshold $soft_threshold \
--num_regs $num_regs \
--modifier_params "'${modifier_params}'" \
--num_crops $num_crops \
--input_size $input_size \
--batch_size $batch_size \
--workers $workers \
--data_shuffle $data_shuffle

