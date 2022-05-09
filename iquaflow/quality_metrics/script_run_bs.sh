## examples
# sh script_run_bs.sh "blur" "10crops1024" "0" "parallel"
# sh script_run_bs.sh "blur" "20crops512" "0" "parallel"
# sh script_run_bs.sh "blur" "50crops256" "0" "parallel"
# sh script_run_bs.sh "blur" "100crops128" "0" "parallel"
# sh script_run_bs.sh "blur" "200crops64" "0" "parallel"

# sh script_run_bs.sh "gsd" "10crops1024" "1" "parallel"
# sh script_run_bs.sh "gsd" "20crops512" "1" "parallel"
# sh script_run_bs.sh "gsd" "50crops256" "1" "parallel"
# sh script_run_bs.sh "gsd" "100crops128" "1" "parallel"
# sh script_run_bs.sh "gsd" "200crops64" "1" "parallel"

# sh script_run_bs.sh "rer" "10crops1024" "2" "parallel"
# sh script_run_bs.sh "rer" "20crops512" "2" "parallel"
# sh script_run_bs.sh "rer" "50crops256" "2" "parallel"
# sh script_run_bs.sh "rer" "100crops128" "2" "parallel"
# sh script_run_bs.sh "rer" "200crops64" "2" "parallel"

# sh script_run_bs.sh "snr" "10crops1024" "3" "parallel"
# sh script_run_bs.sh "snr" "20crops512" "3" "parallel"
# sh script_run_bs.sh "snr" "50crops256" "3" "parallel"
# sh script_run_bs.sh "snr" "100crops128" "3" "parallel"
# sh script_run_bs.sh "snr" "200crops64" "3" "parallel"

# sh script_run_bs.sh "sharpness" "10crops1024" "4" "parallel"
# sh script_run_bs.sh "sharpness" "20crops512" "4" "parallel"
# sh script_run_bs.sh "sharpness" "50crops256" "4" "parallel"
# sh script_run_bs.sh "sharpness" "100crops128" "4" "parallel"
# sh script_run_bs.sh "sharpness" "200crops64" "4" "parallel"

if [ -z $1 ]
then
	param="sharpness"
else
	param=$1
fi

if [ -z $2 ]
then
	crops_folder="10crops1024"
else
	crops_folder=$2
fi

if [ -z $3 ]
then
	gpu="0"
else
	gpu=$3
fi

if [ -z $4 ]
then
	processtype="parallel"
else
	processtype=$4
fi


if [ $crops_folder = "10crops1024" ]; then
sh run_spec_${processtype}.sh cfgs/${param}_10crops1024/bs32 $gpu 12345 1;
sh run_spec_${processtype}.sh cfgs/${param}_10crops1024/bs16 $gpu 12345 2;
sh run_spec_${processtype}.sh cfgs/${param}_10crops1024/bs8 $gpu 12345 3;
sh run_spec_${processtype}.sh cfgs/${param}_10crops1024/bs4 $gpu 12345 5;
fi

if [ ${crops_folder} = "20crops512" ]; then
sh run_spec_${processtype}.sh cfgs/${param}_20crops512/bs64 $gpu 12345 2;
sh run_spec_${processtype}.sh cfgs/${param}_20crops512/bs32 $gpu 12345 3;
sh run_spec_${processtype}.sh cfgs/${param}_20crops512/bs16 $gpu 12345 6;
sh run_spec_${processtype}.sh cfgs/${param}_20crops512/bs8 $gpu 12345 8;
fi

if [ ${crops_folder} = "50crops256" ]; then
sh run_spec_${processtype}.sh cfgs/${param}_50crops256/bs128 $gpu 12345 2;
sh run_spec_${processtype}.sh cfgs/${param}_50crops256/bs64 $gpu 12345 5;
sh run_spec_${processtype}.sh cfgs/${param}_50crops256/bs32 $gpu 12345 7;
sh run_spec_${processtype}.sh cfgs/${param}_50crops256/bs16 $gpu 12345 9;
fi

if [ ${crops_folder} = "100crops128" ]; then
sh run_spec_${processtype}.sh cfgs/${param}_100crops128/bs256 $gpu 12345 4;
sh run_spec_${processtype}.sh cfgs/${param}_100crops128/bs128 $gpu 12345 5;
sh run_spec_${processtype}.sh cfgs/${param}_100crops128/bs64 $gpu 12345 9;
sh run_spec_${processtype}.sh cfgs/${param}_100crops128/bs32 $gpu 12345 12;
fi

if [ ${crops_folder} = "200crops64" ]; then
sh run_spec_${processtype}.sh cfgs/${param}_200crops64/bs256 $gpu 12345 11; 
sh run_spec_${processtype}.sh cfgs/${param}_200crops64/bs128 $gpu 12345 13;
sh run_spec_${processtype}.sh cfgs/${param}_200crops64/bs64 $gpu 12345 14; 
sh run_spec_${processtype}.sh cfgs/${param}_200crops64/bs32 $gpu 12345 15;
fi


