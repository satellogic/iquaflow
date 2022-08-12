if [ -z $1 ]
then
	gpu="0"
else
	gpu=$1
fi

if [ -z $2 ]
then
	processtype="sequential"
else
	processtype=$2
fi


sh run_spec_${processtype}.sh cfgs_multihead $gpu 12345 1;

