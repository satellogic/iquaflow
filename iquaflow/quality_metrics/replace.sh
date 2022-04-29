#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/X11/b‌​in
if [ -z $1 ]
then
echo "You must provide the STRING TO BE REPLACED as first argument"
exit N
else
SUBSTRING1=$1
fi
if [ -z $2 ]
then
echo "You must provide the STRING TO PUT AS REPLACEMENT as second argument"
exit N
else
SUBSTRING2=$2
fi
if [ -z $3 ]
then
echo "You must provide path of FOLDER as third argument"
FOLDER="."
else
FOLDER=$3
fi

path="${FOLDER}"/*
for file in $path
do
if ! [ -d $file ] ; then
echo $file
sed -i "s/${SUBSTRING2}/${SUBSTRING2}/g" $file
fi
done
#find $FOLDER -name "*.cfg" -type f -exec sed -i 's/${SUBSTRING2}/${SUBSTRING2}/' {} \; 
