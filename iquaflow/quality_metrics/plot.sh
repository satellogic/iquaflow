for file in *.cfg
do
   python3 regressor.py --cfg_path=$file --plot_only
done

for file in */*.cfg
do
   python3 regressor.py --cfg_path=$file --plot_only
done

for file in */*/*.cfg
do
   python3 regressor.py --cfg_path=$file --plot_only
done
