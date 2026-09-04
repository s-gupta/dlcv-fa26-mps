python sweep.py --sweep samples --num_features 200     --sweep_min 4 --sweep_max 6400 --sweep_count 16     --out_dir runs/sweep-samples/features-200/
python sweep.py --sweep samples --num_features 400     --sweep_min 4 --sweep_max 6400 --sweep_count 16     --out_dir runs/sweep-samples/features-400/
python sweep.py --sweep samples --num_features 800     --sweep_min 4 --sweep_max 6400 --sweep_count 16     --out_dir runs/sweep-samples/features-800/
python plot_sweep.py --run_dir runs/sweep-samples/features-200/ --run_dir runs/sweep-samples/features-400/ --run_dir runs/sweep-samples/features-800/ --out sweep-samples.png --xlabel "number of training examples N"

python sweep.py --sweep features --num_train 200     --sweep_min 4 --sweep_max 8192 --sweep_count 16     --out_dir runs/sweep-features/samples-200/
python sweep.py --sweep features --num_train 400     --sweep_min 4 --sweep_max 8192 --sweep_count 16     --out_dir runs/sweep-features/samples-400/
python sweep.py --sweep features --num_train 800     --sweep_min 4 --sweep_max 8192 --sweep_count 16     --out_dir runs/sweep-features/samples-800/
python plot_sweep.py --run_dir runs/sweep-features/samples-200/ --run_dir runs/sweep-features/samples-400/ --run_dir runs/sweep-features/samples-800/ --out sweep-features.png --xlabel "number of features W"

