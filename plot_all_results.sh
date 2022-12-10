source /Users/kai/Documents/0_uni/Abschlussarbeiten/Bachelorprojekt/rl_venv/bin/activate

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../trained_final

rm -rf ../plotting_data

declare -a seed_dirs=( "seed_42" "seed_99" "seed_7")

for seed_dir in "${seed_dirs[@]}"
do
    cd "$seed_dir"
    for d in *
        do
        python ../../code/collect_plotting_data.py "$d" "$seed_dir"
        done
    cd ..
done

cd ../code

declare -a algorithms=("dqn" "a2c" "ppo")

for algo in "${algorithms[@]}"
do
    python plot_data.py "$algo"
done