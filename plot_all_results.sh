source /Users/kai/Documents/0_uni/Abschlussarbeiten/Bachelorprojekt/rl_venv/bin/activate

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../trained

rm -rf ../plotting_data

for d in * ;
    do
        if ! [ "$d" = 'plot_all_results.sh' ]; then
            python ../code/collect_plotting_data.py "$d"
        fi
    done

cd ../code

declare -a algorithms=("dqn" "a2c" "ppo")

for algo in "${algorithms[@]}"
do
    python plot_data.py "$algo"
done