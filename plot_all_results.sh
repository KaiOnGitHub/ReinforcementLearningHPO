source /Users/kai/Documents/0_uni/Abschlussarbeiten/Bachelorprojekt/rl_venv/bin/activate

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../trained

for d in * ;
    do
        if ! [ "$d" = 'plot_all_results.sh' ]; then
            python ../code/plot_results.py "$d"
        fi
    done