source /Users/kai/Documents/0_uni/Abschlussarbeiten/Bachelorprojekt/rl_venv/bin/activate

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../code

training_files=(
    "run_rs.py"
    "run_pbt.py"
    # "run_a2c_pbt.py"
    # "run_dqn_pbt.py"
    # "run_ppo_pbt.py"
)

smac_training_files=(
    "run_smac_mf.py"
)

declare -a seeds=(
    "42"
    "99"
    "7"
)

declare -a algorithms=(
    "A2C"
    "DQN"
    "PPO"
)

declare -a envs=(
    "CartPole-v1"
    # "Acrobot-v1" 
    # "MountainCar-v0"
)

containsElement () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

for f in * ;
    do
    
    containsElement "$f" "${training_files[@]}"
    isTrainingFile=$?
        if ! [ $isTrainingFile == 1 ]; then
            for seed in "${seeds[@]}"
                do
                    for algo in "${algorithms[@]}"
                        do
                            for env in "${envs[@]}"
                                do
                                    python ../code/"$f" $algo $env $seed
                                done
                        done
                done
        fi
    
    containsElement "$f" "${smac_training_files[@]}"
    isSmacTrainingFile=$?
        if ! [ $isSmacTrainingFile == 1 ]; then
            for seed in "${seeds[@]}"
                do
                    for algo in "${algorithms[@]}"
                        do
                            for env in "${envs[@]}"
                                do
                                    python ../code/"$f" $algo $env $seed
                                done
                        done
                done
        fi

    done