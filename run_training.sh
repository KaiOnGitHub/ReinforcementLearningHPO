source /Users/kai/Documents/0_uni/Abschlussarbeiten/Bachelorprojekt/rl_venv/bin/activate

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../code

training_files=(
    "run_a2c_pbt.py"
    "run_a2c_rs.py"
    "run_dqn_pbt.py"
    "run_dqn_rs.py"
    "run_ppo_pbt.py"
    "run_ppo_rs.py"
)

smac_training_files=(
     #"smac_mf_a2c_Acrobot.py"
     "smac_mf_a2c_CartPole.py"
     #"smac_mf_a2c_MountainCar.py"
     #"smac_mf_dqn_Acrobot.py"
     "smac_mf_dqn_CartPole.py"
     #"smac_mf_dqn_MountainCar.py"
     #"smac_mf_ppo_Acrobot.py"
     "smac_mf_ppo_CartPole.py"
     #"smac_mf_ppo_Mountaincar.py"
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
            # python ../code/"$f" "Acrobot-v1" 
            # python ../code/"$f" "MountainCar-v0"
            python ../code/"$f" "CartPole-v1"
        fi
    
    containsElement "$f" "${smac_training_files[@]}"
    isSmacTrainingFile=$?
        if ! [ $isSmacTrainingFile == 1 ]; then
            python ../code/"$f"
        fi

    done