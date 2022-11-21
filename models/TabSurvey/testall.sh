#!/bin/bash

N_TRIALS=2
EPOCHS=3

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# "LinearModel" "KNN" "DecisionTree" "RandomForest"
# "XGBoost" "CatBoost" "LightGBM"
# "MLP" "TabNet" "VIME"
# MODELS=( "LinearModel" "KNN" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" "TabNet" "VIME")

declare -A MODELS
MODELS=( ["LinearModel"]=$SKLEARN_ENV
         ["KNN"]=$SKLEARN_ENV
         # ["SVM"]=$SKLEARN_ENV
         ["DecisionTree"]=$SKLEARN_ENV
         ["RandomForest"]=$SKLEARN_ENV
         ["XGBoost"]=$GBDT_ENV
         ["CatBoost"]=$GBDT_ENV
         ["LightGBM"]=$GBDT_ENV
         ["MLP"]=$TORCH_ENV
         ["TabNet"]=$TORCH_ENV
         ["VIME"]=$TORCH_ENV
         ["TabTransformer"]=$TORCH_ENV
         ["ModelTree"]=$GBDT_ENV
         ["NODE"]=$TORCH_ENV
         ["DeepGBM"]=$TORCH_ENV
         ["RLN"]=$KERAS_ENV
         ["DNFNet"]=$KERAS_ENV
         ["STG"]=$TORCH_ENV
         ["NAM"]=$TORCH_ENV
         ["DeepFM"]=$TORCH_ENV
         ["SAINT"]=$TORCH_ENV
         ["DANet"]=$TORCH_ENV
          )

CONFIGS=( "config/adult.yml"
          "config/covertype.yml"
          "config/california_housing.yml"
          "config/higgs.yml"
          )

# conda init bash
eval "$(conda shell.bash hook)"

for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS[$model]}"

    conda activate "${MODELS[$model]}"

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS

    conda deactivate

  done

done
