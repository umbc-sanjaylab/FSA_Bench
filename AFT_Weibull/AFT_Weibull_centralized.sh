#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
 
DATADIR="/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/federated_data"
LOGROOT="/home/il66977@ad.umbc.edu/Benchmarking_FSA/AFT_Weibull/logs/federated_data/centralized"
MODELDIR_ROOT="/home/il66977@ad.umbc.edu/Benchmarking_FSA/AFT_Weibull/models/federated_data/centralized"

mkdir -p "$LOGROOT" "$MODELDIR_ROOT" 

##################################################### Distributed datasets ##################################################### 
federated_datasets=("TCGA" "Aids2" "LeukSurv" "Pbc3" "d.oropha.rec" "cgd" "uis" "scania" "seer" "Dialysis")
total_fold=(5 5 3 3 3 3 5 5 5 3)            # Number of folds
total_center=(1 1 1 1 1 1 1 1 1 1)        # Number of clients
total_batch_size=(32 32 4 4 4 8 16 8 64 32) # Federated learning batch sizes  
 
# command to run only 3 datasets per job based on an argument
GROUP_ID="${1:-1}" # default group = 1
GROUP_SIZE=10

START_INDEX=$(((GROUP_ID - 1) * GROUP_SIZE))
END_INDEX=$((START_INDEX + GROUP_SIZE - 1))
TOTAL_DATASETS=${#federated_datasets[@]}

# Slice into selected_datasets and preserve source indices
selected_datasets=()
selected_indices=()

for ((i = START_INDEX; i < TOTAL_DATASETS && i <= END_INDEX; i++)); do
	selected_datasets+=("${federated_datasets[$i]}")
	selected_indices+=("$i")
done

# Optional: set DRY_RUN=1 to only print commands
DRY_RUN="${DRY_RUN:-0}"

echo "--- Checking GPUs with nvidia-smi ---"
nvidia-smi || true

for selection_idx in "${!selected_datasets[@]}"; do
	dataset="${selected_datasets[$selection_idx]}"
	dataset_idx="${selected_indices[$selection_idx]}"

	batch_size="${total_batch_size[$dataset_idx]}"
	fold_count="${total_fold[$dataset_idx]}"

	cross_folds_run_id=()
	for ((fold_id = 0; fold_id < fold_count; fold_id++)); do
		cross_folds_run_id+=("$fold_id")
	done

	echo "${cross_folds_run_id[@]}"

	for five_fold_run_id in "${cross_folds_run_id[@]}"; do
		# Structured dirs per dataset + hyperparams

		RUN_DIR="${LOGROOT}/${dataset}/"
		MODEL_DIR="${MODELDIR_ROOT}/${dataset}/"

		mkdir -p "$RUN_DIR" "$MODEL_DIR"

		TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
		RUN_ID="${TIMESTAMP}_five_fold_run_id${five_fold_run_id}"

		LOGFILE="${RUN_DIR}/${RUN_ID}.log"
		LOG_FILE_NAME="${RUN_DIR}/${RUN_ID}" # without .log (if your python expects this)

		echo "===================================================="
		echo "Run: dataset=$dataset five_fold_run_id=$five_fold_run_id"
		echo "Logs:  $RUN_DIR"
		echo "Model: $MODEL_DIR"
		echo "File:  $LOGFILE"
		echo "===================================================="

		CMD=(python experiments.py
			--model="AFT_Weibull"
			--dataset="$dataset"
			--alg="all_in"
			--lr=1e-3
			--batch_size="$batch_size"
			--epochs=250
			--n_parties=1
			--reg=5e-5
			--modeldir="${MODEL_DIR}/"
			--comm_round=1
			--optimizer="adam"
			--partition="centralized"
			--device="cuda:0"
			--datadir="$DATADIR"
			--logdir="${LOGROOT}/${dataset}/"
			--log_file_name="$LOG_FILE_NAME"
			--dropout_p=0.05
			--patience=25
			--eval_every=5
			--sample=1.0
			--epsilon=5.5
			--init_seed=5
			--run_id="$five_fold_run_id"
		)

		if [[ "$DRY_RUN" == "1" ]]; then
			printf 'DRY_RUN: %q ' "${CMD[@]}"
			echo
			continue
		fi

		(
			exec > >(tee -a "$LOGFILE") 2>&1
			echo "--- Date: $(date) ---"
			echo "--- CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ---"
			nvidia-smi || true
			echo "--- Command ---"
			printf '%q ' "${CMD[@]}"
			echo
			echo "--------------"

			"${CMD[@]}"
		)
	done
done
