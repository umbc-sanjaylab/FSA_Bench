#!/usr/bin/env bash
set -euo pipefail 
export CUDA_VISIBLE_DEVICES=0

DATADIR="/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/federated_data/"
LOGROOT="/home/il66977@ad.umbc.edu/Benchmarking_FSA/Fed_CoxPH/logs/federated_data/federated"
MODELDIR_ROOT="/home/il66977@ad.umbc.edu/Benchmarking_FSA/Fed_CoxPH/models/federated_data/federated"

mkdir -p "$LOGROOT" "$MODELDIR_ROOT" 
  
################################################### distributed dataset #########################################################
federated_datasets=("seer" "Aids2" "LeukSurv" "Pbc3" "d.oropha.rec" "cgd" "uis" "scania" "Dialysis")
total_fold=(5 5 3 3 3 3 5 5 3)                   # Number of folds
total_center=(3 4 24 6 6 4 2 5 67)               # Number of clients
total_batch_size=(128 64 32 32 32 32 64 32 32)   # Federated learning batch sizes  

# Optional: set DRY_RUN=1 to only print commands
DRY_RUN="${DRY_RUN:-0}"

# NEW: run only 2 datasets per job based on an argument
GROUP_ID="${1:-1}"
GROUP_SIZE=10

# Validate GROUP_ID is a positive integer
if ! [[ "$GROUP_ID" =~ ^[1-9][0-9]*$ ]]; then
	echo "ERROR: GROUP_ID must be a positive integer. Example: bash $0 1"
	exit 1
fi

START_INDEX=$(((GROUP_ID - 1) * GROUP_SIZE))
END_INDEX=$((START_INDEX + GROUP_SIZE - 1))

TOTAL_DATASETS=${#federated_datasets[@]}
if ((START_INDEX >= TOTAL_DATASETS)); then
	echo "ERROR: GROUP_ID=$GROUP_ID selects datasets [$START_INDEX..$END_INDEX], but only $TOTAL_DATASETS datasets exist."
	echo "TIP: max GROUP_ID is $(((TOTAL_DATASETS + GROUP_SIZE - 1) / GROUP_SIZE))"
	exit 1
fi

# Slice into selected_datasets
selected_datasets=()
selected_indices=()
for ((i = START_INDEX; i < TOTAL_DATASETS && i <= END_INDEX; i++)); do
	selected_datasets+=("${federated_datasets[$i]}")
	selected_indices+=("$i")
done

echo "===================================================="
echo "GROUP_ID=$GROUP_ID (size=$GROUP_SIZE) => indices [$START_INDEX..$END_INDEX]"
echo "Selected datasets (${#selected_datasets[@]}): ${selected_datasets[*]}"
echo "===================================================="

echo "--- Checking GPUs with nvidia-smi ---"
nvidia-smi || true

# Main loops (use selected_datasets instead of high_dimensional_datasets)
for selection_idx in "${!selected_datasets[@]}"; do

	dataset="${selected_datasets[$selection_idx]}"
	dataset_idx="${selected_indices[$selection_idx]}"

	batch_size="${total_batch_size[$dataset_idx]}"
	fold_count="${total_fold[$dataset_idx]}"
	no_of_client="${total_center[$dataset_idx]}"

	if ((fold_count < 1)); then
		echo "ERROR: total_fold for dataset=$dataset must be >= 1, got $fold_count."
		exit 1
	fi

	cross_folds_run_id=()
	for ((fold_id = 0; fold_id < fold_count; fold_id++)); do
		cross_folds_run_id+=("$fold_id")
	done

	for five_fold_run_id in "${cross_folds_run_id[@]}"; do

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
			--model="CoxPH"
			--dataset="$dataset"
			--alg="fedavg"
			--lr=1e-3
			--batch_size="$batch_size"
			--epochs=250
			--n_parties="$no_of_client"
			--reg=5e-5
			--modeldir="${MODEL_DIR}/"
			--comm_round=50
			--optimizer="adam"
			--partition="federated"
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
