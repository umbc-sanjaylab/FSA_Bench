#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

DATADIR="/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/small_data/"
LOGROOT="/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepPseudo/logs/small_data/IID"
MODELDIR_ROOT="/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepPseudo/models/small_data/IID"

mkdir -p "$LOGROOT" "$MODELDIR_ROOT" 
 
high_censored_datasets=("dataDIVAT1" "dataDIVAT2" "dataDIVAT3" "divorce" "hdfail" "flchain" "prostateSurvival" "FRTCS" "vlbw" "nwtco" "actg")  


############################################################ large datasets #####################################################
large_datasets=("UnempDur" "grace" "Framingham" "support2" "acath" "TRACE" "metabrick" "rdata" "rott2")
total_fold=(5 5 5 5 5 5 5 5 5)
number_of_clients=(5 5 5 5 5 5 5 5 5) 
total_batch_size=(64 32 64 128 64 32 32 32 64)


############################################################ omics datasets #####################################################
omics_datasets=("ovarian1data" "Ovarian2_data" "GE1_data" "GE2_data" "GE3_data" "GE4_data" "GE5_data" "GE6_data" "Melanoma_itraq_data" "Melanoma_nano_data" "gse4335" "AML_Bull" "dataOvarian1" "gse3143" "gse1992" "MCLcleaned" "metabric-hd" "vdv" "chop" "phpl04K8a" "Bergamaschi")
total_fold=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
total_batch_size=(16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16)
number_of_clients=(3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3)

############################################################ small datasets #####################################################
small_datasets=("veteran" "follic" "pbc" "Lung_data" "Melanoma_clinical_data" "hepatoCellular" "glioma" "wpbc" "ova" "Unemployment" "Melanoma" "micro.censure" "e1684" "zinc" "stagec" "Z243" "nki70" "mgus" "whas500" "ovarian" "breast" "pharmacoSmoking" "burn" "cost" "GBSG2" "FRTCS" "vlbw")
total_fold=(3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3)
total_batch_size=(64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64)
number_of_clients=(3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3)


# command to run only 3 datasets per job based on an argument
GROUP_ID="${1:-1}" # default group = 1
GROUP_SIZE=10

START_INDEX=$(((GROUP_ID - 1) * GROUP_SIZE))
END_INDEX=$((START_INDEX + GROUP_SIZE - 1))
TOTAL_DATASETS=${#small_datasets[@]}

# Slice into selected_datasets and preserve source indices
selected_datasets=()
selected_indices=()

for ((i = START_INDEX; i < TOTAL_DATASETS && i <= END_INDEX; i++)); do
	selected_datasets+=("${small_datasets[$i]}")
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
	number_of_client="${number_of_clients[$dataset_idx]}"

	cross_folds_run_id=()
	for ((fold_id = 0; fold_id < fold_count; fold_id++)); do
		cross_folds_run_id+=("$fold_id")
	done

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
			--model="DeepPseudo"
			--dataset="$dataset"
			--alg="fedavg"
			--lr=1e-3
			--batch_size="$batch_size"
			--epochs=250
			--n_parties="$number_of_client"
			--reg=5e-5
			--modeldir="${MODEL_DIR}/"
			--comm_round=50
			--optimizer="adam"
			--partition="iid"
			--device="cuda:0"
			--datadir="$DATADIR"
			--logdir="${LOGROOT}/${dataset}/"
			--log_file_name="$LOG_FILE_NAME"
			--dropout_p=0.05
			--patience=25
			--sample=1.0
			--epsilon=5.5
			--init_seed=5
			--pseudo_perc_interval=10
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
