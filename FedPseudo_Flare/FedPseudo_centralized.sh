#!/usr/bin/env bash  
export CUDA_VISIBLE_DEVICES=1

# Limit CPU usage
CPU_CORES="${CPU_CORES:-0-3}"     # use only logical CPU cores 0,1,2,3
CPU_THREADS="${CPU_THREADS:-1}"  # number of CPU threads allowed

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export BLIS_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"

# PyTorch-specific thread control
export TORCH_NUM_THREADS="$CPU_THREADS"
set -euo pipefail


DATADIR="/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/federated_data"
LOGROOT="/home/il66977@ad.umbc.edu/Benchmarking_FSA/FedPseudo_Flare/logs/federated_data/centralized"
MODELDIR_ROOT="/home/il66977@ad.umbc.edu/Benchmarking_FSA/FedPseudo_Flare/models/federated_data/centralized"

mkdir -p "$LOGROOT" "$MODELDIR_ROOT"
models=("FedPDNN" "FedPLSTM" "FedPAttn") 

############################################################ omics datasets #####################################################
omics_datasets=("ovarian1data" "Ovarian2_data" "GE1_data" "GE2_data" "GE3_data" "GE4_data" "GE5_data" "GE6_data" "Melanoma_itraq_data" "Melanoma_nano_data" "gse4335" "AML_Bull" "dataOvarian1" "gse3143" "gse1992" "MCLcleaned" "metabric-hd" "vdv" "chop" "phpl04K8a" "Bergamaschi")
total_fold=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
total_batch_size=(16 8 8 16 8 16 8 32 8 8 8 16 64 16 8 16 64 8 32 32 8)   

############################################################ large datasets #####################################################
large_datasets=("UnempDur" "grace" "Framingham" "support2" "acath" "TRACE" "metabrick" "rdata" "rott2")
total_fold=(5 5 5 5 5 5 5 5 5)
total_batch_size=(64 64 64 64 64 64 64 64 64)   
 

############################################################ small datasets #####################################################
small_datasets=("veteran" "follic" "pbc" "Lung_data" "hepatoCellular" "wpbc" "ova" "Unemployment" "Melanoma" "micro.censure" "e1684" "zinc" "stagec"  "nki70" "mgus" "whas500" "pharmacoSmoking" "burn" "cost" "GBSG2" "FRTCS" "vlbw" "glioma" "breast" "ovarian" "Melanoma_clinical_data")
total_fold=(3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  3 3 3 3 3 3)
total_batch_size=(64 64 64 64 64 64 64 64 64 64 64  64 64 64 64 64 64 64 64 64 64 64 64 64 64 64)
total_center=(3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  3 3 3 3 3 3 3 2 2 2 2)


####################################################### high censored datasets ###################################################
high_censored_datasets=("dataDIVAT1" "dataDIVAT2" "dataDIVAT3" "divorce" "hdfail" "flchain" "prostateSurvival" "nwtco" "actg")  
total_fold=(5 5 3 5 5 5 3 3 3)
total_batch_size=(64 64 64 64 256 64 128 64 64)  
total_center=(5 5 5 5 5 5 5 5 5)   

####################################################### federated datasets ###################################################
federated_datasets=("TCGA" "Aids2" "LeukSurv" "Pbc3" "d.oropha.rec" "cgd" "uis" "scania" "seer" "Dialysis")
total_fold=(5 5 3 3 3 3 5 5 5 3)                   # Number of folds
total_center=(1 1 1 1 1 1 1 1 1 1)               # Number of clients
total_batch_size=(128 64 32 32 32 32 64 32 128 32) # Federated learning batch sizes 

# Optional: set DRY_RUN=1 to only print commands
DRY_RUN="${DRY_RUN:-0}"

# command to run only 3 datasets per job based on an argument
GROUP_ID="${1:-1}" # default group = 1
GROUP_SIZE=30

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

	for model in "${models[@]}"; do
		for five_fold_run_id in "${cross_folds_run_id[@]}"; do

			# Structured dirs per dataset + hyperparams
			RUN_DIR="${LOGROOT}/${dataset}/"
			MODEL_DIR="${MODELDIR_ROOT}/${dataset}/"

			mkdir -p "$RUN_DIR" "$MODEL_DIR"

			echo "===================================================="
			echo "Run: dataset=$dataset model=$model five_fold_run_id=$five_fold_run_id"
			echo "Logs:  $RUN_DIR"
			echo "Model: $MODEL_DIR"
			echo "===================================================="

			CMD=(taskset -c "$CPU_CORES" python job_fedpseudo.py
				--model="$model"
				--dataset="$dataset"
				--alg="all_in"
				--lr=1e-3
				--batch_size="$batch_size"
				--epochs=250
				--n_parties=1
				--reg=5e-5
				--modeldir="${MODEL_DIR}/"
				--comm_round=2
				--optimizer="adam"
				--partition="centralized"
				--device="cuda:0"
				--datadir="$DATADIR"
				--logdir="$RUN_DIR"
				--dropout_p=0.05
				--patience=25
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
done
