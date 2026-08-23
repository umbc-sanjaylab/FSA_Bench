import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# Load results
# ============================================================

RESULT_FILE = "data/TCGA.csv"

df = pd.read_csv(RESULT_FILE)


# ============================================================
# Models
# ============================================================

PROPOSED_MODELS = [
    "FedPDNN",
    "FedPLSTM",
    "FedPAttn"
]

BASELINE_MODELS = [
    "Fed_KM",
    "AFT_Weibull",
    "CoxPH",
    "DeepPseudo",
    "DeepHit",
    "LinearPH",
    "NNPH",
    "NNnPH"
]

PARTITIONS = [
    "centralized",
    "federated"
]


# ============================================================
# Helper function
# ============================================================

def run_wilcoxon(x, y):
    """
    Paired two-sided Wilcoxon signed-rank test.

    Removes pairs where either model has NaN.
    """

    paired = pd.concat(
        [x.rename("x"), y.rename("y")],
        axis=1
    ).dropna()

    x = paired["x"].values
    y = paired["y"].values

    n = len(x)

    if n < 2:
        return np.nan, np.nan, n

    # If all paired differences are zero,
    # Wilcoxon cannot be computed normally.
    if np.allclose(x - y, 0):
        return 0.0, 1.0, n

    try:
        statistic, p_value = wilcoxon(
            x,
            y,
            alternative="greater",
            zero_method="wilcox",
            method="auto"
        )

        return statistic, p_value, n

    except ValueError:
        return np.nan, np.nan, n


# ============================================================
# 1. DATASET-LEVEL WILCOXON
# ============================================================

dataset_results = []


for partition in PARTITIONS:

    partition_df = df[
        df["partition"] == partition
    ].copy()

    # --------------------------------------------------------
    # Average folds within each dataset
    # --------------------------------------------------------

    dataset_mean_df = (
        partition_df
        .groupby("dataset")[PROPOSED_MODELS + BASELINE_MODELS]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------------
    # Compare each proposed model against every baseline
    # --------------------------------------------------------

    for proposed_model in PROPOSED_MODELS:

        for baseline_model in BASELINE_MODELS:

            statistic, p_value, n = run_wilcoxon(
                dataset_mean_df[proposed_model],
                dataset_mean_df[baseline_model]
            )

            if n > 0:
                mean_proposed = (
                    dataset_mean_df[proposed_model]
                    .dropna()
                    .mean()
                )

                mean_baseline = (
                    dataset_mean_df[baseline_model]
                    .dropna()
                    .mean()
                )
            else:
                mean_proposed = np.nan
                mean_baseline = np.nan

            dataset_results.append({
                "partition": partition,
                "level": "dataset",
                "proposed_model": proposed_model,
                "baseline_model": baseline_model,
                "n": n,
                "proposed_mean": mean_proposed,
                "baseline_mean": mean_baseline,
                "mean_difference": (
                    mean_proposed - mean_baseline
                    if n > 0 else np.nan
                ),
                "statistic": statistic,
                "p_value": p_value
            })


dataset_results = pd.DataFrame(dataset_results)


# ============================================================
# Holm-Bonferroni correction
#
# Correction is performed separately for:
#     centralized + proposed model
#     federated + proposed model
#
# Therefore each proposed model has its own family of
# baseline comparisons within each partition.
# ============================================================

dataset_results["p_holm"] = np.nan
dataset_results["significant"] = False


for partition in PARTITIONS:

    for proposed_model in PROPOSED_MODELS:

        mask = (
            (dataset_results["partition"] == partition) &
            (dataset_results["proposed_model"] == proposed_model)
        )

        valid = (
            mask &
            dataset_results["p_value"].notna()
        )

        if valid.sum() > 0:

            corrected = multipletests(
                dataset_results.loc[valid, "p_value"],
                alpha=0.05,
                method="holm"
            )

            dataset_results.loc[valid, "p_holm"] = corrected[1]
            dataset_results.loc[valid, "significant"] = corrected[0]


# ============================================================
# Save dataset-level results
# ============================================================

dataset_results.to_csv(
    "wilcoxon_dataset_level_results.csv",
    index=False
)


# ============================================================
# Print dataset-level results
# ============================================================

print("\n" + "=" * 100)
print("DATASET-LEVEL WILCOXON SIGNED-RANK TEST")
print("=" * 100)

for partition in PARTITIONS:

    print(f"\n\nPARTITION: {partition.upper()}")

    for proposed_model in PROPOSED_MODELS:

        print(f"\n--- {proposed_model} ---")

        temp = dataset_results[
            (dataset_results["partition"] == partition) &
            (dataset_results["proposed_model"] == proposed_model)
        ].copy()

        print(
            temp[
                [
                    "baseline_model",
                    "n",
                    "proposed_mean",
                    "baseline_mean",
                    "mean_difference",
                    "statistic",
                    "p_value",
                    "p_holm",
                    "significant"
                ]
            ].to_string(index=False)
        )


# ============================================================
# 2. FOLD-LEVEL WILCOXON
# ============================================================

fold_results = []


for partition in PARTITIONS:

    partition_df = df[
        df["partition"] == partition
    ].copy()

    for proposed_model in PROPOSED_MODELS:

        for baseline_model in BASELINE_MODELS:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Pair observations using dataset + fold_id.
            #
            # This ensures that FedPDNN fold 0 of TCGA is
            # compared with the corresponding baseline fold 0
            # of TCGA.
            # ------------------------------------------------

            paired = partition_df[
                [
                    "dataset",
                    "fold_id",
                    proposed_model,
                    baseline_model
                ]
            ].dropna()

            x = paired[proposed_model]
            y = paired[baseline_model]

            statistic, p_value, n = run_wilcoxon(x, y)

            if n > 0:
                mean_proposed = x.mean()
                mean_baseline = y.mean()
            else:
                mean_proposed = np.nan
                mean_baseline = np.nan

            fold_results.append({
                "partition": partition,
                "level": "fold",
                "proposed_model": proposed_model,
                "baseline_model": baseline_model,
                "n": n,
                "proposed_mean": mean_proposed,
                "baseline_mean": mean_baseline,
                "mean_difference": (
                    mean_proposed - mean_baseline
                    if n > 0 else np.nan
                ),
                "statistic": statistic,
                "p_value": p_value
            })


fold_results = pd.DataFrame(fold_results)


# ============================================================
# Holm-Bonferroni correction for fold-level tests
# ============================================================

fold_results["p_holm"] = np.nan
fold_results["significant"] = False


for partition in PARTITIONS:

    for proposed_model in PROPOSED_MODELS:

        mask = (
            (fold_results["partition"] == partition) &
            (fold_results["proposed_model"] == proposed_model)
        )

        valid = (
            mask &
            fold_results["p_value"].notna()
        )

        if valid.sum() > 0:

            corrected = multipletests(
                fold_results.loc[valid, "p_value"],
                alpha=0.05,
                method="holm"
            )

            fold_results.loc[valid, "p_holm"] = corrected[1]
            fold_results.loc[valid, "significant"] = corrected[0]


# ============================================================
# Save fold-level results
# ============================================================

fold_results.to_csv(
    "wilcoxon_fold_level_results.csv",
    index=False
)


# ============================================================
# Print fold-level results
# ============================================================

print("\n" + "=" * 100)
print("FOLD-LEVEL WILCOXON SIGNED-RANK TEST")
print("=" * 100)

for partition in PARTITIONS:

    print(f"\n\nPARTITION: {partition.upper()}")

    for proposed_model in PROPOSED_MODELS:

        print(f"\n--- {proposed_model} ---")

        temp = fold_results[
            (fold_results["partition"] == partition) &
            (fold_results["proposed_model"] == proposed_model)
        ].copy()

        print(
            temp[
                [
                    "baseline_model",
                    "n",
                    "proposed_mean",
                    "baseline_mean",
                    "mean_difference",
                    "statistic",
                    "p_value",
                    "p_holm",
                    "significant"
                ]
            ].to_string(index=False)
        )