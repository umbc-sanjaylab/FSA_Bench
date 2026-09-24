import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests


######################### DATA #########################

data = {
    "Dataset": [
        "TCGA", "Aids2", "LeukSurv", "Pbc3", "d.oropha.rec",
        "cgd", "uis", "scania", "seer", "Dialysis"
    ],

    "Fed. KM": [
        0.5004, 0.5012, 0.5062, 0.5000, 0.5015,
        0.5000, 0.5019, 0.5000, 0.5035, 0.5113
    ],

    "AFT Weibull": [
        0.7480, 0.5012, 0.6594, 0.4771, 0.6472,
        0.4762, 0.5379, 0.4875, 0.8078, 0.6141
    ],

    "Fed. CoxPH": [
        0.7523, 0.5539, 0.6635, 0.4260, 0.6593,
        0.5118, 0.5644, 0.4844, 0.8068, 0.6549
    ],

    "LinearPH": [
        0.7506, 0.5259, 0.6274, np.nan, 0.6094,
        0.5769, 0.5516, 0.4906, 0.7855, np.nan
    ],

    "DeepPseudo": [
        0.7360, 0.5322, 0.5572, 0.5044, 0.5080,
        0.5476, 0.5015, 0.5050, 0.7380, 0.5299
    ],

    "DeepHit": [
        0.6781, 0.5517, 0.4612, np.nan, 0.5896,
        0.5538, 0.5074, 0.5191, 0.7140, np.nan
    ],

    "NNPH": [
        0.7357, 0.5214, 0.6450, np.nan, 0.6020,
        0.6797, 0.5238, 0.5009, 0.8200, np.nan
    ],

    "NNnPH": [
        0.7157, 0.5474, 0.6431, np.nan, 0.6409,
        0.6391, 0.5097, 0.5033, 0.8178, np.nan
    ],

    "FedPDNN": [
        0.7113, 0.5630, 0.6686, 0.5679, 0.6487,
        0.6800, 0.5192, 0.5157, 0.7787, 0.5547
    ],

    "FedPLSTM": [
        0.7151, 0.5548, 0.6706, 0.5653, 0.6423,
        0.5984, 0.5284, 0.5054, 0.8039, 0.5701
    ],

    "FedPAttn": [
        0.7297, 0.5559, 0.6735, 0.5905, 0.6396,
        0.6147, 0.5407, 0.5087, 0.8150, 0.5942
    ]
}

df = pd.DataFrame(data)


######################### COMPARATORS #########################
comparators = [ "Fed. KM", "AFT Weibull", "Fed. CoxPH", "LinearPH", "DeepPseudo", "DeepHit", "NNPH", "NNnPH"]

 
######################### FUNCTION FOR WILCOXON + RANK-BISERIAL CORRELATION #########################

def wilcoxon_comparison(fedpattn, comparator):
    """
    Compare FedPAttn against one comparator.

    Positive difference = FedPAttn performs better.
    """

    # Keep only paired observations
    mask = fedpattn.notna() & comparator.notna()

    x = fedpattn[mask].to_numpy()
    y = comparator[mask].to_numpy()

    # FedPAttn - Comparator
    differences = x - y

    # Remove exact zero differences
    differences = differences[differences != 0]

    n = len(differences)

    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "W": np.nan,
            "p": np.nan,
            "Med. ΔC": np.nan,
            "r_rb": np.nan
        }
 
    ######################### Wilcoxon signed-rank test #########################
    result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method="auto")

    W = result.statistic
    p = result.pvalue 
   
    ########################## W-L = FedPAttn wins - losses  ######################### 
    wins = np.sum(differences > 0)
    losses = np.sum(differences < 0)

    
    ###### Rank-biserial correlation, r_rb = (W+ - W-) / (W+ + W-)  ###################  
    ranks = rankdata(np.abs(differences), method="average") 
    W_plus = np.sum(ranks[differences > 0])
    
    W_minus = np.sum(ranks[differences < 0]) 
    r_rb = (W_plus - W_minus) / (W_plus + W_minus)

     
    ########################## Median C-index difference ################################  

    median_delta = np.median(differences) 
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "W": W,
        "p": p,
        "Med. ΔC": median_delta,
        "r_rb": r_rb
    }

 
######################################## 4. RUN ALL COMPARISONS  ########################## 

results = []
for comparator in comparators:
    result = wilcoxon_comparison(df["FedPAttn"], df[comparator])
    result["Comparator"] = comparator 
    results.append(result) 
    
results_df = pd.DataFrame(results)
 
######################################## 5. HOLM CORRECTION ###################################
results_df["p_Holm"] = np.nan
valid = results_df["p"].notna()
results_df.loc[valid, "p_Holm"] = multipletests( results_df.loc[valid, "p"], method="holm" )[1]

 
############################ 6. FORMAT W-L COLUMN ################################################  
results_df["W–L"] = ( results_df["wins"].astype(int).astype(str) + "–" + results_df["losses"].astype(int).astype(str))

 
############################  7. REORDER COLUMNS  #####################################################  
results_df = results_df[[ "Comparator", "n", "W–L", "W", "p", "p_Holm", "Med. ΔC", "r_rb" ]]
 
#############################  8. ROUND NUMBERS ############################################################# 
results_df["W"] = results_df["W"].round(0).astype(int)
results_df["p"] = results_df["p"].map( lambda x: f"{x:.5f}") 
results_df["p_Holm"] = results_df["p_Holm"].map( lambda x: f"{x:.5f}")
 
results_df["Med. ΔC"] = results_df["Med. ΔC"].map( lambda x: f"{x:+.4f}") 
results_df["r_rb"] = results_df["r_rb"].map( lambda x: f"{x:+.3f}")

 
####################################### 9. BOLD Holm-CORRECTED SIGNIFICANT RESULTS ################################  

# Need numeric p_Holm again for checking significance
holm_numeric = multipletests(results_df["p"].astype(float),method="holm")[1]
results_df["p_Holm_numeric"] = holm_numeric 

results_df["p_Holm"] = [ f"\\textbf{{{p:.5f}}}" if p < 0.05 else f"{p:.5f}"  for p in holm_numeric] 
results_df.drop(columns=["p_Holm_numeric"], inplace=True)
 
####################################### 10. PRINT RESULT ######################################################### 
print(results_df.to_string(index=False))
results_df.to_csv("result.csv")