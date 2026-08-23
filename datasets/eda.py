import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', None)


def process_longitudinal_study(df):
    new_df = df.copy()

    # df = pd.read_csv("heart.csv")

    # ensure sorted intervals per patient
    df2 = new_df.sort_values(["pid", "time", "time2"])

    # build list of (time, time2) tuples per pid
    pid_intervals = (
        df2.groupby("pid")
        .apply(lambda g: list(zip(g["time"].astype(float), g["time2"].astype(float))))
        .to_dict()
    )

    # print in your requested format
    for pid, intervals in pid_intervals.items():
        intervals_str = ", ".join([f"({int(s) if s.is_integer() else s}, {int(e) if e.is_integer() else e})"
                                  for s, e in intervals])
        print(f"pid {pid}: {intervals_str}")
    return new_df


def print_data_information(file_name=None, colName=None):
    df = pd.read_csv(
        f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/raw_data/{file_name}.csv")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    if file_name == "TCGA":

        df2 = df.rename(columns={
            "status": "event",
            "age_at_index": "num_age_at_index",
            "age_at_diagnosis": "num_age_at_diagnosis"
        })[["time", "event", "num_age_at_index", "num_age_at_diagnosis"]].copy()

        df3 = df.drop(columns=[ "age_at_index",  "age_at_diagnosis", "time", "status"]).copy() 
        df3 = df3.add_prefix("fac_")

        df = pd.concat([df2, df3], axis=1)

    if file_name == "metabrick":
        df.rename(columns={"status": "event", "MKI67": "num_MKI67", "EGFR": "num_EGFR", "PGR": "num_PGR", "ERBB2": "num_ERBB2", "age_at_diagnosis": "num_age_at_diagnosis"}, inplace=True)
        df.rename(columns={"hormone_treatment": "fac_hormone_treatment_ordinal", "radiotherapy": "fac_radiotherapy_ordinal", "chemotherapy": "fac_chemotherapy_ordinal", "ER_positive": "fac_ER_positive_ordinal"}, inplace=True)

    print(file_name)
    # if file_name in ["student_dropout"]:
    #     df = df.rename(columns={"final_result": f"event", "active_days": "time"})
    #     df["event"] = (df["event"] == "Withdrawn").astype(int)
        
    #     num_col = ["studied_credits", "total_clicks", "unique_resources", "num_quiz", "avg_score", "num_assess_attempted", "total_weight", "module_presentation_length"]
    #     fac_col = ["code_module", "code_presentation", "gender", "region", "highest_education", "imd_band", "age_band", "num_of_prev_attempts", "disability"]
        
    #     df = df.rename(columns={col: f"num_{col}" for col in num_col})
    #     df = df.rename(columns={col: f"fac_{col}" for col in fac_col}) 
        
    #     df = df.drop(columns=["id_student", "num_forum"], errors="ignore")
    #     df.to_csv(f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/raw_data/{file_name}.csv", index=False)
    
    
    status_column = 'event' if 'event' in df.columns else 'status'
    time_column = 'time' if 'time' in df.columns else 'duration'

    print(df.head(5))
    print(df.drop(columns=[status_column, time_column]).shape)
    if colName is not None and colName in df.columns:
        print(df[colName].value_counts())

    print("*"*150)
    print(f"events flag: {df[status_column].unique()}")
    value_counts = df[status_column].value_counts()

    count_0 = value_counts.get(0, 0)
    count_1 = value_counts.get(1, 0)
    count_2 = value_counts.get(2, 0)
    total = count_0 + count_1 + count_2

    print(f"events 1: {count_1} and rate : {count_1/(total)*100:.2f}%")
    print(f"events 0: {count_0} and rate: {count_0/(total)*100:.2f}%")
    print(f"events 2: {count_2} and rate: {count_2/(total)*100:.2f}%")

    df.loc[df[status_column] == 2, status_column] = 0
    value_counts = df[status_column].value_counts()

    count_0 = value_counts.get(0, 0)
    count_1 = value_counts.get(1, 0)

    print("*"*150)
    print(f"events 1: {count_1} and rate: {count_1/(total)*100:.2f}%")
    print(f"events 0: {count_0} and rate: {count_0/(total)*100:.2f}%")

    print("*"*150)
    num_cols = [c for c in df.columns if c.startswith('num_')]
    fac_cols = [c for c in df.columns if c.startswith('fac_')]

    # print(num_cols)
    print(
        f"Before: numerical {len(num_cols)} and categorical: {len(fac_cols)}")

    print("*"*150)
    print("Any missing in time:", df[time_column].isna().any())
    print("Any missing in event:", df[status_column].isna().any())
    print("Any missing in numerical covariates:",
          df[num_cols].isna().any().any())
    print("Any missing in categorical covariates:",
          df[fac_cols].isna().any().any())

    print("*"*150)
    miss_ratio = df.isna().mean()

    # Filter only columns where missing rate > 0
    miss_ratio = miss_ratio[miss_ratio > 0]

    if miss_ratio.empty:
        print("No missing values in covariates")
    else:
        print("Covariates with missing values:")
        print(miss_ratio.sort_values(ascending=False))

    print("*"*150)
    print(miss_ratio)

    print(f"length of numerical columns: {len(num_cols)}")
    print(f"length of categorical columns: {len(fac_cols)}")

    print("*"*150)
    for column in fac_cols:
        unique_values = sorted(
            df[column].dropna().unique(),
            key=lambda x: str(x)
        )
        print(f"Column: {column}, Unique Values: {unique_values}")


def save_client_statistics_table(file_name, center_col="fac_inst", save_path=None):
    df = pd.read_csv(
        f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/raw_data/{file_name}.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(df.columns)

    status_col = "event" if "event" in df.columns else "status"

    # convert status=2 to censored
    df.loc[df[status_col] == 2, status_col] = 0

    total_samples = len(df)

    # feature count (exclude time, event/status, and center column)
    exclude_cols = {"time", "duration", "event", "status", center_col}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    n_features = len(feature_cols)

    grouped = df.groupby(center_col)

    centers = []
    sample_counts = []
    sample_pcts = []
    feature_counts = []
    event_counts = []
    event_pcts = []
    censored_counts = []
    censored_pcts = []

    for center, g in grouped:
        n = len(g)
        events = (g[status_col] == 1).sum()
        censored = (g[status_col] == 0).sum()

        centers.append(center)
        sample_counts.append(n)
        sample_pcts.append(f"{(n / total_samples) * 100:.2f}%")
        feature_counts.append(n_features)
        event_counts.append(events)
        event_pcts.append(f"{(events / n) * 100:.2f}%")
        censored_counts.append(censored)
        censored_pcts.append(f"{(censored / n) * 100:.2f}%")

    result_df = pd.DataFrame(
        [
            ["Samples"] + sample_counts,
            ["Samples (%)"] + sample_pcts,
            ["Features"] + feature_counts,
            ["Events"] + event_counts,
            ["Events (%)"] + event_pcts,
            ["Censored"] + censored_counts,
            ["Censored (%)"] + censored_pcts,
        ],
        columns=["Region"] + centers
    )

    if save_path is None:
        save_path = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/{file_name}_client_statistics_table.csv"

    result_df.to_csv(save_path, index=False)

    print(f"Saved to: {save_path}")
    return result_df


if __name__ == "__main__":
    high_censored_datasets = ["dataDIVAT1", "dataDIVAT2", "dataDIVAT3", "student_dropout", "divorce", "hdfail", "flchain", "prostateSurvival", "nwtco","actg"]

    file_name = "student_dropout"
    print_data_information(file_name, colName=None)

    # save_client_statistics_table(file_name, center_col="fac_district")
