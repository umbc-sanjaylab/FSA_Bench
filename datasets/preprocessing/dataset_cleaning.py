import pandas as pd 

############################################### Preprocess Federated Survival Datasets #####################################################


def clean_dataset(df, dataset_name, region_column):

    # A study of survival of patients from 17 cancer
    if dataset_name == "TCGA":

        # rename of the covariates and status variable
        df.rename(columns={"status": "event", "age_at_index": "num_age_at_index",
                  "age_at_diagnosis": "num_age_at_diagnosis"}, inplace=True)

        #  rename the columns with binary suffix
        remove_cols = ['event', 'num_age_at_index',  'num_age_at_diagnosis', 'time', region_column]
        fac_columns = [col for col in df.columns if col not in remove_cols]
        df.rename(  columns={c: f"fac_{c}_binary" for c in fac_columns}, inplace=True)

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # common multi-centric datasets
    elif dataset_name in ["Aids2", "Pbc3"]:

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0"], errors="ignore")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # LeukSurv dataset
    elif dataset_name == "LeukSurv":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0",
                     "num_xcoord", "num_ycoord"], errors="ignore")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Dialysis dataset
    elif dataset_name == "Dialysis":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0", "num_begin"], errors="ignore")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # carcinoma of the oropharynx
    elif dataset_name == "d.oropha.rec":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0"], errors="ignore")

        # rename the covariates with _ordinal suffix
        fac_cols = ["fac_treatm", "fac_grade", "fac_cond", "fac_site", "fac_tstage", "fac_nstage"]
        df = df.rename(columns={col: f"{col}_ordinal" for col in fac_cols})

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Chronic Granulomatous Disease that weakens white blood cells
    elif dataset_name == "cgd":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0", "fac_center"], errors="ignore")

        # Ordinal encoding of the covariate and map the encoding to the column
        agestr_dict = {
            "fac_treat": {'placebo': 0, 'rIFN-g': 1},
            "fac_sex": {'female': 0, 'male': 1},
            "fac_inherit": {"X-linked": 0, "autosomal": 1},
            "fac_steroids": {'N': 1, 'Y': 0},
            "fac_propylac": {'N': 1, 'Y': 0}
        }

        for key, value in agestr_dict.items():
            df[key] = df[key].map(value)

        # rename the columns with ordinal suffix
        rename_dict = {"fac_treat": "fac_treat_binary", "fac_sex": "fac_sex_binary",
                       "fac_propylac": "fac_propylac_binary", "fac_inherit": "fac_inherit_binary", "fac_steroids": "fac_steroids_binary"}
        df = df.rename(columns=rename_dict)

        # cast the type of the covariates to numeric
        for col in rename_dict.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Study of drug addicts undergoing treatment by University of Illinois
    elif dataset_name == "uis":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0"], errors="ignore")

        # binary encoding and rename the covariate with _binary suffix
        df['fac_treat'] = df['fac_treat'].map({'short': 0, 'long': 1})
        df = df.rename(columns={'fac_treat': 'fac_treat_binary'})

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Study of old age life histories above age 50 at rural area of scania, southern Sweden
    elif dataset_name == "scania":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0",
                     "num_birthdate"], errors="ignore")

        # Ordinal encoding of the covariate and map the encoding to the column
        agestr_dict = {
            "fac_sex": {'female': 0, 'male': 1},
            "fac_immigrant": {'no': 1, 'yes': 0},
            "fac_ses": {'lower': 1, 'upper': 0}
        }

        for key, value in agestr_dict.items():
            df[key] = df[key].map(value)

        # rename the columns with ordinal suffix
        rename_dict = {"fac_ses": "fac_ses_binary",
                       "fac_sex": "fac_sex_binary", "fac_immigrant": "fac_immigrant_binary"}
        df = df.rename(columns=rename_dict)

        # cast the type of the covariates to numeric
        for col in rename_dict.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    #  A study of cancer statistics by the SEER program in the United States.
    elif dataset_name == "seer":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0"], errors="ignore")

        # rename the columns with ordinal suffix
        rename_dict = {"status": "event", "Ageatdiagnosis": "num_Ageatdiagnosis", "Race_ord": "fac_Race_ordinal", "Marital_3": "fac_Marital_3_ordinal",
                       "Histology_ord": "fac_Histology_ordinal", "Grade_3": "fac_Grade_3_ordinal", "Tumor_size_ord": "fac_Tumor_size_ordinal",
                       "T.stage_ord": "fac_T_stage_ordinal", "N_stage_ord": "fac_N_stage_ordinal", "M_stage_ord": "fac_M_stage_ordinal",
                       "RT_ord": "fac_RT_ordinal", "Chemo_di": "fac_Chemo_di_ordinal", "Surg_ord": "fac_Surg_ordinal"}
        df = df.rename(columns=rename_dict)

        # cast the type of the covariates to numeric
        for col in rename_dict.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Make compeing risks events binary (1=event occurred, 0=censored/other event)
    if "event" in df.columns:
        df["event"] = (df["event"] == 1).astype("int64")

    return df