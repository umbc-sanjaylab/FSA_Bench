import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def _clean_categories(values):
    categories = [value for value in values if pd.notna(value)]
    if not categories:
        return ["__missing__"]
    return sorted(categories, key=lambda value: (str(type(value)), str(value)))


def _prepare_categorical_frame(df, columns):
    prepared = df[columns].copy()
    for col in columns:
        prepared[col] = prepared[col].where(prepared[col].notna(), "__missing__")
    return prepared


def build_global_cat_schema(df):
    global_cats = [c for c in df.columns if c.startswith( "fac_") and not c.endswith("_ordinal") and not c.endswith("_binary")]
    global_cats_dict = {}

    for c in global_cats:
        global_cats_dict[c] = _clean_categories(df[c].unique().tolist())

    # Determine binary vs multi-class based on GLOBAL union
    bin_cols_name = [col for col, unique_values in global_cats_dict.items() if len(unique_values) == 2]
    ordinal_cols_name = [col for col, unique_values in global_cats_dict.items() if len(unique_values) > 2]

    return global_cats_dict, bin_cols_name, ordinal_cols_name


def encode_with_schema(train, val, test, global_cats_dict, bin_cols_name, multi_cols_name):
    available_cols = set(train.columns) & set(val.columns) & set(test.columns)
    filtered_cats_dict = {
        col: cats for col, cats in global_cats_dict.items() if col in available_cols}
    filtered_bin_cols = [
        col for col in bin_cols_name if col in filtered_cats_dict]
    filtered_multi_cols = [
        col for col in multi_cols_name if col in filtered_cats_dict]

    # Binary encoder (drop if_binary)
    if filtered_bin_cols:
        bin_categories = [filtered_cats_dict[c] for c in filtered_bin_cols]
        train_bin_input = _prepare_categorical_frame(train, filtered_bin_cols)
        val_bin_input = _prepare_categorical_frame(val, filtered_bin_cols)
        test_bin_input = _prepare_categorical_frame(test, filtered_bin_cols)
        bin_enc = OneHotEncoder(
            categories=bin_categories,
            handle_unknown="ignore",
            drop="if_binary",
            sparse_output=False)
        bin_enc.fit(train_bin_input)  # fit is safe; categories fixed

        bin_names = bin_enc.get_feature_names_out(filtered_bin_cols)
        train_bin = pd.DataFrame(bin_enc.transform(
            train_bin_input), index=train.index, columns=bin_names).astype(int)
        val_bin = pd.DataFrame(bin_enc.transform(
            val_bin_input),   index=val.index,   columns=bin_names).astype(int)
        test_bin = pd.DataFrame(bin_enc.transform(
            test_bin_input),  index=test.index,  columns=bin_names).astype(int)
    else:
        train_bin = pd.DataFrame(index=train.index)
        val_bin = pd.DataFrame(index=val.index)
        test_bin = pd.DataFrame(index=test.index)

    # Multi-class OHE with fixed categories
    if filtered_multi_cols:
        multi_categories = [filtered_cats_dict[c] for c in filtered_multi_cols]
        train_multi_input = _prepare_categorical_frame(
            train, filtered_multi_cols)
        val_multi_input = _prepare_categorical_frame(val, filtered_multi_cols)
        test_multi_input = _prepare_categorical_frame(
            test, filtered_multi_cols)
        ohe = OneHotEncoder(categories=multi_categories,
                            handle_unknown="ignore", sparse_output=False)
        ohe.fit(train_multi_input)

        ohe_names = ohe.get_feature_names_out(filtered_multi_cols)
        train_ohe = pd.DataFrame(ohe.transform(
            train_multi_input), index=train.index, columns=ohe_names)
        val_ohe = pd.DataFrame(ohe.transform(
            val_multi_input),   index=val.index,   columns=ohe_names)
        test_ohe = pd.DataFrame(ohe.transform(
            test_multi_input),  index=test.index,  columns=ohe_names)
    else:
        train_ohe = pd.DataFrame(index=train.index)
        val_ohe = pd.DataFrame(index=val.index)
        test_ohe = pd.DataFrame(index=test.index)

    # drop original fac (non-ordinal)
    fac_cols = list(filtered_cats_dict.keys())

    train_out = pd.concat([train.drop(columns=fac_cols, errors="ignore"), train_bin, train_ohe], axis=1)
    val_out = pd.concat([val.drop(columns=fac_cols, errors="ignore"),   val_bin,   val_ohe], axis=1)
    test_out = pd.concat([test.drop(columns=fac_cols, errors="ignore"),  test_bin,  test_ohe], axis=1)

    # Enforce identical schema/order across splits
    train_cols = sorted(train_out.columns)
    train_out = train_out.reindex(train_cols, axis=1)
    val_out = val_out.reindex(train_cols, axis=1, fill_value=0)
    test_out = test_out.reindex(train_cols, axis=1, fill_value=0)

    return train_out, val_out, test_out
