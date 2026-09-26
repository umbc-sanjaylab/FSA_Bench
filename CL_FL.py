from federated.client import local_train_net, train_net_all 

def run_centralized(args, data, model_spec, eval_times):
    """
    Run centralized training and evaluation.

    Parameters
    ----------
    args
        Experiment configuration.

    data
        PreparedSurvivalData containing centralized and client-level data.

    model_spec
        Model configuration from MODEL_REGISTRY.

    eval_times
        Evaluation time points used for survival metrics.

    Returns
    -------
    dict
        Final test metrics.
    """

    print("\n================ Centralized Training ================")

    # ------------------------------------------------------------
    # 1. Retrieve centralized train / validation / test data
    # ------------------------------------------------------------

    train = data.centralized.train
    val = data.centralized.val
    test = data.centralized.test

    feature_names = data.feature_names

    input_dim = train.X.shape[1]

    print(f"Train shape : {train.X.shape}")
    print(f"Val shape   : {val.X.shape}")
    print(f"Test shape  : {test.X.shape}")
    print(f"Input dim   : {input_dim}")

    # ------------------------------------------------------------
    # 2. Initialize AFT-Weibull model
    # ------------------------------------------------------------

    model_class = model_spec["model"]

    model = model_class(
        input_dim=input_dim
    ).to(args.device)

    # ------------------------------------------------------------
    # 3. Train and evaluate
    # ------------------------------------------------------------

    try:

        (
            test_cindex,
            test_brier,
            test_nbll,
            test_p_value,
            test_mae_po,
        ) = train_net_all(
            0,                      # net_id
            model,
            train.X,
            train.time,
            train.event,
            val.X,
            val.time,
            val.event,
            test.X,
            test.time,
            test.event,
            eval_times,
            feature_names,
            args,
            args.device,
        )

    except Exception as exc:

        print(
            f"[Centralized training error] "
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        test_cindex = np.nan
        test_brier = np.nan
        test_nbll = np.nan
        test_p_value = np.nan
        test_mae_po = np.nan

    # ------------------------------------------------------------
    # 4. Return common result format
    # ------------------------------------------------------------

    results = {
        "val_c_index": np.nan,
        "c_idx": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "p_value": test_p_value,
        "weighted_MAE_PO": test_mae_po,
        "comm_round_done": 1,
    }

    print("\nCentralized Test Results")
    print(f"C-index       : {test_cindex}")
    print(f"IBS           : {test_brier}")
    print(f"NBLL          : {test_nbll}")
    print(f"D-cal p-value : {test_p_value}")
    print(f"MAE-PO        : {test_mae_po}")

    return results