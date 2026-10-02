import argparse 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CoxPH', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='metabric', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--Pseudo_value', type=str, default='non-DP', help='the pseudo value calculation strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=50, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=20, help='number of patience')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0, help="fold number of 5-fold cross validation")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./Fed_CoxPH/data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./Fed_CoxPH/logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./Fed_CoxPH/models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1.0, help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_perc_interval', type=int, default=10,  help='Interval for pseudo values calculation')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')

    args = parser.parse_args()
    return args 

def compute_eval_time(args):
    local_unique_times = []

    for client_id in range(args.n_parties):
        train_df, val_df, _, _, _ = get_local_dataset(args.dataset, args.partition, client_id, args.run_id, args)
        train_time, _ = train_df["time"], train_df["event"]

        val_time, _ = val_df["time"], val_df["event"]
        local_unique_times.append(
            np.unique(np.concatenate([train_time, val_time])))

    glob_unique_time = np.unique(np.concatenate(local_unique_times))
    pseudo_evaltime = np.percentile(glob_unique_time, np.arange(
        args.pseudo_perc_interval, 99, args.pseudo_perc_interval))
    return pseudo_evaltime