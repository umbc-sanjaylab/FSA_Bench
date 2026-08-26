import random
import torch
import numpy as np


################################################## Federated data Partition #########################################################

def federated_partition(data, num_centers, region_column):
    """
    Split data according to region across each center
    Arguments:
        data -- combined data
        num_centers -- number of centers
        region_column -- region column of the dataframe. 
    Returns:
        Dict with center_id : indices of data assigned to center
    """

    # Get unique regions
    unique_values = sorted(data[region_column].unique())

    # Optional check
    if num_centers != len(unique_values):
        print(
            f"Warning: num_centers ({num_centers}) != number of regions ({len(unique_values)})")

    dict_center_idxs = {}
    for _, region in enumerate(unique_values):

        # Get indices for this region and store it in a dictionary
        region_idxs = data.index[data[region_column] == region].tolist()
        dict_center_idxs[region] = region_idxs

    return dict_center_idxs

################################################## IID data Partition #########################################################


def sample_iid(data, num_centers, seed_no):
    """
    Randomly split data evenly across each center
    Arguments:
        data -- combined data
        num_centers -- number of centers
    Returns:
        Dict with center_id : indices of data assigned to center
    """
    rng = np.random.default_rng(seed_no) 
    n = data.shape[0]
    idxs = rng.permutation(n)
    batch_idxs = np.array_split(idxs, num_centers)
    dict_center_idxs = {i: batch_idxs[i] for i in range(num_centers)}
    return dict_center_idxs  

################################################## Non-IID data Partition #####################################################

def sample_by_quantiles(df, num_centers):
    """
    Split samples across centers based on time quantiles (non-IID by time)
    Arguments:
        df -- combined data as dataframe
        num_centers -- number of centres
    Returns:
        Dict with center_id : indices of data assigned to center
    """

    # Use time as the non-IID driver (earlier time vs later time distribution shift)
    data = np.array(df['time'])
    dict_center_idxs, all_idxs = {}, np.array([i for i in range(len(data))])

    # Track quantile cutoff and previously selected indices to avoid overlap
    quantile = 1 / num_centers
    previous_idxs = torch.zeros(len(data), dtype=torch.bool).numpy()

    for i in range(num_centers):
        # Guard: quantile should never exceed 1.0
        if quantile > 1:
            ValueError

        cutoff = np.quantile(data, quantile)
        selected_idxs = data <= cutoff
        idxs_in_quantile = selected_idxs & ~previous_idxs

        # Update mask so next quantile only takes new samples
        previous_idxs = previous_idxs | idxs_in_quantile

        # Store indices assigned to center i for this quantile slice
        dict_center_idxs[i] = all_idxs[idxs_in_quantile]
        quantile += 1 / num_centers

    return dict_center_idxs


def mergeDictionary(dict_1, dict_2):
    """
    Merge two client-index dictionaries by concatenating indices per client
    Arguments:
    dict_1 -- first dict {client_id: idxs}
    dict_2 -- second dict {client_id: idxs}
    Returns:
    merged dict {client_id: concatenated idxs}
    """
    ds = [dict_1, dict_2]
    d = {}
    for k in dict_1.keys():
        d[k] = np.concatenate(list(d[k] for d in ds))
    return d


def non_iid_skewed(data, num_centers, seed_no):
    """
    Create a skewed non-IID split:
        - A subset is split IID across clients
        - Remaining samples are split by time-quantiles across clients
    Arguments:
        data -- combined dataframe
        num_centers -- number of clients
        seed_no -- random seed
    Returns:
        Dict with client_id : indices of data assigned to client
    """
    rng = np.random.default_rng(seed_no)
    random.seed(seed_no)

    n = data.shape[0]  
    all_idxs = np.arange(n)

    # Select a chunk to behave like IID (same size as one client's share)
    # idxs1 = np.array(random.sample(range(0, n), int(n / num_centers)))
    # idxs2 = np.setdiff1d(idxs, idxs1)
    
    # Select a chunk to behave like IID (same size as one client's share)
    idxs1 = rng.choice(all_idxs, size=int(n / num_centers), replace=False)
    idxs2 = np.setdiff1d(all_idxs, idxs1)

    # Split the dataframe into IID-part and non-IID-by-time part
    data1 = data.iloc[idxs1].reset_index(drop=True)
    data2 = data.iloc[idxs2].reset_index(drop=True)

    # IID on subset, quantile-based non-IID on remaining subset
    dict_center_idxs1_relative = sample_iid(data1, num_centers, seed_no)
    dict_center_idxs2_relative = sample_by_quantiles(data2, num_centers)

    # Merge both index allocations per client
    dict_center_idxs = {}

    for client_id in range(num_centers):
        # map back to original data indices
        iid_original = idxs1[dict_center_idxs1_relative[client_id]]
        noniid_original = idxs2[dict_center_idxs2_relative[client_id]]

        dict_center_idxs[client_id] = np.concatenate(
            [iid_original, noniid_original]
        )
    return dict_center_idxs 

############################################## Method for Non-IID Censoring ####################################################


def non_iid_censor(data, num_centers, seed_no):
    """_summary_

    Args:
        data (_type_): _description_
        num_centers (_type_): _description_
        seed_no (_type_): _description_

    Returns:
        dict_center_idxs: _description_
    """
    np.random.seed(seed_no)
    random.seed(seed_no)
    column = "status" if 'status' in data.columns else "event"

    data_cen = data[data[column] == 0]
    data_uncen = data[data[column] == 1]

    data_cen_index = np.array(data_cen.index.to_list())
    np.random.shuffle(data_cen_index)
    data_cen_index = data_cen_index.tolist()
    data_uncen_index = np.array(data_uncen.index.to_list())
    np.random.shuffle(data_uncen_index)
    data_uncen_index = data_uncen_index.tolist()

    uncen_rate = [0.15, 0.30, 0.45, 0.60, 0.75]
    cen_rate = [1.0 - x for x in uncen_rate]
    uncen_rate.insert(0, 0.0)
    cen_rate.insert(0, 0.0)
    batch_idxs = []
    for i in range(num_centers):
        batch_idxs.append(data_uncen_index[int(uncen_rate[i]*len(data)/num_centers):int(uncen_rate[i+1]*len(data)/num_centers)]+(
            data_cen_index[int(cen_rate[i]*len(data)/num_centers):int(cen_rate[i+1]*len(data)/num_centers)]))
    dict_center_idxs = {i: batch_idxs[i] for i in range(num_centers)}
    return dict_center_idxs

################################################################################################################################
