import json
import itertools


def verify_duplicate_molecules(filenames):
    """
    Given a list of filenames corresponding to tweezers datafiles, checks whether any files shared
    the same 'unique' ID.

    Arguments:
        filenames (list of str): list of names of experimental data files.

    Raises:
        ValueError: if >1 filenames have the same ID.
    """
    # check for molecules with duplicate IDs
    molecule_ids = [f.split('#')[1][:-4] for f in filenames]
    molecule_id_counts = [(mid, molecule_ids.count(mid)) for mid in molecule_ids]
    duplicate_ids = list(set([mid for (mid, count) in molecule_id_counts if count > 1]))
    n_duplicate_ids = len(molecule_ids) - len(set(molecule_ids))
    if n_duplicate_ids > 0:
        raise ValueError(
            f'{n_duplicate_ids} files have duplicate molecule IDs: {", ".join(duplicate_ids)}'
        )
    return


def load_manual_region_lims(data_path):
    """
    Loads a json file containing manually-defined time limits as a function of unique molecule ID.

    Arguments:
        data_path (str): path to a json file.

    Returns:
        manual_region_limits (dict): time limits for each unique molecule ID.
    """
    with open(f'{data_path}/manual_region_lims.json') as f_in:
        return json.load(f_in)


def apply_manual_region_limits(data_Lc, limits):
    """
    Given constant-distance data, truncates the time series at the beginning / end according to a
    pair of min / max time limits.

    Arguments:
        data_Lc (pd.DataFrame): experimental data corresponding to one contiguous region of
                                constant-distance data.
        limits (tuple of float): min, max time used to filter experimental data.

    Returns:
        data_Lc (pd.DataFrame): experimental data corresponding to one contiguous region of
                                constant-distance data, filtered for min / max time limits.
    """
    # apply manual region time limits, if applicable
    manual_region_t_start, manual_region_t_end = limits
    if manual_region_t_start is not None:
        data_Lc = data_Lc.loc[data_Lc['time'] >= manual_region_t_start]
    if manual_region_t_end is not None:
        data_Lc = data_Lc.loc[data_Lc['time'] <= manual_region_t_end]
    data_Lc = data_Lc.reset_index()

    return data_Lc


def group_states(states):
    """
    Given a sequence of values, and returns a list of regions of consecutive identical values
    defined by their start/end indexes.

    Arguments:
        states (list): any sequence of values.

    Returns:
        regions (list): list of regions of consecutive identical values defined by their start/end =
                        indexes in the initial list.
    """
    grouped_states = [(key, sum(1 for _ in group)) for key, group in itertools.groupby(states)]

    regions = []
    total_count = 0
    for state, count in grouped_states:
        if state:
            regions.append((total_count, total_count + count))
        total_count += count

    return regions
