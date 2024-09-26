import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats

import helpers


def load_data(
    file_path,
    data_period,
    savgol_window_tl=1,
    savgol_window_sb=1,
    peak_heights=[0, 1],
    t_min=0,
    require_unfolded_MBP=True,
):
    """
    Given the path to an experimental data file, loads the raw data and determines all valid
    regions of constant-distance data.

    Arguments:
        file_path (str): relative path to an experimental data file.
        data_period (float): time gap between consective data points (1 / data frequency) [s].
        savgol_window_tl (int): window length of Savitzky-Golay filter used to determine the length
                                change during processive translocation ("tl").
        savgol_window_sb (int): window length of Savitzky-Golay filter used to determine the length
                                change during retrograde movement (slipback, "sb").
        peak_heights (tuple of float): (min, max) measured length which defines a valid region of
                                       constant-distance data [nm].
        region_t_min (float): minimum duration of a valid region of constant-distance data [s].
        require_unfolded_MBP (bool): additional constraint that MBP molecule should be unfolded
                                     - true in the majority of cases in this assay.

    Returns:
        data_Lc_output (pd.DataFrame): loaded experimental data file, filtered to contain valid
                                       constant-distance data only.
        molecule (str): unique ID defining the data file loaded.
    """

    # get data file ID information, to differentiate between files
    # TWOM data filenames are in the format:
    #     '[date]-[time] [description] #[bead pair id]-[molecule id].tdms'
    # therefore, full id is between '#' and '.'
    filename = file_path.split('/')[-1]
    molecule = filename.split('#')[1][:-4]
    data_Lc = pd.read_csv(file_path, index_col=0)

    data_Lc['avgLc'] = signal.savgol_filter(data_Lc['proteinLc'], savgol_window_tl, 1)
    data_Lc['avgLc_slipback'] = signal.savgol_filter(
        data_Lc['proteinLc'], savgol_window_sb, 1
    )

    # Find regions of constant-distance measurement in data file
    # In the raw data, 'region' refers to the measurement state (pulling, stationary, etc.),
    #     whereas later I redefine it as the index column for marking continuous regions of valid
    #     measurement data -- a similar purpose, but not identical
    # First, initialise an 'is_signal' column with zeros, but change to '1' if the Lc value is
    #     within the measurement range
    data_Lc['is_signal'] = 0
    data_Lc.loc[
        (data_Lc['proteinLc'] > peak_heights[0])
        & (data_Lc['proteinLc'] < peak_heights[1]),
        'is_signal',
    ] = 1

    # Then, detect when a column of 1s (indicating a continuous signal within the desired
    #     contour length limits) changes to 0 and back
    # diff() takes the difference between each element and that in the row above it -- for
    #     continous regions, diff always == 0, so cumsum down column doesn't change
    # once a zero appears, diff = 1, so cumsum++
    # therefore, cumsum increases by 1 when contiguous region is broken
    data_Lc['region'] = data_Lc['is_signal'].diff().ne(0).cumsum()

    output_columns = ['time', 'forceX', 'proteinLc', 'avgLc', 'avgLc_slipback', 'region']
    data_Lc_output = pd.DataFrame(columns=output_columns)
    new_region = 1
    n_min = int(t_min / data_period)
    for region in list(set(data_Lc['region'])):
        data_in_region = data_Lc.loc[
            (data_Lc['region'] == region) & (data_Lc['is_signal'] == 1)
        ].copy()

        if len(data_in_region) >= n_min and require_unfolded_MBP:
            data_in_region['unfolded'] = (data_in_region['proteinLc'] > 120)

            grouped_states = [
                (key, sum(1 for _ in group))
                for key, group in itertools.groupby(data_in_region['unfolded'])
            ]
            if not grouped_states[0][0]:
                data_in_region = data_in_region.iloc[grouped_states[0][1] :]

        # final check in case valid regions got removed in previous steps
        if len(data_in_region) >= n_min: 
            data_in_region['region'] = new_region
            new_region += 1
            data_Lc_output = pd.concat([data_Lc_output, data_in_region[output_columns]])

    return data_Lc_output, molecule


def calculate_slopes(length, time, segment_size=250, start_idx=0):
    """
    Given a constant-distance time series, segments the data into consecutive linear regions of
    defined duration.

    Arguments:
        length (np.array): measured lengths [nm].
        time (np.array): time coordinates corresponding to each measured length [s].
        segment_size (int): length of each segment [# data points].
        start_idx (int): offset from beginning of the time series to begin segmentation [# data
                         points].

    Returns:
        xs (np.array of int): x-coordinates segments.
        ys (np.array of int):  of computed segments, determined by linear regression.
        slopes (np.array of int): slopes of computed segments, determined by linear regression.
        intercepts (np.array of int): y-intercept values of computed segments, determined by linear
                                      regression.
        segment_indices (np.array of (tuple of int)): indexes (relative to the raw time series data)
                                                      defining the start/end points of each segment.
    """
    slopes = []
    intercepts = []
    segment_indices = []
    xs, ys = [], []
    iterator = list(range(start_idx, len(length) - segment_size, segment_size)) + [len(length) - 1]
    for start_idx, end_idx in zip(iterator, iterator[1:]):
        slope, intercept, _, _, _ = stats.linregress(
            time[start_idx:end_idx], length[start_idx:end_idx]
        )
        slopes.append(slope)
        intercepts.append(intercept)
        segment_indices.append((start_idx, end_idx))

        if start_idx == 0:
            xs.append(time[0])
            ys.append(slope * time[0] + intercept)

        xs.append(time[end_idx])
        ys.append(slope * time[end_idx] + intercept)

    return xs, ys, slopes, intercepts, segment_indices


def get_target_stats(
    data_path,
    data_period,
    savgol_window_tl,
    savgol_window_sb,
    peak_heights,
    region_t_min,
    segment_time_shewart,
    n_resamples,
    **kwargs,
):
    """
    Given a set of raw tweezers datafiles, detects and segments regions of constant-distance data
    to calculate the mean and standard deviation of segments.

    Used to compute the 'in-control' mean and standard deviation in control (-Cdc48) conditions.

    Arguments:
        data_path (str): relative path to the directory containing experimental (control) data.
        data_period (float): time gap between consective data points (1 / data frequency) [s].
        savgol_window_tl (int): window length of Savitzky-Golay filter used to determine the length
                                change during processive translocation ("tl").
        savgol_window_sb (int): window length of Savitzky-Golay filter used to determine the length
                                change during retrograde movement (slipback, "sb").
        peak_heights (tuple of float): (min, max) measured length which defines a valid region of
                                       constant-distance data [nm].
        region_t_min (float): minimum duration of a valid region of constant-distance data [s].
        segment_time_shewart (float): timespan over which to segment distance-time data, to
                                      construct Shewhart control chart [s].
        n_resamples (int): number of times to resample the raw experimental data within each
                           Shewhart segment.

    Returns:
        target_mean (float): mean dy per segment.
        target_st_dev (float): standard deviation of dys per segment.
    """

    filenames = [f for f in os.listdir(f'{data_path}/wlc_manual_fit') if '.csv' in f]

    dys = np.array([])
    start_offsets = np.linspace(0, segment_time_shewart, n_resamples)
    for filename in filenames:
        data_Lc, molecule = load_data(
            f'{data_path}/wlc_manual_fit/{filename}',
            data_period=data_period,
            savgol_window_tl=savgol_window_tl,
            savgol_window_sb=savgol_window_sb,
            peak_heights=peak_heights,
            t_min=region_t_min,  # s
        )

        if len(data_Lc) == 0:  # i.e. no valid regions
            continue

        for region in data_Lc['region'].unique():
            data_Lc_regional = (
                data_Lc.loc[data_Lc['region'] == region].reset_index(drop=True).copy()
            )
            for i, start_offset in enumerate(start_offsets):
                # get segments
                segment_size = int(segment_time_shewart / data_period)
                start_idx = int(start_offset / data_period)

                _, ys, _, _, _ = calculate_slopes(
                    data_Lc_regional['proteinLc'].to_numpy(),
                    data_Lc_regional['time'].to_numpy(),
                    segment_size,
                    start_idx,
                )

                # compute change in y (dy) and save, to calculate mean and std later
                dys = np.append(dys, np.diff(ys))

    return np.mean(dys), np.std(dys)


def get_passing_regions(states, min_n_successes, max_n_failures):
    """
    Given a sequence of boolean trials, returns regions of consecutive successes. Only includes
    sequences which are at least min_n_successes long, containing at most max_n_failures failures.

    Arguments:
        states (list of bool): sequence of boolean trials.
        min_n_successes (int): minimum number of sequential successes.
        max_n_failures (int): maximum number of failures in one sequence, for the sequence to still
                              be counted.

    Returns:
        regions (list of (tuple of int)): (start, end) coordinates of successful sequences.
    """

    success_count = 0
    fail_count = 0
    success_region_start = None
    most_recent_success = None
    regions = []
    for i, s in enumerate(states):
        if success_region_start is None and s is True:
            success_region_start = i
            most_recent_success = i
            success_count += 1
        elif success_region_start is not None and s is True:
            most_recent_success = i
            success_count += 1
        elif success_region_start is None and s is False:
            continue
        elif success_region_start is not None and s is False:
            fail_count += 1

        if fail_count == max_n_failures + 1:
            if success_count >= min_n_successes:
                regions += [(success_region_start, most_recent_success)]

            success_region_start = None
            most_recent_success = None
            success_count = 0
            fail_count = 0
            i -= 1

    return regions


def get_passing_states(states, min_n_successes, max_n_failures):
    """
    Given a sequence of boolean trials, returns trials which are part of a consecutive sequence of
    successes. Only includes sequences which are at least min_n_successes long, containing at most
    max_n_failures failures.

    Arguments:
        states (list of bool): sequence of boolean trials.
        min_n_successes (int): minimum number of sequential successes.
        max_n_failures (int): maximum number of failures in one sequence, for the sequence to still
                              be counted.

    Returns:
        states (list of bool): trials which make up part of a successful sequences.
    """

    regions = get_passing_regions(states, min_n_successes, max_n_failures)
    states_output = [False] * len(states)
    for idx_lo, idx_hi in regions:
        states_output[idx_lo:idx_hi] = [True] * (idx_hi - idx_lo)

    return np.array(states_output)


def upsample_segments_to_data_period(segment_states, segment_size, start_idx, array_length):
    """
    Given a sequence of boolean trials measured on segmented data, returns another sequence
    'upsampled' to the frequency of the unsegmented data.

    Arguments:
        segment_states (list of bool): sequence of boolean trials, measured on segmented data.
        segment_size (int): length of one segment, relative to unsegmented data frequency.
        start_idx (int): offset from beginning of unsegmented data set where segmentation began.
        array_length (int): length of unsegmented data set.

    Returns:
        states_per_data_point (list of bool): sequence of boolean trials, frequency-upsampled.
    """
    states_per_data_point = [False] * array_length
    for i, state in enumerate(segment_states):
        if state:
            x_lo = i * segment_size + start_idx
            x_hi = min(x_lo + segment_size, array_length)
            states_per_data_point[x_lo:x_hi] = [True] * (x_hi - x_lo)

    return states_per_data_point


def idx_regions_to_time_regions(regions, t_zero, data_period):
    """
    Given a set of regions defined by their (integer) index, returns a set of regions in time
    coordinates (defined for a specific data_period).

    Arguments:
        regions (list of (tuple of int)): indexes defining the start/end points of a set of regions.
        t_zero (float): time coordinate corresponding to index=0 [s].
        data_period (float): time gap between consective data points (1 / data frequency) [s].

    Returns:
        regions (list of (tuple of float)): time coordinates defining the start/end points of a set
                                            of regions [s].
    """
    output_regions = []
    for i_lo, i_hi in regions:
        t_start = i_lo * data_period + t_zero
        t_end = i_hi * data_period + t_zero
        output_regions += [(t_start, t_end)]
    return output_regions


def split_idx_regions_on_idx(regions, splits):
    """
    Given a set of regions defined by their (integer) index and a list of split indexes, splits
    each region overlapping a split index into two discrete regions.

    Arguments:
        regions (list of (tuple of int)): indexes defining the start/end points of a set of regions.
        splits (list of int): indexes where regions should be split, if they overlap there.

    Returns:
        regions (list of (tuple of int)): indexes defining the start/end points of a set of regions,
                                          after splitting.
    """
    # split on slipback centre points
    output_regions = []
    for i_lo, i_hi in regions:
        splitpoints = [i_lo] + [sm for sm in splits if i_lo < sm and sm < i_hi] + [i_hi]
        for i, (split_start, split_end) in enumerate(zip(splitpoints, splitpoints[1:])):
            truncated_lo = False if i == 0 else True
            truncated_hi = False if i == (len(splitpoints) - 2) else True
            output_regions += [(split_start, split_end, truncated_lo, truncated_hi)]

    return output_regions


# ----------------------------- check based on Shewart control rules ---------------------------- #
def check_run_conds_translocation(segments, target_mean, target_st_dev):
    """
    Given a set of segmented distance-time data, and the 'in-control' limits defining the
    corresponding Shewhart charts, computes regions which are deemed 'out-of-control' according to
    the Shewhart run conditions.

    Specifically applies Shewhart run conditions defining a processive translocation event.

    Arguments:
        segments (pd.DataFrame): segmented distance-time data, derived from calculate_slopes().
        target_mean (float): mean dy per segment.
        target_st_dev (float): standard deviation of dys per segment.

    Returns:
        states (list of bool): whether each segment is defined as 'out-of-control'.
    """
    # 1 - Beyond Limits - 1 or more points beyond the control limits (3*sigma)
    # only look at -ve control limit, for decreasing contour length
    states_c1 = [False] * len(segments)
    for j in range(len(segments)):
        if (
            segments['y_diff'].iloc[j] <= target_mean - 3 * target_st_dev
        ):  #  or y_diff >= control_limits[1]:
            states_c1[j] = True

    # 2 - Zone A - 2 out of 3 consecutive points in Zone A or beyond (> 2*sigma)
    is_beyond_zone_a = [
        True if y <= target_mean - 2 * target_st_dev else False for y in segments['y_diff']
    ]
    states_c2 = get_passing_states(is_beyond_zone_a, 2, 1)

    # 3 - Zone B - 4 out of 5 consecutive points in Zone B or beyond (> 1*sigma)
    is_beyond_zone_b = [
        True if y <= target_mean - 1 * target_st_dev else False for y in segments['y_diff']
    ]
    states_c3 = get_passing_states(is_beyond_zone_b, 4, 1)

    # 4 - Zone C - 7 or more consecutive points on one side of the average
    # modified to 6 out of 7 points
    # only look at below the average, for decreasing contour length
    is_below_median = [True if y <= target_mean else False for y in segments['y_diff']]
    states_c4 = get_passing_states(is_below_median, 6, 1)  # 7 in a row -> 7 successes, 0 failures

    # combine states identified this start point
    states = np.any(
        [
            states_c1,
            states_c2,
            states_c3,
            states_c4,
        ],
        0,
    )

    return states


def check_run_conds_slipback(segments, target_mean, target_st_dev):
    """
    Given a set of segmented distance-time data, and the 'in-control' limits defining the
    corresponding Shewhart charts, computes regions which are deemed 'out-of-control' according to
    the Shewhart run conditions.

    Specifically applies Shewhart run conditions defining a retrograde movement (slipback) event.

    Arguments:
        segments (pd.DataFrame): segmented distance-time data, derived from calculate_slopes().
        target_mean (float): mean dy per segment.
        target_st_dev (float): standard deviation of dys per segment.

    Returns:
        states (list of bool): whether each segment is defined as 'out-of-control'.
    """
    # 1 - Beyond Limits - 1 or more points beyond the control limits (3*sigma)
    # only look at +ve control limit, for increasing contour length
    states_c1 = [False] * len(segments)
    for j in range(len(segments)):
        if (
            segments['y_diff'].iloc[j] >= target_mean + 3 * target_st_dev
        ):  #  or y_diff >= control_limits[1]:
            states_c1[j] = True

    # 2 - Zone A - 2 out of 3 consecutive points in Zone A or beyond (> 2*sigma)
    is_beyond_zone_a = [
        True if y >= target_mean + 2 * target_st_dev else False for y in segments['y_diff']
    ]
    states_c2 = get_passing_states(is_beyond_zone_a, 2, 1)

    # 3 - Zone B - 4 out of 5 consecutive points in Zone B or beyond (> 1*sigma)
    is_beyond_zone_b = [
        True if y >= target_mean + 1 * target_st_dev else False for y in segments['y_diff']
    ]
    states_c3 = get_passing_states(is_beyond_zone_b, 4, 1)

    # 4 - Zone C - 7 or more consecutive points on one side of the average
    # only look at below the average, for decreasing contour length
    is_above_median = [True if y >= target_mean else False for y in segments['y_diff']]
    states_c4 = get_passing_states(is_above_median, 7, 0)  # 7 in a row -> 7 successes, 0 failures

    # combine states identified this start point
    states = np.any(
        [
            states_c1,
            states_c2,
            states_c3,
            states_c4,
        ],
        0,
    )

    return states


def do_shewart_assignment(
    data_Lc_regional,
    data_period,
    segment_time,
    start_offsets,
    target_mean,
    target_st_dev,
    molecule=None,
    region=None,
    make_plot=True,
):
    """
    Given a set of constant-distance data, constructs a Shewhart control chart, applies Shewart run
    conditions, and returns detected regions.

    Arguments:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data.
        data_period (float): time gap between consective data points (1 / data frequency) [s].
        segment_time (float): timespan over which to segment distance-time data, to construct
                              Shewhart control chart [s].
        start_offsets (np.array of float): segmentation offsets, used for resampling raw data [s].
        target_mean (float): mean dy per segment measured in -Cdc48 conditions, used to define
                             Shewhart control limits.
        target_st_dev (float): standard deviation of dys per segment measured in -Cdc48 conditions,
                               used to define Shewhart control limits.
        molecule (str): unique ID defining the data file loaded.
        region (int): unique ID defining the region of data selected from the given data file.
        show_plot (bool): whether to generate plots from constructed Shewhart chart.

    Returns:
        fig (plt.Figure): Shewhart chart plot, if generated (None otherwise).
        translocation_regions_overall (list of (tuple of float)):
            time coordinates defining the start/end points of a set all detected translocation
            events [s].
        slipback_regions_overall (list of (tuple of float)):
            time coordinates defining the start/end points of a set all detected retrograde movement
            events [s].
    """

    if make_plot:
        fig, axs = plt.subplots(
            1 + len(start_offsets),
            1,
            sharex=True,
            figsize=(12, 2 * (1 + len(start_offsets))),
        )
        axs[0].plot(
            data_Lc_regional['time'], data_Lc_regional['proteinLc'], color='lightgrey', lw=0.1
        )

        axs[0].set_xlim(
            data_Lc_regional['time'].min(),
            data_Lc_regional['time'].max(),
        )
        axs[0].set_ylabel(r'Protein $L_c$ ($nm$)')

        for ax in axs[:-1]:
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
        axs[-1].set_xlabel(r'Time ($s$)')
        axs[0].set_title(f'Shewart chart: #{molecule}, r. {region}')

    # performs Shewart identification across multiple start indices, and aggregates results
    translocation_states_overall = np.array([False] * len(data_Lc_regional))
    slipback_states_overall = np.array([False] * len(data_Lc_regional))
    for i, start_offset in enumerate(start_offsets):

        # ------------------------------------- get segments ------------------------------------ #
        segment_size = int(segment_time / data_period)
        start_idx = int(start_offset / data_period)

        xs, ys, slopes, intercepts, segment_indices = calculate_slopes(
            data_Lc_regional['proteinLc'].to_numpy(),
            data_Lc_regional['time'].to_numpy(),
            segment_size,
            start_idx
        )
        segments = pd.DataFrame({'x': xs, 'y': ys})
        segments['y_diff'] = segments['y'].diff()

        if make_plot:
            control_lines = [target_mean + i * target_st_dev for i in range(-3, 4, 1)]

            axs[i + 1].plot(segments['x'], segments['y_diff'])

            axs[i + 1].set_ylabel(
                r'$\Delta\ L_c$' + f'\nsegment={segment_time:.1f}s\noffset={start_offset:.1f}s'
            )
            axs[i + 1].set_yticks(control_lines)
            axs[i + 1].set_yticklabels(
                [
                    r'$\mu-3\sigma$',
                    r'$\mu-2\sigma$',
                    r'$\mu-\sigma$,',
                    r'$\mu$',
                    r'$\mu+\sigma$',
                    r'$\mu+2\sigma$',
                    r'$\mu+3\sigma$',
                ]
            )

            for line in control_lines:
                axs[i + 1].axhline(line, color='grey', lw=0.5)

        # ------------------------------- identify translocations ------------------------------- #
        # perform Shewart control chart checks
        states_per_segment = check_run_conds_translocation(segments, target_mean, target_st_dev)

        # convert segment idxs to data_Lc idxs, so selections from different start points can be
        #     combined
        states_per_data_point = upsample_segments_to_data_period(
            states_per_segment, segment_size, start_idx, len(data_Lc_regional)
        )

        # save states identified so far
        translocation_states_overall = np.any(
            [states_per_data_point, translocation_states_overall], 0
        )

        # add identified regions to the plot, if desired
        regions_per_segment = helpers.group_states(states_per_segment)  # group into contiguous regions
        if make_plot:
            for i_lo, i_hi in regions_per_segment:
                axs[i + 1].axvspan(
                    segments['x'].iloc[i_lo - 1],
                    segments['x'].iloc[i_hi - 1],
                    color='blue',
                    alpha=0.1,
                    lw=0,
                    zorder=5,
                )

        # ---------------------------------- identify slipbacks --------------------------------- #
        # perform Shewart control chart checks
        states_per_segment = check_run_conds_slipback(segments, target_mean, target_st_dev)

        # convert segment idxs to data_Lc idxs, so selections from different start points can be
        #     combined
        states_per_data_point = upsample_segments_to_data_period(
            states_per_segment, segment_size, start_idx, len(data_Lc_regional)
        )

        # save states identified so far
        slipback_states_overall = np.any([states_per_data_point, slipback_states_overall], 0)

        # add identified regions to the plot, if desired
        regions_per_segment = helpers.group_states(states_per_segment)
        if make_plot:
            for i_lo, i_hi in regions_per_segment:
                axs[i + 1].axvspan(
                    segments['x'].iloc[i_lo - 1],
                    segments['x'].iloc[i_hi - 1],
                    color='red',
                    alpha=0.1,
                    lw=0,
                    zorder=5,
                )

    # ----------------------------- add ALL identified states to plot --------------------------- #
    translocation_regions_overall = helpers.group_states(translocation_states_overall)
    if make_plot:
        for i_lo, i_hi in translocation_regions_overall:
            axs[0].axvspan(
                data_Lc_regional['time'].iloc[i_lo - 1],
                data_Lc_regional['time'].iloc[i_hi - 1],
                color='blue',
                alpha=0.1,
                lw=0,
                zorder=5,
            )

    slipback_regions_overall = helpers.group_states(slipback_states_overall)
    if make_plot:
        for i_lo, i_hi in slipback_regions_overall:
            axs[0].axvspan(
                data_Lc_regional['time'].iloc[i_lo - 1],
                data_Lc_regional['time'].iloc[i_hi - 1],
                color='orange',
                alpha=0.1,
                lw=0,
                zorder=5,
            )

    return (
        fig if make_plot else None,
        translocation_regions_overall,
        slipback_regions_overall,
    )


def do_max_min_finding(
    data_Lc_regional, regions, event_type, shortest_t, allowed_t_range, allowed_Lc_range
):
    """
    Given a set of detected events, defined by their (rough) start/end coordinates, finds the
    optimal start/end coordinates.

    Does not perform start/end optimisation when a translocation region has been split on a
    slipback event - these events occur in very quick succession, meaning the split coordinate is
    a better estimate of the optimal start/end point than a peak finding algorithm.

    Arguments:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data.
        regions (list of (tuple of float / bool)):
            time coordinates defining the start/end points of a set of detected events [s],
            possibly including boolean values indicated whether the region results from a splitting
            operation.
        event_type (str): type of event being optimised. Must be one of 'translocation' or
                          'slipback'.
        shortest_t (float): minimum event duration, below which will later be filtered out [s].
        allowed_t_range (float): distance (in time) before/after detected region where the algorithm
                                 is allowed to look for more optimal start/end points [s].
        allowed_Lc_range (float): distance (in length) before/after detected region where the
                                  algorithm is allowed to look for more optimal start/end points
                                  [nm].

    Returns:
        regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm]coordinates defining the (optimised) start/end points of a
            set of detected events [s / nm], plus the r-value of the linear regression used to find
            the optimum.
    """
    output_regions = []
    for region in regions:
        if 'translocation' in event_type:
            i_lo, i_hi, truncated_lo, truncated_hi = region
        elif 'slipback' in event_type:
            i_lo, i_hi = region
            truncated_lo, truncated_hi = False, False
    
        t_start = data_Lc_regional['time'].iloc[i_lo - 1]
        t_end = data_Lc_regional['time'].iloc[i_hi - 1]
        t_mid = t_start + (t_end - t_start) / 2

        t_start_compute = t_start if truncated_lo else t_start - allowed_t_range
        t_end_compute = t_end if truncated_hi else t_end + allowed_t_range

        data_Lc_compute_start = data_Lc_regional.loc[
            (data_Lc_regional['time'] >= t_start_compute)
            & (data_Lc_regional['time'] <= t_mid)
        ].reset_index()
        data_Lc_compute_end = data_Lc_regional.loc[
            (data_Lc_regional['time'] >= t_mid)
            & (data_Lc_regional['time'] <= t_end_compute)
        ].reset_index()

        if 'translocation' in event_type:
            Lc_max = data_Lc_compute_start['proteinLc'].max()
            Lc_min = data_Lc_compute_end['proteinLc'].min()
            t_starts = data_Lc_compute_start.loc[
                (data_Lc_compute_start['proteinLc'] >= Lc_max - allowed_Lc_range), 'time'
            ].to_numpy()
            t_ends = data_Lc_compute_end.loc[
                (data_Lc_compute_end['proteinLc'] <= Lc_min + allowed_Lc_range), 'time'
            ].to_numpy()

            dist_key = 'avgLc'

        elif 'slipback' in event_type:
            data_Lc_compute = data_Lc_regional.loc[
                (data_Lc_regional['time'] >= t_start_compute)
                & (data_Lc_regional['time'] <= t_end_compute)
            ].reset_index()

            Lc_max = data_Lc_compute['proteinLc'].max()
            Lc_min = data_Lc_compute['proteinLc'].min()
            t_ends = data_Lc_compute.loc[
                (data_Lc_compute['proteinLc'] >= Lc_max - allowed_Lc_range), 'time'
            ].to_numpy()
            t_starts = data_Lc_compute.loc[
                (data_Lc_compute['proteinLc'] <= Lc_min + allowed_Lc_range), 'time'
            ].to_numpy()

            dist_key = 'avgLc_slipback'

        else:
            raise ValueError(
                f"Incorrect event_type '{event_type}' provided. Must be one of 'translocation' or "
                "'slipback'."
            )

        # get all combinations of t_start / t_end, as long as t_start is before t_end
        t_pairs = [
            (t_start, t_end)
            for t_start, t_end in itertools.product(t_starts, t_ends)
            if (t_end - t_start) >= shortest_t
        ]

        # find best fit
        best_r = None
        best_t_start = None
        best_t_end = None
        for t_start, t_end in t_pairs:
            data_Lc_selected = data_Lc_regional.loc[
                (data_Lc_regional['time'] >= t_start) & (data_Lc_regional['time'] <= t_end)
            ]

            slope, intercept, r, _, _ = stats.linregress(
                data_Lc_selected['time'], data_Lc_selected['proteinLc']
            )
            r = abs(r)

            if best_r is None:
                best_r = r
                best_t_start = t_start
                best_t_end = t_end

            elif r > best_r:
                best_r = r
                best_t_start = t_start
                best_t_end = t_end

        if best_r is not None:
            Lc_start = data_Lc_regional.loc[
                data_Lc_regional['time'] == best_t_start, dist_key
            ].iloc[0]
            Lc_end = data_Lc_regional.loc[data_Lc_regional['time'] == best_t_end, dist_key].iloc[0]
            output_regions += [((best_t_start, Lc_start), (best_t_end, Lc_end), best_r)]

    return output_regions


def filter_short_regions(regions, event_type, shortest_t, shortest_Lc, slowest_v, debug):
    """
    Given a set of detected events, filters out events which are too short or quick to be reliable.

    Arguments:
        regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected events, with r-value.
        event_type (str): type of event being optimised. Must be one of 'translocation' or
                          'slipback'.
        shortest_t (float): minimum event duration, below which will be filtered out [s].
        shortest_Lc (float): minimum event length, below which will be filtered out [nm].
        slowest_v (float): minimum event velocity, below which will be filtered out [nm / s].
        debug (bool): use function in debug mode, i.e. with more verbose console output.

    Returns:
        regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm] coordinates defining the (filtered) start/end points of a
            set of detected events, with r-value.
    """
    output_regions = []

    if debug:
        print(f'\n{event_type}')

    for i, ((t_start, Lc_start), (t_end, Lc_end), r) in enumerate(regions):

        delta_Lc = abs(Lc_start - Lc_end)
        delta_t = t_end - t_start
        velocity = delta_Lc / delta_t

        if delta_t >= shortest_t and delta_Lc >= shortest_Lc and velocity >= slowest_v:
            if debug:
                print(
                    f'{t_start:.1f}-{t_end:.1f}s - pass (v={velocity:.3f}, dt={delta_t:.3f}, '
                    f'dLc={delta_Lc})'
                )
            output_regions += [((t_start, Lc_start), (t_end, Lc_end), r)]

        elif debug:
            print(
                f'{t_start:.1f}-{t_end:.1f}s - fail (v={velocity:.3f}, dt={delta_t:.3f}, '
                f'dLc={delta_Lc})'
            )
    
    return output_regions


def do_edge_optimisation(
    data_Lc_regional,
    event_regions,
    event_type,
    shortest_t,
    shortest_Lc,
    slowest_v,
    allowed_t_range,
    allowed_Lc_range,
    molecule,
    region,
    debug,
):
    """
    Given a set of detected events, defined by their (rough) start/end coordinates, finds the
    optimal start/end coordinates and filters out events which are too short or too quick to be
    relied on.

    Arguments:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data.
        event_regions (list of (tuple of float / bool)):
            time coordinates defining the start/end points of a set of detected events [s],
            possibly including boolean values indicated whether the region results from a splitting
            operation.
        event_type (str): type of event being optimised. Must be one of 'translocation' or
                          'slipback'.
        shortest_t (float): minimum event duration, below which will later be filtered out [s].
        shortest_Lc (float): minimum event length, below which will be filtered out [nm].
        slowest_velocity (float): minimum event velocity, below which will be filtered out [nm / s].
        allowed_t_range (float): distance (in time) before/after detected region where the algorithm
                                 is allowed to look for more optimal start/end points [s].
        allowed_Lc_range (float): distance (in length) before/after detected region where the
                                  algorithm is allowed to look for more optimal start/end points
                                  [nm].
        molecule (str): unique ID defining the data file loaded.
        region (int): unique ID defining the region of data selected from the given data file.
        debug (bool): use function in debug mode, i.e. with more verbose console output.

    Returns:
        event_regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected events, with r-value.
    """

    # do edge optimisation
    event_regions = do_max_min_finding(
        data_Lc_regional,
        event_regions,
        event_type,
        shortest_t,
        allowed_t_range,
        allowed_Lc_range,
    )

    # filter out regions which are too short (in time or length)
    if len(event_regions) > 0:
        event_regions = filter_short_regions(
            event_regions,
            event_type,
            shortest_t,
            shortest_Lc,
            slowest_v,
            debug,
        )

    return event_regions


def merge_overlapping_regions(event_regions):
    """
    Given a set of detected events, merges events which overlap in time.

    Arguments:
        regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected events, with r-value.
        event_type (str): type of event being optimised. Must be one of 'translocation' or
                          'slipback'.
        shortest_t (float): minimum event duration, below which will be filtered out [s].
        shortest_Lc (float): minimum event length, below which will be filtered out [nm].
        slowest_velocity (float): minimum event velocity, below which will be filtered out [nm / s].
        debug (bool): use function in debug mode, i.e. with more verbose console output.

    Returns:
        regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm] coordinates defining the (merged) start/end points of a
            set of detected events, with r-value.
    """

    # early break if there aren't enough events for there to be any overlaps
    if len(event_regions) <= 1:
        return event_regions

    # else ...
    output_regions = []
    if len(event_regions[0]) == 3:  # form (t_start, Lc_start), (t_end, Lc_end) - translocations
        ((t_start, Lc_start), (t_end, Lc_end), r) = event_regions[0]
        for (this_t_start, this_Lc_start), (this_t_end, this_Lc_end), r in event_regions[1:]:
            region_overlaps = this_t_start < t_end
            if region_overlaps:
                t_end = this_t_end
                Lc_end = this_Lc_end
            else:
                output_regions += [((t_start, Lc_start), (t_end, Lc_end), r)]
                t_start = this_t_start
                t_end = this_t_end
                Lc_start = this_Lc_start
                Lc_end = this_Lc_end

        output_regions += [((t_start, Lc_start), (t_end, Lc_end), r)]

    elif len(event_regions[0]) == 2:  # form t_start, t_end - slipbacks
        (t_start, t_end) = event_regions[0]
        for this_t_start, this_t_end in event_regions[1:]:
            region_overlaps = this_t_start < t_end
            if region_overlaps:
                t_end = this_t_end
            else:
                output_regions += [(t_start, t_end)]
                t_start = this_t_start
                t_end = this_t_end

        output_regions += [(t_start, t_end)]

    else:
        raise ValueError('Unknown region format.')

    return output_regions


def remove_translocation_overlaps(translocation_regions, slipback_regions):
    """
    Given a set of translocation of slipback events, removes sections of slipbacks which overlap
    with a translocation.

    Arguments:
        translocation_regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected translocations, with r-value.
        slipback_regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected slipbacks, with r-value.

    Returns:
        slipback_regions (list of (tuple of (tuple of float) / float)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected slipbacks, with r-value, ensuring that none overlap a translocation.
    """

    # early break if there aren't enough events for there to be any overlaps
    if len(translocation_regions) == 0 or len(slipback_regions) == 0:
        return slipback_regions

    # else ...
    # for remaining events, adjust start/end points to not overlap translocation
    # TODO this is really ugly, fix

    output_regions = []
    for (t_start, Lc_start), (t_end, Lc_end), r in slipback_regions:
        add_event = True
        for (translocation_t_start, translocation_Lc_start), (
            translocation_t_end,
            translocation_Lc_end,
        ), r in translocation_regions:
            starts_mid_translocation = (t_start > translocation_t_start) and (
                t_start < translocation_t_end
            )
            ends_mid_translocation = (t_end > translocation_t_start) and (
                t_end < translocation_t_end
            )

            if starts_mid_translocation and ends_mid_translocation:
                add_event = False
                break

            if starts_mid_translocation:
                t_start = translocation_t_end
                Lc_start = translocation_Lc_end

            if ends_mid_translocation:
                t_end = translocation_t_start
                Lc_end = translocation_Lc_start

        if add_event:
            output_regions += [((t_start, Lc_start), (t_end, Lc_end), r)]

    return output_regions


def make_event_plots(
    data_Lc_regional,
    translocation_regions,
    slipback_regions,
    molecule,
    region,
    data_path,
    save_plot,
    show_plot,
):
    """
    Generates a plot for of each detected translocation and slipback event.

    Arguments:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data.
        translocation_regions (list of (tuple of float / bool)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected translocation events, with r-value.
        slipback_regions (list of (tuple of float / bool)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected slipback events, with r-value.
        molecule (str): unique ID defining the data file loaded.
        region (int): unique ID defining the region of data selected from the given data file.
        data_path (str): relative path where plots should be saved.
        show_plot (bool): whether to display generated plots to the user.
        save_plot (bool): whether to save generated plots to file.
    """
    
    translocation_times = np.array(
        [[[t_start, t_end] for (t_start, _), (t_end, _), _ in translocation_regions]]
    ).flat
    translocation_lengths = np.array(
        [[[Lc_start, Lc_end] for (_, Lc_start), (_, Lc_end), _ in translocation_regions]]
    ).flat
    slipback_times = np.array(
        [[[t_start, t_end] for (t_start, _), (t_end, _), _ in slipback_regions]]
    ).flat
    slipback_lengths = np.array(
        [[[Lc_start, Lc_end] for (_, Lc_start), (_, Lc_end), _ in slipback_regions]]
    ).flat

    fig_all, ax_all = plt.subplots(figsize=(12, 4))
    fig_event, ax_event = plt.subplots(figsize=(4, 4))
    for ax in [ax_all, ax_event]:
        ax.plot(data_Lc_regional['time'], data_Lc_regional['proteinLc'], color='lightgrey', lw=0.2)
        for ((t_start, _), (t_end, _), _) in translocation_regions:
            ax.axvspan(t_start, t_end, color='b', alpha=0.1, lw=0, zorder=5)
        for ((t_start, _), (t_end, _), _) in slipback_regions:
            ax.axvspan(t_start, t_end, color='r', alpha=0.1, lw=0, zorder=5)

        if len(translocation_regions) > 0:
            ax.scatter(
                translocation_times, translocation_lengths, color='b', alpha=0.3, s=3, zorder=5
            )
        if len(slipback_regions) > 0:
            ax.scatter(slipback_times, slipback_lengths, color='r', alpha=0.3, s=3, zorder=5)

        ax.set_xlim(data_Lc_regional['time'].min(), data_Lc_regional['time'].max())
        ax.set_xlabel(r'Time ($s$)')
        ax.set_ylabel(r'Protein $L_c$ ($nm$)')
        ax.set_title(f'#{molecule}, r. {region}, all translocations')

    plt.tight_layout()

    if save_plot:
        fig_all.savefig(
            rf'{data_path}/automated_analysis/translocations/{molecule}_r{region}_all.png',
            format='png',
            dpi=300,
        )
    if show_plot:
        plt.show()

    plt.close(fig_all)

    if save_plot:
        t_total, Lc_total = 8, 200
        if len(translocation_regions) > 0:
            for i, ((t_start, Lc_start), (t_end, Lc_end), r) in enumerate(translocation_regions):
                delta_t = t_end - t_start
                delta_Lc = Lc_start - Lc_end
                t_pad = (t_total - delta_t) / 2
                Lc_pad = (Lc_total - delta_Lc) / 2

                ax_event.set_xlim(t_start - t_pad, t_end + t_pad)
                ax_event.set_ylim(Lc_end - Lc_pad, Lc_start + Lc_pad)
                ax_event.set_title(f'#{molecule}, r. {region}, translocation #{i}')
                plt.tight_layout()
                fig_event.savefig(
                    rf'{data_path}/automated_analysis/translocations/{molecule}_r{region}_{i}.png',
                    format='png',
                    dpi=300,
                )

        if len(slipback_regions) > 0:
            for i, ((t_start, Lc_start), (t_end, Lc_end), r) in enumerate(slipback_regions):
                delta_t = t_end - t_start
                delta_Lc = Lc_end - Lc_start
                t_pad = (t_total - delta_t) / 2
                Lc_pad = (Lc_total - delta_Lc) / 2

                ax_event.set_xlim(t_start - t_pad, t_end + t_pad)
                ax_event.set_ylim(Lc_start - Lc_pad, Lc_end + Lc_pad)
                ax_event.set_title(f'#{molecule}, r. {region}, slipback #{i}')
                plt.tight_layout()
                fig_event.savefig(
                    rf'{data_path}/automated_analysis/slipbacks/{molecule}_r{region}_{i}.png',
                    format='png',
                    dpi=300,
                )

    plt.close(fig_event)

    return


def build_event_log(
    data_Lc_regional, molecule, region, translocation_regions, slipback_regions, input_event_log
):
    """
    Computes durations and total lengths of each detected event, and adds all data to an event log
    file for further analysis.

    Arguments:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data.
        translocation_regions (list of (tuple of float / bool)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected translocation events, with r-value.
        slipback_regions (list of (tuple of float / bool)):
            time- [s] and length- [nm] coordinates defining the start/end points of a set of
            detected slipback events, with r-value.
        molecule (str): unique ID defining the data file loaded.
        region (int): unique ID defining the region of data selected from the given data file.
        input_event_log (pd.DataFrame): existing event log, which newly-determined events will be
                                        appended to.

    Returns:
        event_log (pd.DataFrame): updated event log, containing new data.
    """
    # save to dataframe
    event_log_regional = pd.DataFrame(columns=input_event_log.columns)
    region_t_start = data_Lc_regional['time'].iloc[0]
    region_t_end = data_Lc_regional['time'].iloc[-1]
    region_delta_t = region_t_end - region_t_start

    # combine translocations
    if len(translocation_regions) > 0:
        for i, ((t_start, Lc_start), (t_end, Lc_end), r) in enumerate(translocation_regions):
            event_log_regional = pd.concat(
                [
                    event_log_regional,
                    pd.DataFrame(
                        {
                            'molecule': [molecule],
                            'region_id': [region],
                            'region_t_start': [region_t_start],
                            'region_t_end': [region_t_end],
                            'region_delta_t': [region_delta_t],
                            'event_type': ['translocation'],
                            'event_id': [i],
                            't_start': [t_start],
                            't_end': [t_end],
                            'delta_t': [t_end - t_start],
                            'Lc_start': [Lc_start],
                            'Lc_end': [Lc_end],
                            'delta_Lc': [Lc_start - Lc_end],
                            'r': [r],
                        }
                    ),
                ]
            )
    else:
        event_log_regional = pd.concat(
            [
                event_log_regional,
                pd.DataFrame(
                    {
                        'molecule': [molecule],
                        'region_id': [region],
                        'region_t_start': [region_t_start],
                        'region_t_end': [region_t_end],
                        'region_delta_t': [region_delta_t],
                        'event_type': ['no-translocations'],
                        'event_id': [None],
                        't_start': [None],
                        't_end': [None],
                        'delta_t': [None],
                        'Lc_start': [None],
                        'Lc_end': [None],
                        'delta_Lc': [None],
                        'r': [None],
                    }
                ),
            ]
        )

    if len(slipback_regions) > 0:
        for i, ((t_start, Lc_start), (t_end, Lc_end), r) in enumerate(slipback_regions):
            event_log_regional = pd.concat(
                [
                    event_log_regional,
                    pd.DataFrame(
                        {
                            'molecule': [molecule],
                            'region_id': [region],
                            'region_t_start': [region_t_start],
                            'region_t_end': [region_t_end],
                            'region_delta_t': [region_delta_t],
                            'event_type': ['slipback'],
                            'event_id': [i],
                            't_start': [t_start],
                            't_end': [t_end],
                            'delta_t': [t_end - t_start],
                            'Lc_start': [Lc_start],
                            'Lc_end': [Lc_end],
                            'delta_Lc': [Lc_end - Lc_start],
                            'r': [r],
                        }
                    ),
                ]
            )
    else:
        event_log_regional = pd.concat(
            [
                event_log_regional,
                pd.DataFrame(
                    {
                        'molecule': [molecule],
                        'region_id': [region],
                        'region_t_start': [region_t_start],
                        'region_t_end': [region_t_end],
                        'region_delta_t': [region_delta_t],
                        'event_type': ['no-slipbacks'],
                        'event_id': [None],
                        't_start': [None],
                        't_end': [None],
                        'delta_t': [None],
                        'Lc_start': [None],
                        'Lc_end': [None],
                        'delta_Lc': [None],
                        'r': [None],
                    }
                ),
            ]
        )

    event_log_regional = event_log_regional.sort_values('t_start')

    return pd.concat([input_event_log, event_log_regional])
