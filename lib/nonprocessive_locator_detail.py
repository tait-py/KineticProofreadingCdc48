import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

import helpers


def compute_noise(data_Lc, data_period, noise_window, alignment='centre'):
    """
    Given a set of constant-distance data, computes the noise signal.

    Arguments:
        data_Lc_regional (str): experimental data corresponding to one contiguous region of
                                constant-distance data.
        data_period (float): time gap between consective data points (1 / data frequency) [s].
        noise_window (float): width of moving window used to calculate noise [s].
        alignment (str): where to align the noise calculation, relative to the moving window. Must
                         be one of 'left', 'right' or 'centre'
    Returns:
        data_Lc_regional (str): experimental data corresponding to one contiguous region of
                                constant-distance data, including computed noise signal.
    """
    noise_window_n = round(noise_window / data_period)

    if alignment == 'left':
        shift_factor = -noise_window_n
    elif alignment == 'right':
        shift_factor = 0
    elif alignment == 'centre':
        shift_factor = round(-noise_window_n / 2)
    else:
        raise ValueError('alignment must be one of "left", "right" or "centre"')

    data_Lc['noise'] = (
        data_Lc['proteinLc']
        .rolling(noise_window_n)
        .apply(lambda x: np.std(x.iloc[:noise_window_n]))
        .shift(shift_factor)
    )

    return data_Lc


def load_data(
    filepath, filename, data_period, savgol_window, peak_heights, region_t_min, noise_window
):
    """
    Given the path to an experimental data file, loads the raw data and determines all valid
    regions of constant-distance data. Additionally

    Arguments:
        file_path (str): relative path to the directory containing all experimental data files.
        filename (str): name of experimental data file to be loaded.
        data_period (float): time gap between consective data points (1 / data frequency) [s].
        savgol_window (int): window length of Savitzky-Golay filter used to extract regions of
                             constant-distance data.
        peak_heights (tuple of float): (min, max) measured length which defines a valid region of
                                       constant-distance data [nm].
        region_t_min (float): minimum duration of a valid region of constant-distance data [s].
        noise_window (float): width of moving window used to calculate noise [s].

    Returns:
        data_Lc_output (pd.DataFrame): loaded experimental data file, filtered to contain valid
                                       constant-distance data only, with computed noise.
        molecule (str): unique ID defining the data file loaded.
    """
    # get data file ID information, to differentiate between files
    # TWOM data filenames are in the format:
    #     '[date]-[time] [description] #[bead pair id]-[molecule id].tdms'
    # therefore, full id is between '#' and '.'
    molecule = filename.split('#')[1][:-4]

    # noise data computation takes a long time - load from file, if possible
    try:
        data_Lc = pd.read_csv(rf'{filepath}/noise/noise_{molecule}_{noise_window:.1f}s.csv')
    except FileNotFoundError:
        print('re-computing noise')
        data_Lc = pd.read_csv(f'{filepath}/wlc_manual_fit/{filename}', index_col=0)
        data_Lc = compute_noise(data_Lc, data_period, noise_window, 'centre')
        data_Lc.to_csv(rf'{filepath}/noise/noise_{molecule}_{noise_window:.1f}s.csv')
        print('finished')

    data_Lc['avgLc'] = signal.savgol_filter(data_Lc['proteinLc'], savgol_window, 1)

    # Find regions of constant-distance measurement in data file
    # In the raw data, 'region' refers to the measurement state (pulling, stationary, etc.),
    #     whereas later I redefine it as the index column for marking continuous regions of valid
    #     measurement data -- a similar purpose, but not identical
    # First, initialise an 'is_signal' column with zeros, but change to '1' if the Lc value is
    #     within the measurement range
    data_Lc['is_signal'] = 0
    data_Lc.loc[
        (data_Lc['proteinLc'] > peak_heights[0]) & (data_Lc['proteinLc'] < peak_heights[1]),
        'is_signal',
    ] = 1

    # Find regions of constant-distance measurement in data file
    # In the raw data, 'region' refers to the measurement state (pulling, stationary, etc.),
    #     whereas later I redefine it as the index column for marking continuous regions of valid
    #     measurement data -- a similar purpose, but not identical
    # First, initialise an 'is_signal' column with zeros, but change to '1' if the Lc value is
    #     within the measurement range
    data_Lc['region'] = data_Lc['is_signal'].diff().ne(0).cumsum()

    region_n_min = int(region_t_min / data_period)
    output_columns = ['time', 'forceX', 'proteinLc', 'avgLc', 'region', 'noise']
    data_Lc_output = pd.DataFrame(columns=output_columns)
    new_region = 1
    for region in list(set(data_Lc['region'])):
        data_in_region = data_Lc.loc[
            (data_Lc['region'] == region) & (data_Lc['is_signal'] == 1)
        ].copy()
        if len(data_in_region) >= region_n_min:
            data_in_region['region'] = new_region
            new_region += 1
            data_Lc_output = pd.concat([data_Lc_output, data_in_region[output_columns]])

    return data_Lc_output, molecule


def load_stepfit(data_path, data_Lc, molecule, region):
    """
    Given the path to a set of AutoStepfinder output data, loads the correct fit for a given
    molecule ID and computes the contiguous regions between (un)folding events.

    Arguments:
        data_path (str): relative path to the directory containing experimental data.
        data_Lc (pd.DataFrame): experimental data corresponding to one contiguous region of
                                constant-distance data.
        molecule (str): unique ID defining the data file loaded.
        region (int): unique ID defining the region of data selected from the given data file.

    Returns:
        step_regions (list of (tuple of int)): (start, end) indexes of contiguous data with no
                                               (un)folding.
    """
    matching_stepfit_filenames = [
        f
        for f in os.listdir(rf'{data_path}/autostepfinder/StepFit_Result')
        if f'_{molecule}_r{region}_fits.txt' in f
    ]

    if len(matching_stepfit_filenames) > 1:
        raise ValueError('multiple stepfit files identified')
    elif len(matching_stepfit_filenames) == 1:
        steps_data = pd.read_csv(
            f'{data_path}/autostepfinder/StepFit_Result/{matching_stepfit_filenames[0]}'
        )
        step_regions = helpers.group_states(steps_data['FinalFit'])
    else:
        step_regions = [[0, len(data_Lc)]]

    return step_regions


def compute_noise_enrichment(data_Lc_regional, step_regions, noise_window_n):
    """
    Given constant-distance data with noise signal and a set of regions between (un)folding steps,
    normalises the noise signal within each (un)folding region.

    Arguments:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data.
        step_regions (list of (tuple of int)): (start, end) indexes of contiguous data with no
                                               (un)folding.
        noise_window_n (int): width of moving window used to calculate noise [# data points].

    Returns:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data, with a normalised noise signal.
    """
    # compute noise enrichment (i.e. normalise noise values by their median)
    data_Lc_regional['noise'] = data_Lc_regional['noise'] ** 2
    lower_95 = np.array(
        sorted(data_Lc_regional['noise'].dropna().to_numpy())[
            : int(len(data_Lc_regional) * 0.95)
        ]
    )
    data_Lc_regional['noise'] = data_Lc_regional['noise'] / np.median(lower_95)

    # split data on (un)folding events identified by AutoStepfinder

    data_Lc_regional['step_median'] = None
    for i_start, i_end in step_regions:
        step_data = data_Lc_regional.loc[i_start : (i_end - 1), 'noise'].dropna().to_numpy()

        # re-compute median, to align 3sigma windows (not to rescale data, as earlier)
        lower_20 = np.array(sorted(step_data)[: int(len(step_data) * 0.20)])
        data_Lc_regional.loc[i_start : (i_end - 1), 'step_median'] = np.median(lower_20)

        # remove noise measured during folding events (will be detected by a simple noise cutoff
        #     when they shouldn't be)
        data_Lc_regional.loc[
            i_start - (noise_window_n // 2) : (i_start + noise_window_n // 2), 'noise'
        ] = None
        data_Lc_regional.loc[
            i_end - (noise_window_n // 2) : (i_end + noise_window_n // 2), 'noise'
        ] = None

    return data_Lc_regional


def get_target_stats(
    data_path,
    data_period,
    savgol_window,
    peak_heights,
    region_t_min,
    noise_window,
    **kwargs,
):
    """
    Given a set of raw tweezers datafiles, detects regions of constant-distance data and computes
    noise signal, to calculate standard deviation of this signal.

    Used to compute the 'in-control' standard deviation in control (-Cdc48) conditions.

    Arguments:
        data_path (str): relative path to the directory containing experimental (control) data.
        data_period (float): time gap between consective data points (1 / data frequency) [s].
        savgol_window (int): window length of Savitzky-Golay filter used to extract regions of
                             constant-distance data.
        peak_heights (tuple of float): (min, max) measured length which defines a valid region of
                                       constant-distance data [nm].
        region_t_min (float): minimum duration of a valid region of constant-distance data [s].
        noise_window (float): width of moving window used to calculate noise [s].

    Returns:
        target_st_dev (float): standard deviation of noise signal.
    """

    manual_region_lims = helpers.load_manual_region_lims(data_path)
    filenames = [f for f in os.listdir(f'{data_path}/wlc_manual_fit') if '.csv' in f]

    noises = np.array([])
    for filename in filenames:
        data_Lc, molecule = load_data(
            data_path,
            filename,
            data_period=data_period,
            savgol_window=savgol_window,
            peak_heights=peak_heights,
            region_t_min=region_t_min,
            noise_window=noise_window,
        )

        if len(data_Lc) == 0:  # i.e. no valid regions
            continue

        for region in data_Lc['region'].unique():
            data_Lc_regional = data_Lc.loc[data_Lc['region'] == region]

            # some data files have erroneous errors - manually remove these if present
            data_Lc_regional = helpers.apply_manual_region_limits(
                data_Lc_regional, manual_region_lims[molecule][str(region)]
            )

            # compute noise enrichment (i.e. normalise noise values by their median)
            data_Lc_regional['noise'] = data_Lc_regional['noise'] ** 2
            lower_95 = np.array(
                sorted(data_Lc_regional['noise'].dropna().to_numpy())[
                    : int(len(data_Lc_regional) * 0.95)
                ]
            )
            data_Lc_regional['noise'] = data_Lc_regional['noise'] / np.median(lower_95)

            # take sample std as the lowest 95% (to avoid massive signals during folding events)
            lower_95 = np.array(
                sorted(data_Lc_regional['noise'].dropna().to_numpy())[
                    : int(len(data_Lc_regional) * 0.95)
                ]
            )

            # save computed noises, to calculate std later
            noises = np.append(noises, lower_95)
    
    return np.std(noises)


def merge_nearby_regions(event_regions, noise_window_n):
    """
    Given a set of detected events, merges events which are sufficiently close in time. Merges
    events which are within one noise window, as these cannot be reliably distinguished by the
    moving window.

    Arguments:
        event_regions (list of ((tuple of float)): time-coordinates defining the start/end points
                                                   of a set of detected events [s].
        noise_window_n (int): width of moving window used to calculate noise [# data points].

    Returns:
        event_regions (list of ((tuple of float)): time-coordinates defining the start/end points
                                                   of a set of detected events [s], where nearby
                                                   events are merged.
    """
    # early break if there aren't enough events for there to be any overlaps
    if len(event_regions) <= 1:
        return event_regions

    # else ...
    output_regions = []
    (i_start, i_end) = event_regions[0]
    for this_i_start, this_i_end in event_regions[1:]:
        region_nearby = (this_i_start - i_end) < noise_window_n
        if region_nearby:
            i_end = this_i_end
        else:
            output_regions += [(i_start, i_end)]
            i_start = this_i_start
            i_end = this_i_end

    output_regions += [(i_start, i_end)]

    return output_regions


def make_event_plots(
    data_Lc_regional,
    nonprocessive_regions,
    noise_window,
    target_st_dev,
    molecule,
    region,
    data_path,
    save_plot,
    show_plot,
):
    """
    Generates a plot for of each non-processive event.

    Arguments:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data.
        nonprocessive_regions (list of ((tuple of float)):
            time-coordinates defining the start/end points of a set of detected events [s].
        noise_window (float): width of moving window used to calculate noise [s].
        target_st_dev (float): standard deviation of (in-control) noise signal.
        molecule (str): unique ID defining the data file loaded.
        region (int): unique ID defining the region of data selected from the given data file.
        data_path (str): relative path where plots should be saved.
        show_plot (bool): whether to display generated plots to the user.
        save_plot (bool): whether to save generated plots to file.
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    axs[0].plot(data_Lc_regional['time'], data_Lc_regional['proteinLc'], color='lightgrey', lw=0.2)
    axs[0].plot(data_Lc_regional['time'], data_Lc_regional['avgLc'], color='k', lw=0.5)
    for i_start, i_end in nonprocessive_regions:
        axs[0].axvspan(
            data_Lc_regional['time'].iloc[i_start],
            data_Lc_regional['time'].iloc[i_end],
            color='r',
            alpha=0.3,
            lw=0,
            zorder=5,
        )

    axs[0].set_xlim(data_Lc_regional['time'].min(), data_Lc_regional['time'].max())
    axs[0].tick_params(axis='x', bottom=False, labelbottom=False)
    axs[0].set_ylabel(r'$L_c$ ($nm$)')

    axs[1].plot(data_Lc_regional['time'], data_Lc_regional['noise'], color='orange', lw=0.5)

    for i in range(-3, 4, 1):
        axs[1].plot(
            data_Lc_regional['time'],
            data_Lc_regional['step_median'] + i * target_st_dev,
            lw=0.2,
            color='grey',
        )

    axs[1].set_xlabel(r'Time ($s$)')
    axs[1].set_ylabel(rf'$\sigma^2$ ($w=${noise_window}$s$)')
    axs[1].set_ylim(0, 10)

    plt.tight_layout()
    if save_plot:
        fig.savefig(
            rf'{data_path}/automated_analysis/{molecule}_r{region}_all.png',
            format='png',
            # DPI=300,
        )
    if show_plot:
        plt.show()

    plt.close(fig)

    return


def build_event_log(
    data_Lc_regional,
    nonprocessive_regions,
    molecule,
    region,
    input_event_log,
):
    """
    Computes durations of each detected event, and adds all data to an event log file for further
    analysis.

    Arguments:
        data_Lc_regional (pd.DataFrame): experimental data corresponding to one contiguous region
                                         of constant-distance data.
        nonprocessive_regions (list of ((tuple of float)):
            time-coordinates defining the start/end points of a set of detected events [s].
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

    if len(nonprocessive_regions) > 0:
        for i, (i_start, i_end) in enumerate(nonprocessive_regions):
            t_start = data_Lc_regional.loc[i_start, 'time']
            t_end = data_Lc_regional.loc[i_end, 'time']

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
                            'event_type': ['nonprocessive'],
                            'event_id': [i],
                            't_start': [t_start],
                            't_end': [t_end],
                            'delta_t': [t_end - t_start],
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
                        'event_type': ['no-nonprocessives'],
                        'event_id': [None],
                        't_start': [None],
                        't_end': [None],
                        'delta_t': [None],
                    }
                ),
            ]
        )

    return pd.concat([input_event_log, event_log_regional])
