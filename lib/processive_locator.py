import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import helpers
import settings
import processive_locator_detail as detail


def processive_locator(
    data_path,
    target_mean,
    target_st_dev,
    data_period,
    savgol_window_tl,
    savgol_window_sb,
    peak_heights,
    region_t_min,
    segment_time_shewart,
    n_resamples,
    tl_shortest_t,
    tl_shortest_Lc,
    tl_slowest_v,
    tl_allowed_t_range,
    tl_allowed_Lc_range,
    sb_shortest_t,
    sb_shortest_Lc,
    sb_slowest_v,
    sb_allowed_t_range,
    sb_allowed_Lc_range,
    show_plot,
    save_plot,
    debug,
    **kwargs,
):
    """
    Given a set of raw tweezers datafiles, computes all regions corresponding to processive
    translocation and retrograde movement by Cdc48, and saves log of event data to file.

    Event detection is based on the Shewhart control chart paradigm[1-3], described in detail in
    the Methods section.

    Arguments:
        data_path (str): relative path to the directory containing experimental data.
        target_mean (float): mean dy per segment measured in -Cdc48 conditions, used to define
                             Shewhart control limits.
        target_st_dev (float): standard deviation of dys per segment measured in -Cdc48 conditions,
                               used to define Shewhart control limits.
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
        tl_shortest_t (float): minimum translocation duration [s].
        tl_shortest_Lc (float): minimum translocation length [nm].
        tl_slowest_v (float): minimum translocation velocity [nm / s].
        tl_allowed_t_range (float):
            distance (in time) before/after detected translocation regions where the algorithm is
            allowed to look for more optimal start/end points [s].
        tl_allowed_Lc_range (float):
            distance (in length) before/after detected translocation regions where the algorithm is
            allowed to look for more optimal start/end points [nm].
        sb_shortest_t (float): minimum slipback duration [s].
        sb_shortest_Lc (float): minimum slipback length [nm].
        sb_slowest_v (float): minimum slipback velocity [nm / s].
        sb_allowed_t_range (float):
            distance (in time) before/after detected slipback regions where the algorithm is
            allowed to look for more optimal start/end points [s].
        sb_allowed_Lc_range (float):
            distance (in length) before/after detected slipback regions where the algorithm is
            allowed to look for more optimal start/end points [nm].
        show_plot (bool): whether to display generated plots to the user.
        save_plot (bool): whether to save generated plots to file.
        debug (bool): use function in debug mode, i.e. with more verbose console output.

    References:
        [1] W. A. Shewhart, Economic control of quality of manufactured product. (D. Van Nostrand
            Company, Inc, New York, 1931).
        [2] C.-I. Li, N.-C. Su, P.-F. Su, Y. Shyr, The design of and R control charts for skew
            normal distributed data. Commun. Stat. Theory Methods 43, 4908-4924 (2014).
        [3] Western Electric Company, Statistical Quality Control Handbook. (Western Electric
            Company, Indianapolis, 1956).
    """

    manual_region_lims = helpers.load_manual_region_lims(data_path)

    event_log = pd.DataFrame(columns=[
        'molecule',
        'region_id',
        'region_t_start',
        'region_t_end',
        'region_delta_t',
        'event_type',
        'event_id',
        't_start',
        't_end',
        'delta_t',
        'Lc_start',
        'Lc_end',
        'delta_Lc',
        'r',
    ])

    if not show_plot:  # more efficient if plots are not required
        matplotlib.use('Agg')

    # collect names of all raw data files
    filenames = [
        f for f in os.listdir(f'{data_path}/wlc_manual_fit') if '.csv' in f and 'IGNORE' not in f
    ]

    # verify that no data files have duplicate IDs
    helpers.verify_duplicate_molecules(filenames)

    n_files = len(filenames)
    start_offsets = np.linspace(0, segment_time_shewart, n_resamples)
    for file_count, filename in enumerate(filenames):

        data_Lc, molecule = detail.load_data(
            f'{data_path}/wlc_manual_fit/{filename}',
            data_period=data_period,
            savgol_window_tl=savgol_window_tl,
            savgol_window_sb=savgol_window_sb,
            peak_heights=peak_heights,
            t_min=region_t_min,
        )

        print(f'Processing #{molecule} ({file_count + 1}/{n_files}).')

        if len(data_Lc) == 0:  # i.e. no valid regions
            continue
        
        # detect events for each valid region of constant-distance data
        for region in list(set(data_Lc['region'])):
            data_Lc_regional = data_Lc.loc[data_Lc['region'] == region]

            # some data files have erroneous errors - manually remove these if present
            data_Lc_regional = helpers.apply_manual_region_limits(
                data_Lc_regional, manual_region_lims[molecule][str(region)]
            )

            print(
                f'Processing #{molecule} ({file_count + 1}/{n_files}) - starting Shewart '
                'identification.'
            )

            # construct Shewart control chart, and detect out-of-control regions
            (fig_shewart, translocation_regions, slipback_regions) = detail.do_shewart_assignment(
                data_Lc_regional,
                data_period=data_period,
                segment_time=segment_time_shewart,
                start_offsets=start_offsets,
                target_mean=target_mean,
                target_st_dev=target_st_dev,
                molecule=molecule,
                region=region,
                make_plot=(show_plot or save_plot),
            )

            # show or save Shewart chart (if desired)
            if show_plot or save_plot:
                fig_shewart.subplots_adjust(left=0.1, bottom=0.05, right=0.99, top=0.97, hspace=0.2)
                if save_plot:
                    fig_shewart.savefig(
                        rf'{data_path}/automated_analysis/locator/{molecule}_r{region}_shewart.png',
                        format='png',
                        dpi=300,
                    )
                if not show_plot:
                    plt.close(fig_shewart)

            print(
                f'Processing #{molecule} ({file_count + 1}/{n_files}) - starting edge optimisation.'
            )
            # Cdc48 cannot translocate and slip back simultaneously - therefore, split identified
            # translocations around midpoint of identified slipbacks
            slipback_mids = [
                int(i_lo + (i_hi - i_lo) / 2) for i_lo, i_hi in slipback_regions
            ]
            translocation_regions = detail.split_idx_regions_on_idx(
                translocation_regions, slipback_mids,
            )

            # optimise event start/end points, and filter out events which are too short or quick
            translocation_regions = detail.do_edge_optimisation(
                data_Lc_regional,
                translocation_regions,
                event_type='translocation',
                shortest_t=tl_shortest_t,
                shortest_Lc=tl_shortest_Lc,
                slowest_v=tl_slowest_v,
                allowed_t_range=tl_allowed_t_range,
                allowed_Lc_range=tl_allowed_Lc_range,
                molecule=molecule,
                region=region,
                debug=debug,
            )
            slipback_regions = detail.do_edge_optimisation(
                data_Lc_regional,
                slipback_regions,
                event_type='slipback',
                shortest_t=sb_shortest_t,
                shortest_Lc=sb_shortest_Lc,
                slowest_v=sb_slowest_v,
                allowed_t_range=sb_allowed_t_range,
                allowed_Lc_range=sb_allowed_Lc_range,
                molecule=molecule,
                region=region,
                debug=debug,
            )

            # merge events where two translocations have been optimised to overlapping positions,
            # and remove cases where translocation and slipback are marked simultaneously.
            translocation_regions = detail.merge_overlapping_regions(translocation_regions)
            slipback_regions = detail.merge_overlapping_regions(slipback_regions)
            slipback_regions = detail.remove_translocation_overlaps(
                translocation_regions, slipback_regions
            )

            print(
                f'Processing #{molecule} ({file_count + 1}/{n_files}) - saving translocations.\n'
            )

            # construct plots and event log
            detail.make_event_plots(
                data_Lc_regional,
                translocation_regions,
                slipback_regions,
                molecule,
                region,
                data_path,
                save_plot,
                show_plot,
            )

            event_log = detail.build_event_log(
                data_Lc_regional,
                molecule,
                region,
                translocation_regions,
                slipback_regions,
                event_log,
            )

    # save event log to file
    event_log = event_log.reset_index()
    event_log.to_csv(f'{data_path}/automated_analysis/event_log.csv')

    return


if __name__ == '__main__':

    # define data paths
    data_path_ctrl = r'../data/conformationally-free_ctrl'
    data_path_expt = r'../data/conformationally-free_expt'

    # compute target statistics by processing control data, using parameters defined in settings
    target_mean, target_st_dev = detail.get_target_stats(
        data_path_ctrl, **settings.processive_kwargs
    )

    processive_locator(data_path_expt, target_mean, target_st_dev, **settings.processive_kwargs)
