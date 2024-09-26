import os
import pandas as pd
import matplotlib

import helpers
import settings
import nonprocessive_locator_detail as detail


def nonprocessive_locator(
    data_path,
    target_st_dev,
    data_period,
    noise_window,
    savgol_window,
    peak_heights,
    region_t_min,
    event_t_min,
    show_plot,
    save_plot,
    debug,
    **kwargs,
):
    """
    Given a set of raw tweezers datafiles, computes all regions corresponding to nonprocessive
    action by Cdc48, and saves log of event data to file.

    Event detection is based on a measured increase in pseudo-variance of the data (\hat{\sigma^2}),
    as defined in the Methods section. In code documentation, this pseudo-variance (\hat{\sigma^2})
    is referred to as 'noise'.

    Arguments:
        data_path (str): relative path to the directory containing experimental data.
        target_st_dev (float): standard deviation of dys per segment measured in -Cdc48 conditions,
                               used to define Shewhart control limits.
        data_period (float): time gap between consective data points (1 / data frequency) [s].
        noise_window (float): width of moving window used to calculate noise [s].
        savgol_window (int): window length of Savitzky-Golay filter used to extract regions of
                             constant-distance data.
        peak_heights (tuple of float): (min, max) measured length which defines a valid region of
                                       constant-distance data [nm].
        region_t_min (float): minimum duration of a valid region of constant-distance data [s].
        event_t_min (float): minimum duration of a non-processive event [s].
        show_plot (bool): whether to display generated plots to the user.
        save_plot (bool): whether to save generated plots to file.
        debug (bool): use function in debug mode, i.e. with more verbose console output.
    """

    manual_region_lims = helpers.load_manual_region_lims(data_path)

    event_log = pd.DataFrame(
        columns=[
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
        ]
    )

    if not show_plot:
        matplotlib.use('Agg')

    filenames = [f for f in os.listdir(f'{data_path}/wlc_manual_fit') if '.csv' in f]

    # check for molecules with duplicate IDs
    helpers.verify_duplicate_molecules(filenames)

    noise_window_n = int(noise_window / data_period)
    event_n_min = int(event_t_min / data_period)

    for filename in filenames:
        data_Lc, molecule = detail.load_data(
            data_path,
            filename,
            data_period=data_period,
            savgol_window=savgol_window,
            peak_heights=peak_heights,
            region_t_min=region_t_min,
            noise_window=noise_window,
        )

        print(f'Processing #{molecule}.')

        for region in data_Lc['region'].unique():
            data_Lc_regional = data_Lc.loc[data_Lc['region'] == region]

            # some data files have erroneous errors - manually remove these if present
            data_Lc_regional = helpers.apply_manual_region_limits(
                data_Lc_regional, manual_region_lims[molecule][str(region)]
            )

            # load AutoStepfinder fit
            step_regions = detail.load_stepfit(data_path, data_Lc, molecule, region)

            # compute noise enrichment
            data_Lc_regional = detail.compute_noise_enrichment(
                data_Lc_regional, step_regions, noise_window_n
            )

            # identify nonprocessive regions
            data_Lc_regional['marked'] = data_Lc_regional['noise'] > (
                data_Lc_regional['step_median'] + 3 * target_st_dev
            )

            # find continuous regions, merge events which occur within noise window, and remove
            #     events which are too short
            nonprocessive_regions = helpers.group_states(data_Lc_regional['marked'])
            nonprocessive_regions = [
                (i_start, i_end)
                for i_start, i_end in nonprocessive_regions
                if i_end - i_start > event_n_min
            ]

            # construct plots and event log
            detail.make_event_plots(
                data_Lc_regional,
                nonprocessive_regions,
                noise_window,
                target_st_dev,
                molecule,
                region,
                data_path,
                save_plot,
                show_plot,
            )

            event_log = detail.build_event_log(
                data_Lc_regional,
                nonprocessive_regions,
                molecule,
                region,
                event_log,
            )

            # save data_Lc_regional - data required for some main-text figures
            data_Lc_regional.to_csv(
                f'{data_path}/automated_analysis/nonprocessive_noise/data_Lc_regional_'
                f'{molecule}_r{region}'
            )

    # save event log to file
    event_log = event_log.reset_index()
    event_log.to_csv(f'{data_path}/automated_analysis/event_log.csv')

    return


if __name__ == '__main__':
    # TODO update paths
    data_path_ctrl = r'../data/ubi-chain_ctrl'

    # choose between normal (ATP) and ATPgS conditions
    data_path_expt = r'../data/ubi-chain_expt'
    # data_path_expt = r'../data/ubi-chain_atpgs'

    # compute target statistics by processing control data, using parameters defined in settings
    target_st_dev = detail.get_target_stats(data_path_ctrl, **settings.nonprocessive_kwargs)

    nonprocessive_locator(data_path_expt, target_st_dev, **settings.nonprocessive_kwargs)
