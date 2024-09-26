# --------------------------------- processive locator settings --------------------------------- #
"""
Parameters are fully-defined in processive_locator.processive_locator, and in the Methods section.
"""
processive_kwargs = {

    # finding constant-distance regions
    'data_period': 0.002,
    'savgol_window_tl': 51,
    'savgol_window_sb': 11,
    'peak_heights': [0, 350],
    'region_t_min': 30,

    # finding translocation / slipback events
    'segment_time_shewart': 0.2,
    'n_resamples': 19,

    # filtering translocation / slipback events
    'tl_shortest_t': 0.2,
    'tl_shortest_Lc': 18,
    'tl_slowest_v': 5,
    'tl_allowed_t_range': 1,
    'tl_allowed_Lc_range': 5,
    'sb_shortest_t': 0.02,
    'sb_shortest_Lc': 18,
    'sb_slowest_v': 0,
    'sb_allowed_t_range': 0,
    'sb_allowed_Lc_range': 20,

    # program setup
    'show_plot': True,
    'save_plot': False,
    'debug': True,
}


# ------------------------------- non-processive locator settings ------------------------------- #
"""
Parameters are fully-defined in nonprocessive_locator.nonprocessive_locator, and in the Methods
section.
"""
nonprocessive_kwargs = {
    # finding constant-distance regions
    'data_period': 0.002,  # s / data point
    'savgol_window': 51,
    'peak_heights': [0, 350],  # nm
    'region_t_min': 30,  # s
    'noise_window': 1,  # s

    # filtering non-processive events
    'event_t_min': 1,  # s

    # program setup
    'show_plot': True,
    'save_plot': False,
    'debug': True,
}
