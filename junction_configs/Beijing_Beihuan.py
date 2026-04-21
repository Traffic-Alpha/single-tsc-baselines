'''
Author: WANG Maonan
Date: 2026-04-14 14:42:21
LastEditTime: 2026-04-14 20:28:12
LastEditors: WANG Maonan
Description: 北京北环路口配置
'''

JUNCTION = {
    "tls_id": "INT1",
    # ===== easy 路网 =====
    "easy_low_density": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    "easy_high_density": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    "easy_fluctuating_commuter": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    "easy_increasing_demand": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    "easy_random_perturbation": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    # ===== normal 路网 =====
    "normal_low_density": {
        "num_phases": 3, 
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    "normal_high_density": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    "normal_fluctuating_commuter": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    "normal_increasing_demand": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
    "normal_random_perturbation": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [3, 3, 3],
    },
}
