'''
Author: WANG Maonan
Date: 2026-04-14 14:42:25
LastEditTime: 2026-04-14 14:45:31
LastEditors: WANG Maonan
Description: 
'''
'''
@Author: WANG Maonan
@Description: 天津直镜岛路口配置
'''

JUNCTION = {
    "tls_id": "INT1",
    # ===== easy 路网 =====
    "easy_low_density": {
        "num_phases": 4,  # TODO: 请确认
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    "easy_high_density": {
        "num_phases": 4,
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    "easy_fluctuating_commuter": {
        "num_phases": 4,
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    "easy_increasing_demand": {
        "num_phases": 4,
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    "easy_random_perturbation": {
        "num_phases": 4,
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    # ===== normal 路网 =====
    "normal_low_density": {
        "num_phases": 4,  # TODO: 请确认
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    "normal_high_density": {
        "num_phases": 4,
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    "normal_fluctuating_commuter": {
        "num_phases": 4,
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    "normal_increasing_demand": {
        "num_phases": 4,
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
    "normal_random_perturbation": {
        "num_phases": 4,
        "num_seconds": 600,
        "fix_phase_durations": [2, 2, 2, 2],
    },
}
