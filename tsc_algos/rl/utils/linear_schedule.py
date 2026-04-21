'''
@Author: WANG Maonan
@Description: Linear learning rate schedule
'''
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func
