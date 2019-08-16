from datetime import datetime, timedelta

def get_time_steps(start_time: datetime, end_time: datetime, stepsize = 1, return_tuple=False):
    '''
    This function returns a time range array between start_time and end_time.
    Args:
        start_time: datetime - the start of the range.
        end_time: datetime - the end of the range.
        stepsize: int/timedelta - number of hours representing the desired resolution.
        return_tuple: bool - if True then return tuples of (start,end) representing the windows. If False
                             (by default) then return only the array of the datetimes range.

    Returns:
        yields the next element (datetime or tuple of two datetimes).
    '''
    if isinstance(stepsize,int):
        stepsize = timedelta(hours=stepsize)

    cur_time_step = start_time
    while cur_time_step < end_time:
        next_time_step = cur_time_step + stepsize
        if next_time_step > end_time:
            next_time_step = end_time

        if return_tuple:
            yield (cur_time_step, next_time_step)
        else:
            yield cur_time_step

        cur_time_step = next_time_step

    if not return_tuple:
        yield end_time
