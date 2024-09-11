# legacy_code.py

"""
This file contains leftover or unresolved code from the original implementation.
It is not currently connected to the main workflow but may be useful later in the
refactoring process. The code snippets are preserved here for reference.
"""

# Old Learning Rates
# ACCEPTABLE_LR = [1, 0.01, 0.1, 0.5, 2, 3, 4, 5, 10, 20]  # Removed 0.01 and 20 to save time
# ACCEPTABLE_LR = [1, 0.1, 0.5, 2, 3, 4, 5, 10]  # Another version of the learning rates
ACCEPTABLE_LR = [0.5, 1]  # Current version being used

# Old Batch Sizes
# ACCEPTABLE_BATCH_SIZES = [256, 32, 60000, 4000]
ACCEPTABLE_BATCH_SIZES = [32, 256, 4000]  # Current version being used

# Pruning Percentages
PERCENTAGES = range(35, 100, 5)  # Ranges from 35% to 95% in steps of 5%

# Fitting Function and Stop Criteria (Old Code)
"""
For higher pruning percentages (35% and above), we are forcing all runs to B_max = 1000,
and removing the possibility of stopping earlier by removing the fitting function stop criteria.

try:
    params, inv_func = inv_fit(range(len(model.ces)), np.array(model.ces))
    a, b, c = params
    model.ce_asy_list.append(c)
    model.a_list.append(a)
    model.b_list.append(b)

    if 0.95 * model.ce_asy_list[-2] < model.ce_asy_list[-1] < 1.05 * model.ce_asy_list[-2]:
        model.stop_counter += 1
    if model.stop_counter > 4:
        print('stop training (fit)')
    if len(model.ces) > 29:
        model.stop_training = True
        break
    else:
        model.stop_counter = 0

except (RuntimeError, TypeError) as error:
    model.ce_asy_list.append(0)
    model.a_list.append(0)
    model.b_list.append(0)
"""

# Fitting Function to Stop Runs Using Inverse Fit
"""
def inv_fit(x, y):
    def inv_func(x, a, b, c):
        return (a / (x-b)) + c

    # Define lower and upper bounds for a, b, and c
    lower_bounds = [-np.inf, -np.inf, 0]  # -np.inf means no lower bound
    upper_bounds = [np.inf, np.inf, np.log(10)]  # np.inf means no upper bound

    popt, _ = curve_fit(inv_func, x, y, p0=(10, -10, 0.5))
    return popt, inv_func
"""
