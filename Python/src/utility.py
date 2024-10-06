import numpy as np
import pandas as pd
import random
import math


def total_poisson_dev(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    nlogn = np.where(y == 0, 0, y * np.log(y / y_pred))
    dev = 2 * (nlogn - (y - y_pred))
    return dev.sum()


def compile_data(
    data_dict,
    data_path,
    cal_year="CalendarYear",
    data_tab="Data",
    unique_id="unique_id",
    fold="random_fold",
    holdout="holdout",
    seed_holdout=777,
    seed_cv=55,
    training_prop=0.8,
    num_fold=5,
):
    """
    Compile and preprocess data from multiple Excel files into a single DataFrame.

    This function performs the following operations:
    1. Imports data from Excel files specified in data_dict.
    2. Concatenates all data into a single DataFrame.
    3. Assigns a unique identifier to each datapoint.
    4. Splits the data into training and holdout sets.
    5. Assigns cross-validation folds to the training data and the holdout fold to the test data.

    Parameters:
    -----------
    data_dict : dict
        A dictionary mapping calendar years to Excel file names.
    data_path : str
        The path to the directory containing the Excel files.
    cal_year : str, optional
        The name of the column to store the calendar year (default: "CalendarYear").
    data_tab : str, optional
        The name of the sheet in the Excel files to read data from (default: "Data").
    unique_id : str, optional
        The name of the column to store the unique identifier (default: "unique_id").
    fold : str, optional
        The name of the column to store the cross-validation fold assignments (default: "random_fold").
    holdout : str, optional
        The name of the column to store the holdout flag (default: "holdout").
    seed_holdout : int, optional
        The random seed for holdout set selection (default: 777).
    seed_cv : int, optional
        The random seed for cross-validation fold assignment (default: 55).
    training_prop : float, optional
        The proportion of data to use for training (default: 0.8).
    num_fold : int, optional
        The number of cross-validation folds to create (default: 5).

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the compiled data with additional columns for
        unique_id, holdout flag, and cross-validation fold assignments.

    Raises:
    -------
    AssertionError
        If the sum of training and holdout data lengths doesn't match the total number of rows.

    Notes:
    ------
    - The function assumes that all Excel files have the same structure.
    - The holdout set is randomly selected, and the remaining data is used for training.
    - Cross-validation folds are only assigned to the training data.
    - The holdout set is assigned to an additional fold (num_fold + 1).
    """

    # Import the data by calendar years
    data_map = {}
    for year, file in data_dict.items():
        data_map[year] = pd.read_excel(f"{data_path}/{file}", data_tab)
        data_map[year][cal_year] = year

    # Concatenate the data together into a single dataframe
    df_data = pd.concat(data_map.values())
    # Set a unique identifier for each datapoint
    df_data[unique_id] = range(len(df_data))

    # Generate a list of random numbers without replacement
    random.seed(seed_holdout)

    random_numbers = random.sample(list(range(len(df_data))), len(df_data))

    # Check the random_numbers generated
    random_numbers_1 = np.sort(np.array(random_numbers))
    print((random_numbers_1 == np.array(range(len(df_data)))).sum())

    # Define row indices for training and those for holdout
    train_size = int(training_prop * len(df_data))
    index_train = random_numbers[:train_size]
    index_test = random_numbers[train_size:]

    # Assertion to check the training and holdout indices
    assert len(index_train) + len(index_test) == len(
        df_data
    ), f"Sum of lengths of training and holdout data is not equal to the total number of rows in the original data."

    # Define a holdout flag
    df_data[holdout] = np.where(df_data[unique_id].isin(index_train), 0, 1)

    # Define a column for random folds for cross validations
    np.random.seed(seed_cv)
    df_data[fold] = np.random.randint(1, num_fold + 1, len(df_data))
    df_data[fold] = np.where(df_data[holdout] == 0, df_data[fold], num_fold + 1)

    return df_data


def round_down_to_power_of_10(x):
    power = math.floor(math.log10(abs(x)))
    return 10**power


def _round_to_multiple(x, a, round_func):
    return round_func(x / a) * a


def double_lift_plot(
    data, predict, target, exposure, n_bin=None, n_dec=None, endpoint_quantile=0.05
):
    """
    Generate a double lift plot table for comparing two predictive models.

    This function creates a table that can be used to visualize the performance
    of two predictive models relative to each other and to the actual target values.
    It bins the data based on the ratio of the two model predictions and calculates
    average values within each bin.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input data containing predictor columns, target column, and exposure column.
    predict : list of str
        Names of two columns in 'data' containing the predictions from two models.
    target : str
        Name of the column in 'data' containing the actual target values.
    exposure : str
        Name of the column in 'data' containing the exposure values.
    n_bin : int, optional
        Number of bins to use. If None, bins are determined automatically based on data range.
    n_dec : int, optional
        Number of decimal places to round the bin edges. If None, no rounding is performed.
    endpoint_quantile : float or tuple of float, optional
        Quantile(s) to use for determining the endpoint of the ratio range.
        If a single float, it's used as the lower quantile and (1 - value) as the upper quantile.
        If a tuple, the two values are used as lower and upper quantiles respectively.
        Default is 0.05, which uses the 5th and 95th percentiles.

    Returns:
    --------
    pandas.DataFrame
        A table with the following columns:
        - 'band': The bin number
        - 'ratio_band': The interval of ratios for each bin
        - Columns for the target and each predictor, containing average values per bin
        - 'exposure': Total exposure in each bin

    Notes:
    ------
    - The function applies a base multiplier to align the predictions with the overall target rate.
    - Bin widths are determined adaptively based on the range of the ratio between the two predictors.
    - The resulting table can be used to create a double lift plot, showing how the two models
      perform relative to each other and to the actual target values across different ratio bands.
    """
    T = data[[*predict, target, exposure]].copy()

    # Apply base multiplier to the predictions
    o_cf = T[target].sum() / T[exposure].sum()
    for p in predict:
        T[p] *= o_cf / (T[p].sum() / T[exposure].sum())

    T["ratio"] = T[predict[0]] / T[predict[1]]

    if isinstance(endpoint_quantile, (list, tuple)):
        lq, rq = endpoint_quantile[:2]
    else:
        lq, rq = endpoint_quantile, 1 - endpoint_quantile

    pcl, pcr = T.ratio.quantile([lq, rq])

    # Simplified bin_width calculation
    pc_range = pcr - pcl
    bin_width = next(
        w
        for r, w in [
            (0.08, pc_range / 8),
            (0.16, 0.01),
            (0.4, 0.02),
            (1, 0.05),
            (2, 0.1),
            (5, 0.2),
            (float("inf"), 0.5),
        ]
        if pc_range < r
    )

    pcl_ = _round_to_multiple(pcl, bin_width, math.ceil)
    pcr_ = _round_to_multiple(pcr, bin_width, math.floor)

    if n_bin is None:
        k = np.arange(pcl_, pcr_ + bin_width, bin_width)
    else:
        k = np.linspace(pcl_, pcr_, n_bin)

    k[0], k[-1] = min(T.ratio.min(), k[0]), max(T.ratio.max(), k[-1])

    if n_dec is not None:
        k = np.round(k, n_dec)

    T["ratio_band"] = pd.cut(T.ratio, bins=k, include_lowest=True)

    output_table = (
        T.groupby("ratio_band")
        .agg(
            {
                **{col: "sum" for col in [target, *predict, exposure]},
            }
        )
        .reset_index()
    )

    for col in [target, *predict]:
        output_table[col] /= output_table[exposure]

    output_table["band"] = range(1, len(output_table) + 1)
    output_table.set_index("band", inplace=True)

    return output_table

