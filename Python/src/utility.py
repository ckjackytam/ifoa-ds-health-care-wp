import numpy as np
import pandas as pd
import random

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

