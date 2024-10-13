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


def round_to_multiple(x, a, round_func):
    return round_func(x / a) * a


def select_ticks(x_index, x_label):
    n = len(x_index)
    if n <= 10:
        step = 1
    elif n <= 30:
        step = 3
    elif n <= 50:
        step = 5
    elif n <= 100:
        step = 7
    else:
        step = 20
    return x_index[::step], x_label[::step]


class DoubleLift:
    """
    A class for creating and visualizing double lift tables to compare two prediction models against a target variable.

    This class calculates a double lift table by binning the ratio of two prediction models and comparing their
    performance against a target variable. It also provides a method to plot the results.

    Attributes:
        double_lift_table (pandas.DataFrame): Resulting double lift table after calculations.

    Methods:
        dl_table(): Calculates the double lift table.
        plot_double_lift(ax, title, leftylabel, rightylabel, target_legend, new_model_legend, baseline_legend,
                         palette, tick_fontsize, label_fontsize, title_fontsize):
            Plots the double lift chart on the given matplotlib axis.

    """

    def __init__(
        self,
        ypred_df,
        predict,
        target,
        weight,
        boundary,
        approx_num_band=10,
    ):
        """
        Initialize the DoubleLift object.

        Args:
            ypred_df (pandas.DataFrame): DataFrame containing predictions and target values.
            predict (list): List of column names for the two prediction models to compare. First component is the
                new model; second component is the baseline.
            target (str): Column name of the target variable.
            weight (str): Column name of the weight variable.
            boundary (tuple): Tuple of (left_bound, right_bound) for quantile-based ratio binning.
            approx_num_band (int, optional): Approximate number of bands for ratio binning. Defaults to 10.
        """
        self.ypred_df = ypred_df
        self.predict = predict
        self.target = target
        self.weight = weight
        self.boundary = boundary
        self.approx_num_band = approx_num_band
        self.double_lift_table = self.dl_table()

    def dl_table(self):
        """
        Calculate the double lift table.

        This method performs the following steps:
        1. Apply a base adjustment to predictions to align their average levels with the actual average.
        2. Calculates the ratio between the two prediction models.
        3. Determines bin width based on the specified approximate number of bands.
        4. Creates ratio bands and aggregates data within each band.
        5. Formats the resulting table for readability.

        Returns:
            pandas.DataFrame: The calculated double lift table.
        """
        T = self.ypred_df[[*self.predict, self.target, self.weight]].copy()
        self.new_model_predict = self.predict[0]
        self.baseline = self.predict[1]

        # Apply base multiplier to the predictions
        o_cf = T[self.target].sum() / T[self.weight].sum()

        for p in self.predict:
            T[p] *= o_cf / (T[p].sum() / T[self.weight].sum())

        T["ratio"] = T[self.predict[0]] / T[self.predict[1]]

        left_bound, right_bound = self.boundary

        left_ep, right_ep = T.ratio.quantile([left_bound, right_bound])

        # Determine the bin width
        main_range = right_ep - left_ep
        x = np.append(0.01, np.arange(0.05, 0.55, 0.05))
        len_list = [len(np.arange(left_ep, right_ep + i, i)) for i in x]
        closest_index = min(
            range(len(len_list)), key=lambda i: abs(len_list[i] - self.approx_num_band)
        )
        bin_width = x[closest_index]

        left_ep_final = round_to_multiple(left_ep, bin_width, math.ceil)
        right_ep_final = round_to_multiple(right_ep, bin_width, math.floor)
        k = np.arange(left_ep_final, right_ep_final + bin_width, bin_width)
        k[0], k[-1] = min(T.ratio.min(), k[0]), max(T.ratio.max(), k[-1])
        T["ratio_band"] = pd.cut(T.ratio, bins=k, include_lowest=True)

        output_table = (
            T.groupby("ratio_band")
            .agg(
                {
                    **{col: "sum" for col in [self.target, *self.predict, self.weight]},
                }
            )
            .reset_index()
        )

        for col in [self.target, *self.predict]:
            output_table[col] /= output_table[self.weight]

        output_table["ratio_band"] = output_table["ratio_band"].astype(str)
        lowest_band = output_table["ratio_band"].iloc[0]
        comma_loc = lowest_band.find(",")
        output_table["ratio_band"].iloc[0] = (
            "<= " + output_table["ratio_band"].iloc[0][comma_loc + 2 : -1]
        )

        highest_band = output_table["ratio_band"].iloc[-1]
        comma_loc = highest_band.find(",")
        output_table["ratio_band"].iloc[-1] = (
            "> " + output_table["ratio_band"].iloc[-1][1:comma_loc]
        )
        output_table["band"] = range(1, len(output_table) + 1)
        output_table.set_index("band", inplace=True)
        return output_table

    def plot_double_lift(
        self,
        ax,
        title,
        leftylabel,
        rightylabel,
        target_legend,
        new_model_legend,
        baseline_legend,
        palette=["#4e79a7", "#f28e2b", "#e15759"],
        tick_fontsize=14,
        label_fontsize=18,
        title_fontsize=22,
    ):
        """
        Plot the double lift chart on the given matplotlib axis.

        This method creates a line plot for the target and two prediction models, and overlays a bar plot
        for the weight distribution.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
            title (str): Title of the chart.
            leftylabel (str): Label for the left y-axis (lift values).
            rightylabel (str): Label for the right y-axis (population distribution).
            target_legend (str): Legend label for the target variable.
            new_model_legend (str): Legend label for the new model.
            baseline_legend (str): Legend label for the baseline model.
            palette (list, optional): List of colors for the line plot. Defaults to ["#4e79a7", "#f28e2b", "#e15759"].
            tick_fontsize (int, optional): Font size for tick labels. Defaults to 14.
            label_fontsize (int, optional): Font size for axis labels. Defaults to 18.
            title_fontsize (int, optional): Font size for the chart title. Defaults to 22.
        """
        columns = {
            self.target: target_legend,
            self.new_model_predict: new_model_legend,
            self.baseline: baseline_legend,
        }

        self.double_lift_table.rename(
            columns=columns,
            inplace=True,
        )

        sns.lineplot(
            data=self.double_lift_table[columns.values()],
            ax=ax,
            markers=True,
            palette=palette,
        )
        ax.xaxis.set_ticks(np.arange(1, len(self.double_lift_table) + 1, 1))
        xticklabels = self.double_lift_table.ratio_band.to_list()

        # Determine the decimal places of the primary y-axis
        vals = ax.get_yticks()
        diff = np.min(np.diff(vals))
        dp = round_down_to_power_of_10(diff)
        dp = int(max(np.abs(np.log10(dp)) - 2, 0))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.{dp}%}"))

        ax.set_ylabel(leftylabel, fontsize=label_fontsize)
        ax.set_xlabel(
            f"Ratio of {new_model_legend} to {baseline_legend}",
            fontsize=label_fontsize,
        )

        ax.grid(False)
        ax.set_facecolor("white")
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=tick_fontsize)

        self.double_lift_table[self.weight + "_norm"] = (
            self.double_lift_table[self.weight]
            / self.double_lift_table[self.weight].sum()
        )

        ax1 = ax.twinx()
        ax1.bar(
            self.double_lift_table.index,
            self.double_lift_table[self.weight + "_norm"],
            color="lightgray",
            alpha=0.5,
            ec="k",
            label=rightylabel,
        )

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        bars, bar_labels = ax1.get_legend_handles_labels()
        ax.legend(
            lines + bars,
            labels + bar_labels,
            loc="best",
            fontsize=tick_fontsize,
            facecolor="white",
        )

        # format decimal
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0%}"))
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax1.tick_params(axis="y", which="major", labelsize=tick_fontsize)

        ax.set_title(title, fontsize=title_fontsize)
        ax.set_zorder(2)
        ax.patch.set_alpha(0)
        ax1.set_zorder(1)


def actual_vs_predicted(
    ypred_df,
    target_tuple,
    predict_dict,
    weight,
    var,
    is_var_num,
    banding_loc=None,
    plot=False,
    ax=None,
    log_scale=False,
    align_predict=True,
    leftylabel="Frequency",
    rightylabel="Weight (%)",
    palette=["#4e79a7", "#f28e2b", "#e15759"],
    tick_fontsize=14,
    label_fontsize=18,
    title_fontsize=22,
):
    """
    Compare actual vs predicted frequencies and optionally plot the results.

    This function processes a DataFrame containing actual and predicted values,
    calculates frequencies, and optionally creates a plot comparing actual vs
    predicted frequencies along with a weight distribution.

    Parameters:
    -----------
    ypred_df : pandas.DataFrame
        DataFrame containing the prediction data.
    target_tuple : tuple
        A tuple containing the target column name and its display name.
    predict_dict : dict
        A dictionary with keys as prediction column names and values as their display names.
    weight : str
        The name of the column containing weights.
    var : str
        The variable to group by.
    is_var_num : bool
        True if the grouping variable is numeric, False otherwise.
    banding_loc : str, optional
        Path to an Excel file containing banding information.
    plot : bool, default False
        If True, create and display a plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    log_scale : bool, default False
        If True, use a logarithmic scale for the frequency axis.
    align_predict : bool, default True
        If True, align predictions with actual values.
    leftylabel : str, default "Frequency"
        Label for the left y-axis.
    rightylabel : str, default "Weight (%)"
        Label for the right y-axis.
    palette : list, default ["#4e79a7", "#f28e2b", "#e15759"]
        Color palette for the plot.
    tick_fontsize : int, default 14
        Font size for tick labels.
    label_fontsize : int, default 18
        Font size for axis labels.
    title_fontsize : int, default 22
        Font size for the plot title.

    Returns:
    --------
    pandas.DataFrame or None
        If plot is False, returns a DataFrame with grouped and processed data.
        If plot is True, returns None and displays the plot.

    Notes:
    ------
    - The function groups the data by the specified variable and calculates
      frequencies for actual and predicted values.
    - If banding information is provided, it will be used to label the x-axis.
    - The plot (if requested) shows line plots for actual and predicted frequencies,
      and a bar plot for weight distribution.
    """

    df_groupby = ypred_df.groupby(var)[
        [target_tuple[0], *list(predict_dict.keys()), weight]
    ].sum()

    df_groupby["baseline_freq"] = (
        df_groupby[list(predict_dict.keys())[1]] / df_groupby[weight]
    )
    df_groupby["new_model_freq"] = (
        df_groupby[list(predict_dict.keys())[0]] / df_groupby[weight]
    )
    df_groupby["actual_freq"] = df_groupby[target_tuple[0]] / df_groupby[weight]

    # Apply base multiplies to the predictions
    if align_predict:
        df_groupby["new_model_freq"] *= (
            df_groupby[target_tuple[0]].sum()
            / df_groupby[list(predict_dict.keys())[0]].sum()
        )
        df_groupby["baseline_freq"] *= (
            df_groupby[target_tuple[0]].sum()
            / df_groupby[list(predict_dict.keys())[1]].sum()
        )

    columns = {
        "actual_freq": target_tuple[1],
        "new_model_freq": predict_dict[list(predict_dict.keys())[0]],
        "baseline_freq": predict_dict[list(predict_dict.keys())[1]],
    }

    df_groupby.rename(
        columns=columns,
        inplace=True,
    )

    if banding_loc is not None:
        if is_var_num:
            banding = pd.read_excel(
                banding_loc,
                sheet_name=var.replace("_level", ""),
            )
            banding["index"] = range(1, len(banding) + 1)
            banding.set_index("index", inplace=True)
            df_groupby = df_groupby.join(banding["Level"])
            x_index, x_label = select_ticks(
                df_groupby.index.tolist(), df_groupby["Level"].tolist()
            )
        else:
            mapping = pd.read_excel(banding_loc, sheet_name=var.replace("_cat_level", ""))
            df_groupby = df_groupby.join(mapping.set_index("Integer_Value"))
            df_groupby["Categorical_Level"] = df_groupby["Categorical_Level"].astype(
                str
            )
            df_groupby.sort_values("Categorical_Level", inplace=True)
            x_label = df_groupby["Categorical_Level"].to_list()

    if not is_var_num:
        df_groupby.index = df_groupby.index.astype(str)

    if plot:
        sns.lineplot(
            data=df_groupby[columns.values()],
            ax=ax,
            markers=True,
            palette=palette,
        )

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda y, _: "{:.2%}".format(y))
            )
        ax.tick_params(axis="y", which="major", labelsize=tick_fontsize)

        if banding_loc is not None:
            if is_var_num:
                ax.xaxis.set_major_locator(mtick.FixedLocator(x_index))
            ax.set_xticklabels(x_label, fontsize=tick_fontsize)

        ax.set_xlabel("")
        ax.set_ylabel(
            leftylabel + " (Log-Scaled)" if log_scale else leftylabel,
            fontsize=label_fontsize,
        )

        # Determine the decimal places for the primary y-axis
        vals = ax.get_yticks()
        diff = np.min(np.diff(vals))
        dp = round_down_to_power_of_10(diff)
        dp = int(max(np.abs(np.log10(dp)) - 2, 0))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.{dp}%}"))

        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.grid(False)

        df_groupby[weight] /= df_groupby[weight].sum()

        ax1 = ax.twinx()
        ax1.bar(
            df_groupby.index,
            df_groupby[weight],
            color="lightgray",
            alpha=0.5,
            ec="k",
            label=rightylabel,
        )

        ax1.tick_params(axis="y", which="major", labelsize=tick_fontsize)
        ax1.set_ylabel(rightylabel, fontsize=label_fontsize)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.{0}%}"))

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        bars, bar_labels = ax1.get_legend_handles_labels()
        ax.legend(
            lines + bars,
            labels + bar_labels,
            loc="best",
            fontsize=tick_fontsize,
            facecolor="white",
        )

        ax.set_title(
            var.replace("_level", "").replace("_cat_level", ""),
            fontsize=title_fontsize,
        )
        ax.set_zorder(2)
        ax.patch.set_alpha(0)
        ax1.set_zorder(1)
    else:
        return df_groupby

