import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from src.utility import *


def _fs(relativity, weight):
    average_rel = np.sum(relativity * weight) / np.sum(weight)
    square_diff = (relativity - average_rel) ** 2
    average_sq_diff = (np.sum(square_diff * weight) / np.sum(weight)) ** 0.5
    return average_sq_diff / average_rel


def factor_strength(
    num_rel_dict,
    cat_rel_dict,
    inter_rel_dict=None,
    weight="weight",
):
    """
    Calculate factor strength importance for numerical and categorical features,
    as well as the interactions terms.

    This function processes dictionaries of relativities for numerical and categorical
    features, and optionally their interactions, to produce factor strength rankings.
    It handles data cleaning, calculation of factor strengths, and organisation of
    results into DataFrames.

    Parameters:
    -----------
    num_rel_dict : dict
        Dictionary of numerical feature relativities.
        Keys are feature names (str) and values are pandas DataFrames containing
        the feature levels and their corresponding relativities.

    cat_rel_dict : dict
        Dictionary of categorical feature relativities.
        Keys are feature names (str) and values are pandas DataFrames containing
        the feature categories and their corresponding relativities.

    inter_rel_dict : dict, optional
        Dictionary of interaction term relativities.
        Keys are interaction terms (str) in the format "feature1 x feature2"
        and values are pandas DataFrames containing the interaction levels
        and their corresponding relativities.

    weight : str, default "weight"
        Column name in the relativity DataFrames that contains the weights
        to be used in factor strength calculations.

    Returns:
    --------
    tuple
        A tuple containing two pandas DataFrames:
        1. single_fact_df : DataFrame
           Contains factor strengths for individual features.
        2. inter_term_df : DataFrame or None
           Contains factor strengths for interaction terms.
           Returns None if inter_rel_dict is None.

    Notes:
    ------
    - The input dictionaries num_rel_dict, cat_rel_dict, and inter_rel_dict are
        the output from the extract_relativity function.
    - Feature names are cleaned by removing '_level' and '_cat_level' suffixes.
    - For interaction terms, the function checks that all variables used in
      interactions are present in the single feature dictionaries.
    - Factor strengths are sorted in descending order in both output DataFrames.

    Raises:
    -------
    AssertionError
        If any variable in the interaction terms is not present in the single
        factor dictionaries (num_rel_dict and cat_rel_dict).
    """

    # Initiate empty lists
    feature_list = []
    fs_list = []
    feature_type = []

    for var, rel in num_rel_dict.items():
        relativity_df = rel.set_index(var)
        fs_value = _fs(relativity_df["relativity"], relativity_df[weight])
        fs_list.append(fs_value)
        feature_list.append(var.replace("_level", ""))
        feature_type.append("numerical")

    for var, rel in cat_rel_dict.items():
        relativity_df = rel.set_index(var)
        fs_value = _fs(relativity_df["relativity"], relativity_df[weight])
        fs_list.append(fs_value)
        feature_list.append(var.replace("_cat_level", ""))
        feature_type.append("categorical")

    single_fact_df = pd.DataFrame(
        zip(feature_list, feature_type, fs_list),
        columns=["feature", "feature_type", "factor_strength"],
    )
    single_fact_df.sort_values(
        ["factor_strength"], inplace=True, ascending=False, ignore_index=True
    )

    if inter_rel_dict == None:
        inter_term_df = None
    else:
        all_keys = set(num_rel_dict.keys() | cat_rel_dict.keys())
        interaction_terms = list(inter_rel_dict.keys())
        var1_list = [re.split(" x ", v)[0] for v in interaction_terms]
        var2_list = [re.split(" x ", v)[1] for v in interaction_terms]
        all_var_list = list(set(var1_list + var2_list))
        assert all(
            v in all_keys for v in all_var_list
        ), "Some variables in interaction terms are not used as single factors."

        # Initialise empty lists
        inter_term_list = []
        feature1_list = []
        feature2_list = []
        feature1_type = []
        feature2_type = []
        fs_list = []

        for var, rel in inter_rel_dict.items():
            var1, var2 = re.split(" x ", var)
            var1_clean = var1.replace("_cat_level", "").replace("_level", "")
            var2_clean = var2.replace("_cat_level", "").replace("_level", "")
            inter_term_list.append(
                (
                    var1_clean,
                    var2_clean,
                )
            )
            feature1_type.append(
                "numerical" if var1 in num_rel_dict.keys() else "categorical"
            )
            feature2_type.append(
                "numerical" if var2 in num_rel_dict.values() else "categorical"
            )
            relativity_df = rel.set_index([var1, var2])
            fs_value = _fs(relativity_df["relativity"], relativity_df[weight])
            fs_list.append(fs_value)
            feature1_list.append(var1_clean)
            feature2_list.append(var2_clean)

        inter_term_df = pd.DataFrame(
            zip(
                inter_term_list,
                feature1_list,
                feature2_list,
                feature1_type,
                feature2_type,
                fs_list,
            ),
            columns=[
                "interaction_term",
                "feature1",
                "feature2",
                "feature1_type",
                "feature2_type",
                "factor_strength",
            ],
        )
        inter_term_df.sort_values(
            "factor_strength", inplace=True, ascending=False, ignore_index=True
        )
    return (single_fact_df, inter_term_df)


class RatingFactorTrend:
    """
    A class for analyzing and visualizing rating factor trends in the GAM model.

    This class provides functionality to plot trends of numerical and categorical rating factors,
    showing their relativity and weight distribution. It supports both linear and logarithmic scales
    for numerical factors and can handle interaction effects between factors.

    Attributes are initialized through the constructor. See `__init__` method for details.

    Methods:
        plot_trend: Plot the trend for a given variable (numerical or categorical).
    """

    def __init__(
        self,
        num_rel_dict,
        cat_rel_dict,
        numerical_banding,
        categorical_banding,
        inter_rel_dict=None,
        weight="weight",
    ):
        """
        Initialize the RatingFactorTrend object.

        Args:
            num_rel_dict (dict): Dictionary of DataFrames containing relativities for numerical factors.
            cat_rel_dict (dict): Dictionary of DataFrames containing relativities for categorical factors.
            numerical_banding (str): Path to Excel file containing banding information for numerical factors.
            categorical_banding (str): Path to Excel file containing banding information for categorical factors.
            inter_rel_dict (dict, optional): Dictionary of DataFrames containing relativities for interaction terms.
            weight (str, optional): Column name for the weight data in the relativity DataFrames. Defaults to "weight".
        """
        self.num_rel_dict = num_rel_dict
        self.cat_rel_dict = cat_rel_dict
        self.numerical_banding = numerical_banding
        self.categorical_banding = categorical_banding
        self.inter_rel_dict = inter_rel_dict
        self.weight = weight

    def _num_var_trend(
        self,
        var,
        ax,
        log_scale,
        tick_fontsize=14,
        label_fontsize=18,
        title_fontsize=22,
    ):
        """
        Internal method to plot trend for a numerical variable.

        Args:
            var (str): Name of the numerical variable to plot.
            ax (matplotlib.axes.Axes): Matplotlib axes object to plot on.
            log_scale (bool): Whether to use logarithmic scale for relativity.
            tick_fontsize (int): Font size for tick labels.
            label_fontsize (int): Font size for axis labels.
            title_fontsize (int): Font size for plot title.
        """
        relativity_df = self.num_rel_dict[var].set_index(var)
        leftylabel = "Relativity (Log-Scale)" if log_scale else "Relativity"
        sns.lineplot(
            data=relativity_df["relativity"], ax=ax, color="darkred", label=leftylabel
        )

        var_clean = var.replace("_level", "")
        banding = pd.read_excel(self.numerical_banding, sheet_name=var_clean)
        banding["index"] = range(1, len(banding) + 1)
        banding.set_index("index", inplace=True)

        df_groupby = relativity_df.join(banding["Level"])
        x_index, x_label = select_ticks(
            df_groupby.index.tolist(), df_groupby["Level"].tolist()
        )

        ax.xaxis.set_major_locator(mtick.FixedLocator(x_index))
        ax.set_xticklabels(x_label, fontsize=tick_fontsize)
        ax.set_xlabel("")
        ax.set_ylabel(leftylabel, fontsize=label_fontsize)
        ax.grid(False)

        if log_scale:
            ax.set_yscale("log")
            # Format y-axis labels
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda y, _: "{:.0f}".format(y))
            )

        df_groupby[self.weight] /= df_groupby[self.weight].sum()
        rightylabel = "Weight (%)"

        ax1 = ax.twinx()
        ax1.bar(
            df_groupby.index,
            df_groupby[self.weight],
            color="lightgray",
            alpha=0.5,
            ec="k",
            label=rightylabel,
        )

        vals = ax1.get_yticks()
        diff = np.min(np.diff(vals))
        dp = round_down_to_power_of_10(diff)
        dp = int(max(np.abs(np.log10(dp)) - 2, 0))

        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.{dp}%}"))
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax1.tick_params(axis="y", which="major", labelsize=tick_fontsize)
        ax1.set_ylabel(rightylabel, fontsize=label_fontsize)

        ax.set_title(var.replace("_level", ""), fontsize=title_fontsize)

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        bars, bar_labels = ax1.get_legend_handles_labels()
        ax.legend(lines + bars, labels + bar_labels, loc="best", fontsize=tick_fontsize, facecolor="white")

        ax.set_zorder(2)
        ax.patch.set_alpha(0)
        ax1.set_zorder(1)
        ax1.grid(False)
        ax1.set_facecolor("white")

    def _cat_var_trend(
        self,
        var,
        ax,
        tick_fontsize=14,
        label_fontsize=18,
        title_fontsize=22,
    ):
        """
        Internal method to plot trend for a categorical variable.

        Args:
            var (str): Name of the categorical variable to plot.
            ax (matplotlib.axes.Axes): Matplotlib axes object to plot on.
            tick_fontsize (int): Font size for tick labels.
            label_fontsize (int): Font size for axis labels.
            title_fontsize (int): Font size for plot title.
        """
        mapping = pd.read_excel(
            self.categorical_banding, sheet_name=var.replace("_cat_level", "")
        )
        relativity_df = self.cat_rel_dict[var].set_index(var)

        df_groupby = relativity_df.join(mapping.set_index("Integer_Value"), on=var)
        df_groupby["Categorical_Level"] = df_groupby["Categorical_Level"].astype(str)
        df_groupby.sort_values("Categorical_Level", inplace=True)
        df_groupby[self.weight] /= df_groupby[self.weight].sum()

        leftylabel = "Relativity"
        sns.lineplot(
            data=df_groupby.set_index("Categorical_Level")["relativity"],
            ax=ax,
            color="darkred",
            label=leftylabel,
        )

        ax.grid(False)
        ax.set_title(var.replace("_cat_level", ""), fontsize=title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.set_ylabel(leftylabel, fontsize=label_fontsize)
        ax.set_xlabel("")

        # # Set the x-axis tickers
        x_label = df_groupby["Categorical_Level"].to_list()
        ax.set_xticklabels(x_label, fontsize=tick_fontsize, rotation=0)

        rightylabel = "Weight (%)"
        ax1 = ax.twinx()
        ax1.bar(
            df_groupby["Categorical_Level"],
            df_groupby[self.weight],
            color="lightgray",
            alpha=0.5,
            ec="k",
            label=rightylabel,
        )

        vals = ax1.get_yticks()
        diff = np.min(np.diff(vals))
        dp = round_down_to_power_of_10(diff)
        dp = int(max(np.abs(np.log10(dp)) - 2, 0))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.{dp}%}"))
        ax1.tick_params(axis="y", which="major", labelsize=tick_fontsize)
        ax1.set_ylabel(rightylabel, fontsize=label_fontsize)

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        bars, bar_labels = ax1.get_legend_handles_labels()
        ax.legend(lines + bars, labels + bar_labels, loc="best", fontsize=tick_fontsize, facecolor="white")

        ax.set_zorder(2)
        ax.patch.set_alpha(0)
        ax1.set_zorder(1)
        ax1.grid(False)
        ax1.set_facecolor("white")

    def plot_trend(
        self,
        var,
        ax,
        log_scale=False,
        tick_fontsize=14,
        label_fontsize=18,
        title_fontsize=22,
    ):
        """
        Plot trend for a given variable (numerical or categorical).

        This method determines the type of variable and calls the appropriate internal method
        to create the plot.

        Args:
            var (str): Name of the variable to plot.
            ax (matplotlib.axes.Axes): Matplotlib axes object to plot on.
            log_scale (bool, optional): Whether to use logarithmic scale for relativity (only for numerical variables). Defaults to False.
            tick_fontsize (int, optional): Font size for tick labels. Defaults to 14.
            label_fontsize (int, optional): Font size for axis labels. Defaults to 18.
            title_fontsize (int, optional): Font size for plot title. Defaults to 22.

        Raises:
            AssertionError: If the variable is not found in num_rel_dict or cat_rel_dict.
        """
        assert (
            var in self.num_rel_dict.keys() or var in self.cat_rel_dict.keys()
        ), "Variable not used in the model."
        if var.find(" x ") != -1:
            pass
        elif var in self.num_rel_dict.keys():
            self._num_var_trend(
                var,
                log_scale=log_scale,
                ax=ax,
                tick_fontsize=tick_fontsize,
                label_fontsize=label_fontsize,
                title_fontsize=title_fontsize,
            )
        else:
            self._cat_var_trend(
                var,
                ax=ax,
                tick_fontsize=tick_fontsize,
                label_fontsize=label_fontsize,
                title_fontsize=title_fontsize,
            )


class FactorImpCV:
    """
    A class for calculating feature importance using cross-validation with Generalized Additive Models (GAM).

    This class implements a method to compute feature importance scores based on the change in Poisson deviance
    when each feature is "neutralized" (set to its mean or mode). It uses k-fold cross-validation to ensure
    robust estimates.

    Attributes are initialized in the __init__ method.

    Methods:
        fit(i): Fits a GAM model for a specific fold.
        baseline(i, gam_model): Calculates the baseline Poisson deviance for a specific fold.
        __call__(): Computes feature importance scores across all folds.

    Note:
        This class assumes that the necessary libraries (pandas, numpy, and a GAM implementation) 
        are imported and that a `total_poisson_dev` function is available for calculating 
        Poisson deviance.
    """

    def __init__(
        self,
        term_string,
        num_folds,
        X_train_dict,
        X_valid_dict,
        y_train_dict,
        y_valid_dict,
        w_train_dict,
        w_valid_dict,
        n_splines,
        lam_num,
        lam_cat, 
    ):
        """
        Initialize the FactorImpCV object.

        Args:
            term_string (str): A string representation of the GAM formula.
            num_folds (int): The number of cross-validation folds.
            X_train_dict (dict): Dictionary of training feature datasets for each fold.
            X_valid_dict (dict): Dictionary of validation feature datasets for each fold.
            y_train_dict (dict): Dictionary of training target variables for each fold.
            y_valid_dict (dict): Dictionary of validation target variables for each fold.
            w_train_dict (dict): Dictionary of training sample weights for each fold.
            w_valid_dict (dict): Dictionary of validation sample weights for each fold.
            n_splines (int): Number of splines to use in the GAM.
            lam_num (float): Smoothing parameter for numerical features.
            lam_cat (float): L2 regularisation parameter for categorical features.
        """
        self.term_string = term_string
        self.num_folds = num_folds
        self.X_train_dict = X_train_dict
        self.X_valid_dict = X_valid_dict
        self.y_train_dict = y_train_dict
        self.y_valid_dict = y_valid_dict
        self.w_train_dict = w_train_dict
        self.w_valid_dict = w_valid_dict
        self.n_splines = n_splines
        self.lam_num = lam_num 
        self.lam_cat = lam_cat

    def fit(self, i):
        """
        Fit a GAM model for a specific fold.

        Args:
            i (int): The index of the current fold.

        Returns:
            object: A fitted GAM model.
        """
        n_splines = self.n_splines
        lam_num = self.lam_num
        lam_cat = self.lam_cat
        gam_model = GAM(eval(self.term_string), distribution="poisson", link="log").fit(
            self.X_train_dict[i], self.y_train_dict[i], self.w_train_dict[i]
        )
        return gam_model

    def baseline(self, i, gam_model):
        """
        Calculate the baseline Poisson deviance for a specific fold.

        Args:
            i (int): The index of the current fold.
            gam_model (object): A fitted GAM model for the current fold.

        Returns:
            float: The calculated Poisson deviance.
        """
        ypred_df = pd.DataFrame(index=self.X_valid_dict[i].index)
        ypred_df["ypred_0"] = gam_model.predict(self.X_valid_dict[i])
        ypred_df["weight"] = self.w_valid_dict[i]
        ypred_df["ypred"] = ypred_df["ypred_0"] * ypred_df["weight"]
        ypred_df["target"] = self.y_valid_dict[i] * ypred_df["weight"]
        return total_poisson_dev(ypred_df["target"], ypred_df["ypred"])

    def __call__(self):
        """
        Compute feature importance scores across all folds.

        This method fits GAM models for each fold, calculates the baseline deviance,
        and then computes the change in deviance when each feature is "neutralized".
        The change in deviance is used as the measure of feature importance.

        Returns:
            pandas.DataFrame: A DataFrame containing feature names and their importance scores,
                              sorted in descending order of importance.
        """
        self.gam_list = [self.fit(i) for i in range(self.num_folds)]
        self.baseline = np.sum(
            [self.baseline(i, self.gam_list[i]) for i in range(self.num_folds)]
        )
        var_list = list(self.X_train_dict[0].columns)
        fact_imp_list = []

        for var in var_list:
            ypred_dummy = {}
            for i in range(self.num_folds):
                ypred_dummy[i] = pd.DataFrame(index=self.X_valid_dict[i].index)
                X = self.X_valid_dict[i].copy()
                if var.endswith("_cat_level"):
                    X[var] = X[var].mode()[0]
                else:
                    X[var] = X[var].mean()
                ypred_dummy[i]["ypred_0"] = self.gam_list[i].predict(X)
                ypred_dummy[i]["weight"] = self.w_valid_dict[i]
                ypred_dummy[i]["ypred"] = (
                    ypred_dummy[i]["ypred_0"] * ypred_dummy[i]["weight"]
                )
                ypred_dummy[i]["target"] = self.y_valid_dict[i] * ypred_dummy[i]["weight"]
            ypred_combined = pd.concat(ypred_dummy.values(), axis=0)
            dev = total_poisson_dev(
                ypred_combined["target"], ypred_combined["ypred"]
            )
            fact_imp_list.append(dev - self.baseline)
        output = pd.DataFrame(
            zip(var_list, fact_imp_list), columns=["feature", "fact_imp"]
        ).sort_values("fact_imp", ignore_index=True, ascending=False)

        return output
