import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from ..utils import *
from pygam import GAM, s, te, f, l, utils

from . import *


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
                "numerical" if var2 in num_rel_dict.keys() else "categorical"
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


