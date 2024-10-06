from src.utility import *
import numpy as np
from pygam import GAM, s, te, f, l, utils
import pandas as pd
from itertools import chain, product


def create_term_structure(
    num_vars,
    cat_vars,
    feature_df,
    interaction_term_list=None,
    base_pred=None,
    mono_dec_var_list=[],
    mono_inc_var_list=[],
    lam_num="lam_num",
    lam_cat="lam_cat",
    n_splines="n_splines",
    lam_num_inter="lam_num_inter",
    lam_cat_inter="lam_cat_inter",
    n_splines_inter="n_splines_inter",
):
    """
    Generate a term structure string for PyGAM Generalized Additive Model training and corresponding variable mappings.

    This function creates a term string that specifies the structure of a GAM, including smooth terms for
    numerical variables, factor terms for categorical variables, and tensor product interactions. It also
    handles monotonicity constraints and different smoothing parameters - the number of splines
    and regularisation parameters - for each term type.

    Parameters:
    -----------
    num_vars: list of str
        List of numerical variable names.
    cat_vars: list of str
        List of categorical variable names.
    feature_df : pd.DataFrame
        DataFrame containing the training feature data.
    interaction_term_list : list of tuples, optional
        List of variable pairs for interaction terms. Each tuple contains two variable names. Default to None.
    base_pred : str, optional
        Column name of the starting points for GAM. Default is to build the model from scratch. Default to None.
    mono_dec_var_list: list of str, optional
        List of variables with monotonic decreasing constraints. Default to an emtpy list.
    mono_inc_var_list: list of str, optional
        List of variables with monotonic increasing constraints. Default to an empty list.
    lam_num : str, optional
        Name of the smoothing parameter for numerical variables (default: "lam_num").
    lam_cat : str, optional
        Name of the L2 regularisation parameter for categorical variables (default: "lam_cat").
    n_splines : str, optional
        Name of the variable specifying the number of splines (default: "n_splines").
    lam_num_inter : str, optional
        Name of the smoothing parameter for numerical interactions. Default is "lam_num_inter".
    lam_cat_inter : str, optional
        Name of the L2 regularisation parameter for categorical interactions. Default is "lam_cat_inter".
    n_splines_inter : str, optional
        Name of the variable specifying the number of splines for interaction terms. Default is "n_splines_inter".

    Returns:
    --------
    tuple
        A tuple containing:
        1. term_string (str): The generated term string for GAM training.
        2. num_feat_map (dict): Dictionary mapping integer indices to numerical variable names.
        3. cat_feat_map (dict): Dictionary mapping integer indices to categorical variable names.
        4. interact_term_map (dict): Dictionary mapping integer indices to interaction variable pairs.

    Notes:
    ------
    The starting point of GAM (base_pred) is incorporated by a linear term in GAM, which is always the first term in the model.
    """

    # Initiate empty outputs
    term_string = ""
    num_feat_map = {}
    cat_feat_map = {}
    interact_term_map = {}

    # Incorporate the base predictions as a linear term in GAM
    if base_pred is not None:
        v_index = feature_df.columns.tolist().index(base_pred)
        added_string = f"l({v_index}, lam=0)"
        term_string = term_string + added_string
        num_feat_map[0] = base_pred
        is_base_present = 1
    else:
        is_base_present = 0

    for q, v in enumerate(num_vars):
        v_index = feature_df.columns.tolist().index(v)
        if is_base_present == 0 and q == 0:
            added_string = f"s({v_index}, lam={lam_num}, n_splines={n_splines}"
        else:
            added_string = f"+ s({v_index}, lam={lam_num}, n_splines={n_splines}"

        if v in mono_dec_var_list:
            added_string = added_string + ", constraints='monotonic_dec')"
        elif v in mono_inc_var_list:
            added_string = added_string + ", constraints='monotonic_inc')"
        else:
            added_string = added_string + ")"
        term_string = term_string + added_string
        num_feat_map[is_base_present + q] = v

    for q, v in enumerate(cat_vars):
        v_index = feature_df.columns.tolist().index(v)
        added_string = f" + f({v_index}, lam={lam_cat})"
        term_string = term_string + added_string
        cat_feat_map[is_base_present + len(num_vars) + q] = v

    if interaction_term_list == None:
        interact_term_map = None
    else:
        for q, (v1, v2) in enumerate(interaction_term_list):
            v1_index = feature_df.columns.tolist().index(v1)
            v2_index = feature_df.columns.tolist().index(v2)
            v1_type = "numerical" if v1 in num_vars else "categorical"
            v2_type = "numerical" if v2 in num_vars else "categorical"

            if v1 in mono_dec_var_list:
                v1_constraint = "'monotonic_dec'"
            elif v1 in mono_inc_var_list:
                v1_constraint = "'monotonic_inc'"
            else:
                v1_constraint = None

            if v2 in mono_dec_var_list:
                v2_constraint = "'monotonic_dec'"
            elif v2 in mono_inc_var_list:
                v2_constraint = "'monotonic_inc'"
            else:
                v2_constraint = None

            if v1 in num_vars:
                v1_nsplines = n_splines_inter
                v1_lam_inter = lam_num_inter
            else:
                v1_nsplines = max(len(feature_df[v1].unique()), 4)
                v1_lam_inter = lam_cat_inter

            if v2 in num_vars:
                v2_nsplines = n_splines_inter
                v2_lam_inter = lam_num_inter
            else:
                v2_nsplines = max(len(feature_df[v2].unique()), 4)
                v2_lam_inter = lam_cat_inter

            nsplines_list = f"[{v1_nsplines}, {v2_nsplines}]"
            type_list = [v1_type, v2_type]
            constr_list = f"[{v1_constraint}, {v2_constraint}]"
            lam_list = f"[{v1_lam_inter}, {v2_lam_inter}]"
            added_string = f" + te({v1_index}, {v2_index}, lam={lam_list}, n_splines={nsplines_list}, constraints={constr_list}, dtype={type_list})"

            term_string = term_string + added_string
            interact_term_map[is_base_present + len(num_vars) + len(cat_vars) + q] = (
                v1,
                v2,
            )

    return (term_string, num_feat_map, cat_feat_map, interact_term_map)


class OptunaGamObjectiveCV:
    """
    A custom objective class for Optuna optimization of Generalized Additive Models (GAMs).

    This class encapsulates the objective function for Optuna to optimize GAM hyperparameters using cross validation.
    It handles the training of GAMs across multiple folds, prediction on the validation folds,
    and calculation of the Poisson loss for model evaluation.

    Attributes are initialized through the constructor. See `__init__` method for details.

    Methods:
        __call__(trial): The objective function to be optimized by Optuna.

    """

    def __init__(
        self,
        term_string,
        num_folds,
        n_splines_range,
        lam_num_range,
        lam_cat_range,
        X_train_dict,
        X_valid_dict,
        y_train_dict,
        y_valid_dict,
        w_train_dict,
        w_valid_dict,
        n_splines_inter_range=(4, 4),
        lam_num_inter_range=(1, 1, False),
        lam_cat_inter_range=(1, 1, False),
    ):
        """
        Initialize the OptunaGamObjectiveCV.

        Parameters:
            term_string (str): The term structure string for the GAM.
            num_folds (int): Number of cross-validation folds.
            n_splines_range (tuple): Range for number of splines (min, max).
            lam_num_range (tuple): Range for numerical smoothing parameters (min, max, log_scale).
            lam_cat_range (tuple): Range for categorical L2 regularisation parameters (min, max, log_scale).
            X_train_dict (dict): Dictionary of training feature DataFrames for each fold.
            X_valid_dict (dict): Dictionary of validation feature DataFrames for each fold.
            y_train_dict (dict): Dictionary of training target arrays for each fold.
            y_valid_dict (dict): Dictionary of validation target arrays for each fold.
            w_train_dict (dict): Dictionary of training sample weights for each fold.
            w_valid_dict (dict): Dictionary of validation sample weights for each fold.
            n_splines_inter_range (tuple, optional): Range for number of splines in interactions (min, max). Defaults to a single-factor model setup.
            lam_num_inter_range (tuple, optional): Range for numerical interaction smoothing parameters (min, max, log_scale). Defaults to a single-factor model setup.
            lam_cat_inter_range (tuple, optional): Range for categorical interaction L2 regularisation parameters (min, max, log_scale). Defaults to a single-factor model setup.
        """
        self.term_string = term_string
        self.num_folds = num_folds
        self.n_splines_range = n_splines_range
        self.lam_num_range = lam_num_range
        self.lam_cat_range = lam_cat_range
        self.n_splines_inter_range = n_splines_inter_range
        self.lam_num_inter_range = lam_num_inter_range
        self.lam_cat_inter_range = lam_cat_inter_range
        self.X_train_dict = X_train_dict
        self.X_valid_dict = X_valid_dict
        self.y_train_dict = y_train_dict
        self.y_valid_dict = y_valid_dict
        self.w_train_dict = w_train_dict
        self.w_valid_dict = w_valid_dict

    def __call__(self, trial):
        def train_gam(i):

            n_splines = trial.suggest_int(
                "n_splines", self.n_splines_range[0], self.n_splines_range[1]
            )
            lam_num = trial.suggest_float(
                "lam_num",
                self.lam_num_range[0],
                self.lam_num_range[1],
                log=self.lam_num_range[2],
            )
            lam_cat = trial.suggest_float(
                "lam_cat",
                self.lam_cat_range[0],
                self.lam_cat_range[1],
                log=self.lam_cat_range[2],
            )
            n_splines_inter = trial.suggest_int(
                "n_splines_inter",
                self.n_splines_inter_range[0],
                self.n_splines_inter_range[1],
            )
            lam_num_inter = trial.suggest_float(
                "lam_num_inter",
                self.lam_num_inter_range[0],
                self.lam_num_inter_range[1],
                log=self.lam_num_inter_range[2],
            )
            lam_cat_inter = trial.suggest_float(
                "lam_cat_inter",
                self.lam_cat_inter_range[0],
                self.lam_cat_inter_range[1],
                log=self.lam_cat_inter_range[2],
            )
            gam_model = GAM(
                eval(self.term_string),
                distribution="poisson",
                link="log",
            ).fit(
                self.X_train_dict[i],
                self.y_train_dict[i],
                weights=self.w_train_dict[i],
            )
            return gam_model

        try:
            gam_model_map = {}
            for i in range(self.num_folds):
                gam_model_map[i] = train_gam(i)

            # Perform out-of-fold model scoring
            ypred_valid_cv = {}
            for i in range(self.num_folds):
                ypred_valid_cv[i] = pd.DataFrame(index=self.X_valid_dict[i].index)

                ypred_valid_cv[i]["ypred_0"] = gam_model_map[i].predict(
                    self.X_valid_dict[i]
                )
                ypred_valid_cv[i]["weight"] = self.w_valid_dict[i]
                ypred_valid_cv[i]["ypred"] = (
                    ypred_valid_cv[i]["ypred_0"] * ypred_valid_cv[i]["weight"]
                )
                ypred_valid_cv[i]["claim_count"] = (
                    self.y_valid_dict[i] * self.w_valid_dict[i]
                )
            ypred_cv_df = pd.concat(ypred_valid_cv.values(), axis=0)

            poisson_loss = total_poisson_dev(
                ypred_cv_df["claim_count"], ypred_cv_df["ypred"]
            )

        except np.linalg.LinAlgError:
            poisson_loss = 1e8

        return poisson_loss


def extract_relativity(
    num_feat_map,
    cat_feat_map,
    data_df,
    gam_model,
    numerical_banding,
    categorical_banding,
    weight="weight",
    base_adj=1,
    base_pred=None,
    interact_term_map=None,
):
    """
    Extract adjusted intercept and related GAM relativities for numerical and categorical variables
    and interaction terms.

    Parameters:
    -----------
    num_feat_map : dict
        Mapping of the indices of numerical variables to their names.
    cat_feat_map : dict
        Mapping of the indices of categorical variables to their names.
    data_df : pandas.DataFrame
        DataFrame containing the data used to train the model -  must all the features and a weight
        column. Feature columns must be in the same order as in training.
    gam_model : object
        Trained PyGAM model object.
    numerical_banding : str
        Path to Excel file containing banding information for numerical variables.
    categorical_banding : str
        Path to Excel file containing banding information for categorical variables.
    weight : str, optional
        Name of the weight column in data_df. Default is "weight".
    base_adj : float, optional
        Base adjustment factor for the intercept. Default is 1.
    base_pred : str, optional
        Name of the base prediction column, i.e, starting points of the model. Default is None.
    interact_term_map : dict, optional
        Mapping of the indices of interaction terms to their names. Default is None.

    Returns:
    --------
    tuple
        A tuple containing:
        1. intercept (float): Adjusted intercept value.
        2. num_rel_dict (dict): Dictionary of relativities for numerical variables.
        3. cat_rel_dict (dict): Dictionary of relativities for categorical variables.
        4. inter_rel_dict (dict or None): Dictionary of relativities for interaction terms, if any.

    Raises:
    -------
    AssertionError
        If there are duplicated numerical or categorical variables, or if interaction terms
        contain variables not found in num_feat_map or cat_feat_map.

    Notes:
    ------
    - The function assumes that the last coefficient in the GAM model is the intercept.
    - For numerical variables, predictions outside the range of training data are capped at the min/max values.
    - For categorical variables, levels not seen in training data are assigned the relativity of the most frequent level.
    - Relativities are adjusted so that the weighted average level is 1 for each variable.
    - The intercept is adjusted based on the product of all average relativities and the base adjustment.
    """

    assert len(set(num_feat_map.values())) == len(
        num_feat_map
    ), "Duplicated numerical variables found."
    assert len(set(cat_feat_map.values())) == len(
        cat_feat_map
    ), "Duplicated categorical variables found."

    # Define the feature dataframe - must be the same as the one used to train the model
    feature_df = data_df.drop(columns=[weight])

    # Calculate the total number of factors used by the model
    number_factors = len(feature_df)
    # Extract the intercept value from the model (assume the last coefficient is the intercept)
    intercept = np.exp(gam_model.coef_[-1])

    intercept_adj_list = []

    # Initilise an empty dictionary for numerical variables
    num_rel_dict = {}

    if base_pred is not None:
        for k, v in num_feat_map.items():
            if v == base_pred:
                del num_feat_map[k]
                break

    for i, v in num_feat_map.items():
        banding = pd.read_excel(
            numerical_banding,
            sheet_name=v.replace("_level", ""),
        )
        index = list(range(1, len(banding) + 1))
        z = feature_df.columns.tolist().index(
            v
        )  # the column index of the numerical variable
        t = np.zeros((len(index), number_factors))
        t[:, z] = index
        # Need to make use of an internal function from PyGAM to get the partial dependence
        modelmat = gam_model.terms.build_columns(t, term=i)
        partial_dependence = np.exp(
            gam_model._linear_predictor(modelmat=modelmat, term=i)
        )
        prediction = pd.DataFrame(zip(index, partial_dependence), columns=[v, "pred"])

        # Get the min and max from feature_df - assuming this is the same as the training data
        x_min = feature_df[v].min()
        x_max = feature_df[v].max()

        # Get the predictions of the x_min and x_max
        y_min = prediction.loc[prediction[v] == x_min, "pred"].values[0]
        y_max = prediction.loc[prediction[v] == x_max, "pred"].values[0]

        # If the feature value is outside the range seen in the training data, then use the predictions of the x_min / x_max
        prediction.loc[prediction[v] <= x_min, "pred"] = y_min
        prediction.loc[prediction[v] >= x_max, "pred"] = y_max

        # Adjust the relativities such that the weighted average level is 1
        prediction["relativity"] = prediction["pred"].copy()
        num_rel_dict[v] = prediction
        num_rel_dict[v] = num_rel_dict[v].join(data_df.groupby(v)[[weight]].sum(), on=v)
        num_rel_dict[v][weight] = num_rel_dict[v][weight].fillna(0)
        num_rel_dict[v]["rel_weight"] = (
            num_rel_dict[v]["relativity"] * num_rel_dict[v][weight]
        )
        avg_rel = num_rel_dict[v]["rel_weight"].sum() / num_rel_dict[v][weight].sum()
        num_rel_dict[v]["relativity"] /= avg_rel
        intercept_adj_list.append(avg_rel)

    # Initilise an empty dictionary for categorical variables
    cat_rel_dict = {}

    for i, v in cat_feat_map.items():
        mapping = pd.read_excel(
            categorical_banding, sheet_name=v.replace("_cat_level", "")
        )
        mapping["Categorical_Level"] = mapping["Categorical_Level"].astype(str)
        mapping.sort_values("Integer_Value", inplace=True)
        index = mapping["Integer_Value"]

        mapping = pd.read_excel(
            categorical_banding, sheet_name=v.replace("_cat_level", "")
        )
        mapping["Categorical_Level"] = mapping["Categorical_Level"].astype(str)
        mapping.sort_values("Integer_Value", inplace=True)
        index = mapping["Integer_Value"]
        z = feature_df.columns.tolist().index(v)

        unique_val_train = feature_df[v].unique()

        # Any categorical level not in the training data needs to have the relativity of the most frequent level
        index_train = np.where(
            index.isin(unique_val_train), index, feature_df[v].mode().iloc[0]
        )
        len_index = len(index)

        t = np.zeros((len_index, number_factors))
        t[:, z] = index_train
        modelmat = gam_model.terms.build_columns(t, term=i)
        partial_dependence = np.exp(
            gam_model._linear_predictor(modelmat=modelmat, term=i)
        )
        pred = pd.DataFrame(zip(index, partial_dependence), columns=[v, "pred"])

        pred["relativity"] = pred["pred"].copy()
        cat_rel_dict[v] = pred
        cat_rel_dict[v] = cat_rel_dict[v].join(data_df.groupby(v)[[weight]].sum(), on=v)
        cat_rel_dict[v][weight] = cat_rel_dict[v][weight].fillna(0)
        cat_rel_dict[v]["rel_weight"] = (
            cat_rel_dict[v]["relativity"] * cat_rel_dict[v][weight]
        )
        avg_rel = cat_rel_dict[v]["rel_weight"].sum() / cat_rel_dict[v][weight].sum()
        cat_rel_dict[v]["relativity"] /= avg_rel
        intercept_adj_list.append(avg_rel)

    if interact_term_map == None:
        inter_rel_dict = None
    else:
        interaction_map1 = {k: sorted(l) for k, l in interact_term_map.items()}
        assert len(set(interaction_map1.values())) == len(
            interaction_map1
        ), "Duplicated interaction terms found."

        vars_in_interacton = list(interaction_map1.values())
        vars_in_interacton = set(list(chain(*vars_in_interacton)))
        full_vars = set(list(num_feat_map.values()) + list(cat_feat_map.values()))
        assert (
            len(vars_in_interaction - full_vars) == 0
        ), "Some variables in the interaction terms not found in either numerical or categorical dictionary"

        inter_rel_dict = {}
        for i, interaction in interact_term_map.items():

            v1, v2 = interaction
            interaction_name = v1 + " x " + v2

            if v1 in num_feat_map.values():
                banding = pd.read_excel(
                    numerical_banding,
                    sheet_name=v1.replace("_level", ""),
                )
                index1 = list(range(1, len(banding) + 1))
            else:
                # Assume that any variable not in num_feat_map is in cat_feat_map
                mapping = pd.read_excel(
                    categorical_banding, sheet_name=v1.replace("_cat_level", "")
                )
                mapping["Categorical_Level"] = mapping["Categorical_Level"].astype(str)
                mapping.sort_values("Integer_Value", inplace=True)
                index1 = mapping["Integer_Value"]

            if v2 in num_feat_map.values():
                banding = pd.read_excel(
                    numerical_banding,
                    sheet_name=v2.replace("_level", ""),
                )
                index2 = list(range(1, len(banding) + 1))
            else:
                # Assume that any variable not in num_feat_map is in cat_feat_map
                mapping = pd.read_excel(
                    categorical_banding, sheet_name=v2.replace("_cat_level", "")
                )
                mapping["Categorical_Level"] = mapping["Categorical_Level"].astype(str)
                mapping.sort_values("Integer_Value", inplace=True)
                index2 = mapping["Integer_Value"]

            output = list(product(index1, index2))
            output_df = pd.DataFrame(output, columns=[1, 2])
            output_df.sort_values([1, 2], inplace=True)

            t = pd.DataFrame(np.zeros((len(output_df), number_factors)))
            z1 = feature_df.columns.tolist().index(v1)
            z2 = feature_df.columns.tolist().index(v2)

            if v1 in num_feat_map.values():
                x_min1 = feature_df[v1].min()
                x_max1 = feature_df[v1].max()
                output_df["1_train"] = output_df[1].clip(x_min1, x_max1)
            else:
                unique_val_train1 = feature_df[v1].unique()
                output_df["1_train"] = np.where(
                    output_df[1].isin(unique_val_train1),
                    output_df[1],
                    feature_df[v1].mode().iloc[0],
                )

            if v2 in num_feat_map.values():
                x_min2 = feature_df[v2].min()
                x_max2 = feature_df[v2].max()
                output_df["2_train"] = output_df[2].clip(x_min2, x_max2)
            else:
                unique_val_train2 = feature_df[v2].unique()
                output_df["2_train"] = np.where(
                    output_df[2].isin(unique_val_train2),
                    output_df[2],
                    feature_df[v2].mode().iloc[0],
                )

            t[z1] = output_df["1_train"]
            t[z2] = output_df["2_train"]

            modelmat = gam_model.terms.build_columns(t.to_numpy(), term=i)
            output_df["pred"] = np.exp(
                gam_model._linear_predictor(modelmat=modelmat, term=i)
            )

            output_df["relativity"] = output_df["pred"].copy()
            output_df = output_df.join(
                data_df.rename(columns={v1: 1, v2: 2}).groupby([1, 2])[[weight]].sum(),
                on=[1, 2],
            )
            output_df[weight] = output_df[weight].fillna(0)
            output_df["rel_weight"] = output_df["relativity"] * output_df[weight]
            avg_rel = output_df["rel_weight"].sum() / output_df[weight].sum()
            output_df["relativity"] /= avg_rel
            intercept_adj_list.append(avg_rel)

            output_df.rename(columns={1: v1, 2: v2}, inplace=True)
            output_df.drop(columns=["1_train", "2_train"], inplace=True)

            inter_rel_dict[interaction_name] = output_df

    intercept *= np.prod(intercept_adj_list) * base_adj
    return (intercept, num_rel_dict, cat_rel_dict, inter_rel_dict)
