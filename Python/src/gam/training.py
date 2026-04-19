from src.utils import *
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

    This class encapsulates the objective function for Optuna to optimise GAM hyperparameters using cross validation.
    It handles the training of GAMs across multiple folds, prediction on the validation folds,
    and calculation of the Poisson loss for model evaluation.

    Attributes are initialized through the constructor. See `__init__` method for details.

    Methods:
        __call__(trial): The objective function to be optimised by Optuna.

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
        -----------
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


def extract_gam_relativity_num(
    gam_model: GAM,
    X_train: pd.DataFrame,
    w_train: pd.Series,
    num_feat_map: dict[int, str],
    weight: str,
    intercept_adj_list: list[float] = [],
) -> tuple[dict[str, pd.DataFrame], list[float]]:
    """
    Extract relativities for numerical features from a trained GAM model.

    Args:
        gam_model: Fitted PyGAM model.
        X_train: Training feature dataframe.
        w_train: Training weights series.
        num_feat_map: Dictionary mapping term indices to numerical feature names.
        weight: Column name in the relativity DataFrames that contains the weights
        intecept_adj_list: A list to store the average relativity for each feature, which can be used to adjust the model intercept.

    Returns:
        Tuple containing:
        - Dict mapping each feature name to a DataFrame with relativities
        - List of intercept adjustments

    Raises:
        AssertionError: If there are missing levels in the numerical range of any feature.
    """

    # Initilise an empty dictionary for numerical variables
    num_rel_dict = {}

    for k, v in num_feat_map.items():

        # Get the unique levels associated with a numerical variable
        index = np.sort(X_train[v].unique())

        assert (
            np.unique(np.diff(index))[0] == 1 and len(np.unique(np.diff(index))) == 1
        ), "There are probably missing levels in the numerical range. Please check!!"

        # Initialise a 0-array as an input to calculate partial dependency
        t = np.zeros((len(index), len(X_train.columns)))

        # Get the column index of the numerical variable
        z = X_train.columns.tolist().index(v)

        # Populate the column related to the variable for which we seek partial dependency
        t[:, z] = index

        # Need to use an internal function from PyGAM to get the partial dependence
        modelmat = gam_model.terms.build_columns(t, term=k)
        partial_dependence = np.exp(
            gam_model._linear_predictor(modelmat=modelmat, term=k)
        )

        # Concatenate the feature values and PDs
        prediction = pd.DataFrame(zip(index, partial_dependence), columns=[v, "pred"])

        # Adjust the PDs such that its weighted average level is 1
        prediction["relativity"] = prediction["pred"].copy()
        prediction = prediction.join(
            X_train.join(w_train).groupby(v)[[weight]].sum(), on=v
        )
        prediction["rel_weight"] = prediction["relativity"] * prediction[weight]
        avg_rel = prediction["rel_weight"].sum() / prediction[weight].sum()
        prediction["relativity"] /= avg_rel

        # Record the multiplier (this needs to be applied to intercept)
        intercept_adj_list.append(avg_rel)
        num_rel_dict[v] = prediction[[v, "relativity", weight]]
    return (num_rel_dict, intercept_adj_list)


def extract_gam_relativity_cat(
    gam_model: GAM,
    X_train: pd.DataFrame,
    w_train: pd.Series,
    cat_feat_map: dict[int, str],
    weight: str,
    cat_map_table: str,
    intercept_adj_list: list[float] = [],
) -> tuple[dict[str, pd.DataFrame], list[float]]:
    """
    Extract relativities for categorical features from a trained GAM model.

    Args:
        gam_model: Fitted PyGAM model.
        X_train: Training feature dataframe.
        w_train: Training weights series.
        cat_feat_map: Dictionary mapping term indices to categorical feature names.
        weight: Column name in the relativity DataFrames that contains the weights.
        cat_map_table: Location of where the categorical-level-to-integer mapping spreadsheet for categorical variables is saved.
        intecept_adj_list: A list to store the average relativity for each feature, which can be used to adjust the model intercept.

    Returns:
        Tuple containing:
        - Dict mapping each feature name to a DataFrame with relativities
        - List of intercept adjustments

    Raises:
        AssertionError: If there are missing levels in the numerical range of any feature.
    """

    # Get the mapping tables in the form of a dictionary
    cat_map_dict = pd.read_excel(cat_map_table, sheet_name=None)

    # Initilise an empty dictionary for numerical variables
    cat_rel_dict = {}

    for k, v in cat_feat_map.items():
        # Define the clean variable name
        v_clean = v.replace("_cat_level", "")

        # Get the unique levels associated with a categorical variable
        index = np.sort(X_train[v].unique())

        assert (
            np.unique(np.diff(index))[0] == 1 and len(np.unique(np.diff(index))) == 1
        ), "There are probably missing levels in the categorical range. Please check!!"

        # Initialise a 0-array as an input to calculate partial dependency
        t = np.zeros((len(index), len(X_train.columns)))

        # Get the column index of the numerical variable
        z = X_train.columns.tolist().index(v)

        # Populate the column related to the variable for which we seek partial dependency
        t[:, z] = index

        # Need to use an internal function from PyGAM to get the partial dependence
        modelmat = gam_model.terms.build_columns(t, term=k)
        partial_dependence = np.exp(
            gam_model._linear_predictor(modelmat=modelmat, term=k)
        )

        # Concatenate the feature values and PDs
        prediction = pd.DataFrame(zip(index, partial_dependence), columns=[v, "pred"])

        # Adjust the PDs such that its weighted average level is 1
        prediction["relativity"] = prediction["pred"].copy()
        prediction = prediction.join(
            X_train.join(w_train).groupby(v)[[weight]].sum(), on=v
        )
        prediction["rel_weight"] = prediction["relativity"] * prediction[weight]
        avg_rel = prediction["rel_weight"].sum() / prediction[weight].sum()
        prediction["relativity"] /= avg_rel

        # Get the categorical levels
        table = cat_map_dict[v_clean]
        prediction[v_clean] = prediction[v].map(
            table.set_index("Integer_Value")["Categorical_Level"].to_dict()
        )

        # Record the multiplier (this needs to be applied to intercept)
        intercept_adj_list.append(avg_rel)
        cat_rel_dict[v] = prediction[[v, v_clean, "relativity", weight]]
    return (cat_rel_dict, intercept_adj_list)


def extract_gam_relativity_inter(
    gam_model: GAM,
    X_train: pd.DataFrame,
    w_train: pd.Series,
    inter_feat_map: dict[int, str],
    weight: str,
    cat_map_table: str,
    intercept_adj_list: list[float] = [],
) -> tuple[dict[str, pd.DataFrame], list[float]]:
    """
    Extract relativities for categorical features from a trained GAM model.

    Args:
        gam_model: Fitted PyGAM model.
        X_train: Training feature dataframe.
        w_train: Training weights series.
        inter_feat_map: Dictionary mapping term indices to categorical feature names.
        weight: Column name in the relativity DataFrames that contains the weights
        cat_map_table: Location of where the categorical-level-to-integer mapping spreadsheet for categorical variables is saved.
        intecept_adj_list: A list to store the average relativity for each feature, which can be used to adjust the model intercept.

    Returns:
        Tuple containing:
        - Dict mapping each interaction name to a DataFrame with relativities
        - List of intercept adjustments

    Raises:
        AssertionError: If there are missing levels in the numerical range of any feature.
    """


    inter_rel_dict = {}
    vars_in_interaction = list(inter_feat_map.values())
    vars_in_interaction = set(list(chain(*vars_in_interaction)))

    # Get the mapping tables in the form of a dictionary
    cat_map_dict = pd.read_excel(cat_map_table, sheet_name=None)

    for k, v in inter_feat_map.items():

        v1, v2 = v
        interaction_name = v1 + " x " + v2

        index1 = np.sort(X_train[v1].unique())
        index2 = np.sort(X_train[v2].unique())
        v1_clean = v1.replace("_cat_level", "")
        v2_clean = v2.replace("_cat_level", "")

        output = list(product(index1, index2))
        output = np.array(sorted(output, key=lambda x: (x[0], x[1])))

        # Initialise a 0-array as an input to calculate partial dependency
        t = np.zeros((output.shape[0], len(X_train.columns)))
        z1 = X_train.columns.tolist().index(v1)
        z2 = X_train.columns.tolist().index(v2)

        # Populate the column related to the variable for which we seek partial dependency
        t[:, z1] = output[:, 0]
        t[:, z2] = output[:, 1]

        modelmat = gam_model.terms.build_columns(t, term=k)
        partial_dependence = np.exp(gam_model._linear_predictor(modelmat=modelmat, term=k))

        # Concatenate the feature values and PDs
        prediction = pd.DataFrame(
            np.hstack((output, partial_dependence.reshape(-1, 1))), columns=[v1, v2, "pred"]
        )
        # Adjust the PDs such that its weighted average level is 1
        prediction["relativity"] = prediction["pred"].copy()
        prediction = prediction.join(
            X_train.join(w_train).groupby([v1, v2])[[weight]].sum(), on=[v1, v2]
        )
        prediction[weight] = prediction[weight].fillna(0)
        prediction["rel_weight"] = prediction["relativity"] * prediction[weight]
        avg_rel = prediction["rel_weight"].sum() / prediction[weight].sum()
        prediction["relativity"] /= avg_rel
        intercept_adj_list.append(avg_rel)

        # Get the categorical levels
        if v1_clean in cat_map_dict.keys(): 
            table = cat_map_dict[v1_clean]
            prediction[v1_clean] = prediction[v1].map(
                table.set_index("Integer_Value")["Categorical_Level"].to_dict()
            )
        if v2_clean in cat_map_dict.keys(): 
            table = cat_map_dict[v2_clean]
            prediction[v2_clean] = prediction[v2].map(
                table.set_index("Integer_Value")["Categorical_Level"].to_dict()
            )
        col_list = [v1, v2, "relativity", weight]
        if v1_clean in cat_map_dict.keys(): 
            col_list = col_list + [v1_clean]
        if v2_clean in cat_map_dict.keys(): 
            col_list = col_list + [v2_clean]

        inter_rel_dict[interaction_name] = prediction[col_list]

    return (inter_rel_dict, intercept_adj_list)


