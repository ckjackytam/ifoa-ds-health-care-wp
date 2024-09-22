from src.utility import *
import numpy as np
from pygam import GAM, s, te, f, l, utils
import pandas as pd


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
    mono_dec_var_list: list of str
        List of variables with monotonic decreasing constraints.
    feature_df: pandas.DataFrame
        DataFrame containing the training feature data.
    interaction_term_list : list of tuples, optional
        List of variable pairs for interaction terms. Each tuple contains two variable names. Default is None.
    base_pred : str, optional 
        Column name of the starting points for GAM. Default is to build the model from scratch. 
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
            interact_term_map[is_base_present + len(num_vars) + len(cat_vars) + q] = (v1, v2)

    return (term_string, num_feat_map, cat_feat_map, interact_term_map)


class OptunaGamObjectiveCV:
    """
    A custom objective class for Optuna optimization of Generalized Additive Models (GAMs).

    This class encapsulates the objective function for Optuna to optimize GAM hyperparameters using cross validation.
    It handles the training of GAMs across multiple folds, prediction on the validation folds,
    and calculation of the Poisson loss for model evaluation.

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
        Initialize the OptunaGamObjective.

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
                ypred_valid_cv[i]["claim_count"] = self.y_valid_dict[i] * self.w_valid_dict[i]
            ypred_cv_df = pd.concat(ypred_valid_cv.values(), axis=0)

            poisson_loss = total_poisson_dev(
                    ypred_cv_df["claim_count"], ypred_cv_df["ypred"]
                )

        except np.linalg.LinAlgError:
            poisson_loss = 1e8

        return poisson_loss
