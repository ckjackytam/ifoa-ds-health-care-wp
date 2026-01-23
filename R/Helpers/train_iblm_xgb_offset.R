train_iblm_xgb_offset <- function (df_list, response_var, family = "poisson", params = list(), 
                                   nrounds = 1000, objective = NULL, custom_metric = NULL, 
                                   verbose = 0, print_every_n = 1L, early_stopping_rounds = 25, 
                                   maximize = NULL, save_period = NULL, save_name = "xgboost.model", 
                                   xgb_model = NULL, callbacks = list(), ..., strip_glm = TRUE,
                                   offset_var = NULL) 
{
  IBLM:::check_required_names(df_list, c("train", "validate"))
  IBLM:::check_required_names(df_list[["train"]], response_var)
  IBLM:::check_required_names(df_list[["validate"]], response_var)
  stopifnot(length(response_var) == 1, names(df_list[["train"]]) == 
              names(df_list[["validate"]]))
  
  if (!is.null(offset_var)) {
    IBLM:::check_required_names(df_list[["train"]], offset_var)
    IBLM:::check_required_names(df_list[["validate"]], offset_var)
    stopifnot(length(offset_var) == 1)
    
    train_offset <- df_list[["train"]][[offset_var]]
    validate_offset <- df_list[["validate"]][[offset_var]]
    
    if (any(is.na(train_offset)) || any(is.na(validate_offset))) {
      cli::cli_abort("Offset variable cannot contain NA values")
    }
    if (any(!is.finite(train_offset)) || any(!is.finite(validate_offset))) {
      cli::cli_abort("Offset variable must contain finite values")
    }
  }
  
  if (sum(is.na(df_list$train), is.na(df_list$validate), is.na(df_list$test)) > 
      0) {
    cli::cli_abort("'df_list' cannot contain NA values")
  }
  if (any(vapply(df_list$train, is.character, logical(1)))) {
    cli::cli_abort("'df_list' cannot contain character columns. Convert to factor.")
  }
  IBLM:::check_data_variability(df_list[["train"]], response_var)
  train <- list()
  validate <- list()
  
  predictor_vars <- setdiff(names(df_list[["train"]]), c(response_var, offset_var))
  train$responses <- dplyr::pull(df_list[["train"]], response_var)
  validate$responses <- dplyr::pull(df_list[["validate"]], 
                                    response_var)
  train$features <- dplyr::select(df_list[["train"]], dplyr::all_of(predictor_vars))
  validate$features <- dplyr::select(df_list[["validate"]], 
                                     dplyr::all_of(predictor_vars))
  
  if (family == "poisson") {
    glm_family <- stats::poisson()
  }
  else if (family == "gamma") {
    glm_family <- stats::Gamma(link = "log")
  }
  else if (family == "tweedie") {
    glm_family <- statmod::tweedie(var.power = 1.5, link.power = 0)
    glm_family$link <- "log"
  }
  else if (family == "gaussian") {
    glm_family <- stats::gaussian()
  }
  else {
    stop(paste0("family was ", family, " but should be one of: poisson, gamma, tweedie, gaussian"))
  }
  xgb_family_params <- list()
  if (is.null(objective)) {
    if (family == "poisson") {
      xgb_family_params <- utils::modifyList(xgb_family_params, 
                                             list(objective = "count:poisson"))
    }
    else if (family == "gamma") {
      xgb_family_params <- utils::modifyList(xgb_family_params, 
                                             list(objective = "reg:gamma"))
    }
    else if (family == "tweedie") {
      xgb_family_params <- utils::modifyList(xgb_family_params, 
                                             list(tweedie_variance_power = 1.5, objective = "reg:tweedie"))
    }
    else if (family == "gaussian") {
      xgb_family_params <- utils::modifyList(xgb_family_params, 
                                             list(objective = "reg:squarederror"))
    }
    else {
      stop(paste0("family was ", family, " but should be one of: poisson, gamma, tweedie, gaussian"))
    }
  }
  else {
    cli::cli_alert_info("The 'objective' was defined in input and used over default settings")
  }
  predictor_vars <- setdiff(names(df_list[["train"]]), c(response_var, offset_var))
  formula <- stats::as.formula(paste(response_var, "~", paste(predictor_vars, collapse = " + ")))

  if(!is.null(offset_var)){
    test_glm_offset <- function(df_list, response_var, offset_var) {
      predictor_vars <- setdiff(names(df_list[["train"]]), c(response_var, offset_var))
      formula_string <- paste(response_var, "~", paste(predictor_vars, collapse = " + "), "+ offset(log(", offset_var, "))")
      formula_with_offset <- as.formula(formula_string)
      
      glm_model <- stats::glm(formula_with_offset, data = df_list[["train"]], 
                              family = poisson())
      
      return(glm_model)
    }
    
    glm_model <- test_glm_offset(df_list, response_var, offset_var)
  } else {
    glm_model <- stats::glm(formula, data = df_list[["train"]], 
                            family = glm_family)
  }
  
  link <- glm_family$link
  
  train$glm_preds <- unname(stats::predict(glm_model, newdata = df_list[["train"]], 
                                           type = "response"))
  validate$glm_preds <- unname(stats::predict(glm_model, newdata = df_list[["validate"]], 
                                              type = "response"))
  
  if (link == "log") {
    train$targets <- train$responses/train$glm_preds
    validate$targets <- validate$responses/validate$glm_preds
    relationship <- "multiplicative"
  }
  else if (link == "identity") {
    train$targets <- train$responses - train$glm_preds
    validate$targets <- validate$responses - validate$glm_preds
    relationship <- "additive"
  }
  else {
    stop(paste0("link function was ", link, " but should be one of: log, identity"))
  }
  if (!is.null(offset_var)) {
    train_exposure <- df_list[["train"]][[offset_var]]
    validate_exposure <- df_list[["validate"]][[offset_var]]
    
    train$xgb_matrix <- xgboost::xgb.DMatrix(train$features, 
                                             label = train$targets,
                                             weight = train_exposure)
    validate$xgb_matrix <- xgboost::xgb.DMatrix(validate$features, 
                                                label = validate$targets,
                                                weight = validate_exposure)
  } else {
    train$xgb_matrix <- xgboost::xgb.DMatrix(train$features, 
                                             label = train$targets)
    validate$xgb_matrix <- xgboost::xgb.DMatrix(validate$features, 
                                                label = validate$targets)
  }
  xgb_additional_params <- c(list(nrounds = nrounds, objective = objective,
                                  custom_metric = custom_metric, verbose = verbose, print_every_n = print_every_n, 
                                  early_stopping_rounds = early_stopping_rounds, maximize = maximize, 
                                  save_period = save_period, save_name = save_name, xgb_model = xgb_model, 
                                  callbacks = callbacks), list(...))
  params_to_overwrite <- intersect(names(xgb_family_params), 
                                   names(params))
  if (length(params_to_overwrite) > 0) {
    cli::cli_alert_info("The following 'params' were defined in input and used over default settings: {.val {params_to_overwrite}}")
  }
  params <- utils::modifyList(xgb_family_params, params)
  xgb_core_params <- list(params = params, data = train$xgb_matrix, 
                          evals = list(validation = validate$xgb_matrix))
  xgb_all_params <- utils::modifyList(xgb_core_params, xgb_additional_params)
  booster_model <- do.call(xgboost::xgb.train, xgb_all_params)
  if (strip_glm) {
    stripGlmLR <- function(cm) {
      cm$y <- c()
      cm$residuals <- c()
      cm$fitted.values <- c()
      cm$data <- c()
      cm
    }
    glm_model <- stripGlmLR(glm_model)
  }
  iblm_model <- list()
  iblm_model$glm_model <- glm_model
  iblm_model$booster_model <- booster_model
  iblm_model$data$train <- df_list$train
  iblm_model$data$validate <- df_list$validate
  iblm_model$relationship <- relationship
  glm_beta_coeff <- iblm_model$glm_model$coefficients
  coef_names_glm <- names(glm_beta_coeff)
  vartypes <- unlist(lapply(dplyr::select(df_list$train, -dplyr::all_of(c(response_var, offset_var))), 
                            typeof))
  varclasses <- unlist(lapply(dplyr::select(df_list$train, 
                                            -dplyr::all_of(c(response_var, offset_var))), function(x) class(x)[1]))
  predictor_vars <- list()
  predictor_vars$all <- setdiff(names(vartypes), response_var)
  predictor_vars$categorical <- predictor_vars$all[(!vartypes %in% 
                                                      c("integer", "double") | varclasses == "factor")]
  predictor_vars$continuous <- setdiff(predictor_vars$all, 
                                       predictor_vars$categorical)
  cat_levels <- list()
  coeff_names <- list()
  cat_levels$all <- lapply(dplyr::select(df_list$train, dplyr::all_of(predictor_vars$categorical)), 
                           function(x) sort(unique(x)))
  cat_levels$reference <- stats::setNames(lapply(names(cat_levels$all), 
                                                 function(var) {
                                                   all_levels <- cat_levels$all[[var]]
                                                   present_levels <- coef_names_glm[startsWith(coef_names_glm, 
                                                                                               var)]
                                                   present_levels_clean <- gsub(paste0("^", var), "", 
                                                                                present_levels)
                                                   setdiff(all_levels, present_levels_clean)
                                                 }), names(cat_levels$all))
  coeff_names$all_cat <- unlist(lapply(names(cat_levels$all), 
                                       function(x) paste0(x, cat_levels$all[[x]])))
  coeff_names$all <- c("(Intercept)", predictor_vars$continuous, 
                       coeff_names$all_cat)
  coeff_names$reference_cat <- setdiff(coeff_names$all, coef_names_glm)
  iblm_model$response_var <- response_var
  iblm_model$predictor_vars <- predictor_vars
  iblm_model$cat_levels <- cat_levels
  iblm_model$coeff_names <- coeff_names
  iblm_model$xgb_params <- IBLM:::drop_xgb_data_params(xgb_all_params)
  iblm_model$offset_var <- offset_var
  class(iblm_model) <- "iblm"
  return(iblm_model)
}

predict.iblm <- function (object, newdata, trim = NA_real_, type = "response", 
                          ...) 
{
  IBLM:::check_iblm_model(object)
  if (type != "response") {
    cli::cli_abort(c(x = "Only supported type currently is {.val response}", 
                     i = "You supplied {.val {type}}"))
  }
  response_var <- all.vars(object$glm_model$formula)[1]
  offset_var <- object$offset_var
  
  vars_to_exclude <- response_var
  data <- dplyr::select(newdata, -dplyr::any_of(vars_to_exclude))
  
  relationship <- object["relationship"]
  
  if (!is.null(offset_var)) {
    # Check that offset variable exists in newdata
    if (!offset_var %in% names(newdata)) {
      cli::cli_abort(c(x = "Offset variable {.val {offset_var}} not found in newdata",
                       i = "The model was trained with an offset, so predictions require it too"))
    }
  }
  
  glm <- unname(stats::predict(object$glm_model, newdata = data, type = type))
  
  vars_to_exclude_xgb <- c(response_var, offset_var)
  vars_to_exclude_xgb <- vars_to_exclude_xgb[!is.null(vars_to_exclude_xgb)]
  xgb_data <- dplyr::select(newdata, -dplyr::any_of(vars_to_exclude_xgb))
  
  booster <- stats::predict(object$booster_model, xgboost::xgb.DMatrix(xgb_data), 
                            type = type)
  if (!is.na(trim)) {
    truncate <- function(x) {
      return(pmax(pmin(booster, 1 + trim), max(1 - trim, 
                                               0)))
    }
    booster <- truncate(booster)
    booster <- booster * 1/mean(booster)
  }
  if (relationship == "multiplicative") {
    toreturn <- glm * booster
  }
  else if (relationship == "additive") {
    toreturn <- glm + booster
  }
  else {
    cli::cli_abort(c(x = "Invalid relationship attribute: {.val {relationship}}", 
                     i = "Relationship must be either {.val multiplicative} or {.val additive}"))
  }
  return(toreturn)
}