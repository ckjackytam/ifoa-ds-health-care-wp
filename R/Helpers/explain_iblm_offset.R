explain_iblm_offset <- function (iblm_model, data, migrate_reference_to_bias = TRUE) 
{
  check_iblm_model(iblm_model)
  shap <- extract_booster_shap(iblm_model$booster_model, data)
  wide_input_frame <- data_to_onehot_fix(data, iblm_model)
  shap_wide <- shap_to_onehot_fix(shap, wide_input_frame, iblm_model)
  beta_corrections <- beta_corrections_derive_offset(shap_wide, wide_input_frame, 
                                              iblm_model, migrate_reference_to_bias)
  data_glm <- data_beta_coeff_glm_offset(data, iblm_model)
  data_booster <- data_beta_coeff_booster(data, beta_corrections, 
                                          iblm_model)
  data_beta_coeff <- data_glm + data_booster
  list(shap = shap, beta_corrections = beta_corrections, data_beta_coeff = data_beta_coeff, 
       beta_corrected_scatter = create_beta_corrected_scatter(data_beta_coeff = data_beta_coeff, 
                                                              data = data, iblm_model = iblm_model), beta_corrected_density = create_beta_corrected_density(wide_input_frame = wide_input_frame, 
                                                                                                                                                            beta_corrections = beta_corrections, data = data, 
                                                                                                                                                            iblm_model = iblm_model), bias_density = create_bias_density(migrate_reference_to_bias = migrate_reference_to_bias, 
                                                                                                                                                                                                                         shap = shap, data = data, iblm_model = iblm_model), 
       overall_correction = create_overall_correction(shap = shap, 
                                                      iblm_model = iblm_model))
}


beta_corrections_derive_offset <- function(shap_wide, wide_input_frame, iblm_model,
                                           migrate_reference_to_bias = TRUE) {
  
  check_iblm_model(iblm_model)
  coef_names_reference_cat <- iblm_model$coeff_names$reference_cat
  predictor_vars_continuous <- iblm_model$predictor_vars$continuous
  
  beta_corrections <- shap_wide
  
  # continuous-only mask for zero values
  if (length(predictor_vars_continuous) > 0) {
    x <- dplyr::select(wide_input_frame, dplyr::all_of(predictor_vars_continuous))
    shap <- dplyr::select(shap_wide, dplyr::all_of(predictor_vars_continuous))
    
    mask <- dplyr::mutate(x, dplyr::across(dplyr::everything(), ~ dplyr::if_else(.x == 0, 1, 0)))
    shap_for_zeros <- rowSums(as.matrix(mask) * as.matrix(shap), na.rm = TRUE)
  } else {
    shap_for_zeros <- rep(0, nrow(beta_corrections))
  }
  
  # migrate baseline ref columns to bias (ONLY if they actually exist in shap_wide)
  coef_names_reference_cat <- intersect(coef_names_reference_cat, names(shap_wide))
  
  if (migrate_reference_to_bias && length(coef_names_reference_cat) > 0) {
    shap_for_cat_ref <- rowSums(as.matrix(dplyr::select(shap_wide, dplyr::all_of(coef_names_reference_cat))),
                                na.rm = TRUE)
    beta_corrections <- dplyr::mutate(beta_corrections,
                                      dplyr::across(dplyr::all_of(coef_names_reference_cat), ~ 0))
  } else {
    shap_for_cat_ref <- rep(0, nrow(beta_corrections))
  }
  
  beta_corrections$bias <- beta_corrections$bias + shap_for_zeros + shap_for_cat_ref # Zeroes and reference cats added to intercept
  
  # Convert SHAP->"beta-like" only where x != 0; leave NA otherwise (don’t silently zero)
  if (length(predictor_vars_continuous) > 0) {
    beta_corrections <- 
      dplyr::mutate(
        beta_corrections,
        dplyr::across(dplyr::all_of(predictor_vars_continuous), \(sh) {
          xcol <- wide_input_frame[[dplyr::cur_column()]]
          xnum <- suppressWarnings(as.numeric(xcol))
          ok <- is.finite(xnum) & (xnum != 0)
          out <- rep(0, length(sh))
          out[ok] <- sh[ok] / xnum[ok]
          out
        })
      )
  }
  beta_corrections
}

data_beta_coeff_glm_offset <- function(data,
                                       iblm_model,
                                       return_offset = FALSE) {
  
  check_iblm_model(iblm_model)
  
  response_var <- iblm_model$response_var
  glm_mod <- iblm_model$glm_model
  glm_beta_coeff <- glm_mod$coefficients
  
  levels_all_cat <- iblm_model$cat_levels$all
  levels_reference_cat <- iblm_model$cat_levels$reference
  predictor_vars_continuous <- iblm_model$predictor_vars$continuous
  predictor_vars_categorical <- iblm_model$predictor_vars$categorical
  
  # ---- Intercept (handle models fitted with -1 / no intercept) ----
  intercept <- if ("(Intercept)" %in% names(glm_beta_coeff)) {
    unname(glm_beta_coeff[["(Intercept)"]])
  } else {
    0
  }
  
  # ---- Compute offset for *new* data (works even if no offset) ----
  tt <- stats::terms(glm_mod)
  tt_noy <- stats::delete.response(tt)
  
  mf <- stats::model.frame(
    tt_noy,
    data = data,
    na.action = stats::na.pass,
    xlev = glm_mod$xlevels
  )
  
  off <- stats::model.offset(mf)
  if (is.null(off)) off <- rep(0, nrow(data))
  off <- as.numeric(off)
  
  # ---- Precompute categorical coefficient lookup per variable ----
  glm_coeffs_all_cat <- purrr::imap(levels_all_cat, function(levels, var) {
    
    ref <- levels_reference_cat[[var]]
    
    # named vector of coefficients by level (reference is 0)
    out <- stats::setNames(rep(0, length(levels)), levels)
    
    nonref_levels <- levels[levels != ref]
    if (length(nonref_levels) > 0) {
      
      coef_names <- paste0(var, nonref_levels)
      vals <- glm_beta_coeff[coef_names]
      
      # if some coefficients are missing (e.g. aliased), treat as 0
      vals[is.na(vals)] <- 0
      
      out[nonref_levels] <- unname(vals)
    }
    
    out
  })
  
  # ---- Build output frame (without offset initially) ----
  out <- dplyr::select(data, -dplyr::any_of(response_var)) |>
    dplyr::mutate(
      dplyr::across(
        dplyr::all_of(predictor_vars_categorical),
        ~ glm_coeffs_all_cat[[dplyr::cur_column()]][as.character(.)]
      ),
      dplyr::across(
        dplyr::all_of(predictor_vars_continuous),
        ~ {
          b <- glm_beta_coeff[dplyr::cur_column()]
          b <- if (is.na(b)) 0 else unname(b)
          b
        }
      )
    ) |>
    dplyr::mutate(
      bias = intercept,
      .before = 1
    )
  
  # ---- Optionally add offset column ----
  if (return_offset) {
    out <- dplyr::mutate(out, offset = off, .after = bias)
  }
  
  out
}

shap_to_onehot_fix <- function(shap, wide_input_frame, iblm_model) {
  
  check_iblm_model(iblm_model)
  
  levels_all_cat <- iblm_model$cat_levels$all
  response_var   <- iblm_model$response_var
  
  no_cat_toggle <- length(iblm_model$predictor_vars$categorical) == 0
  
  # Helper: normalise names the same way base R tends to (spaces/punct -> '.')
  norm <- function(x) make.names(x, unique = TRUE)
  
  if (no_cat_toggle) {
    shap_wide <- dplyr::mutate(shap, bias = shap$BIAS[1], .before = dplyr::everything())
    return(shap_wide)
  }
  
  # Drop non-feature columns if present
  wide_input_frame <- dplyr::select(
    wide_input_frame,
    -dplyr::any_of(c("(Intercept)", response_var))
  )
  
  # Work with normalised versions of the input frame names
  wide_names_raw  <- names(wide_input_frame)
  wide_names_norm <- norm(wide_names_raw)
  
  # Create a renaming map: normalised -> raw
  # (lets us select normalised safely, then map back to raw columns)
  norm_to_raw <- stats::setNames(wide_names_raw, wide_names_norm)
  
  cat_frame_list <- lapply(names(levels_all_cat), function(var) {
    lvl <- levels_all_cat[[var]]
    
    # Expected one-hot columns (raw intention)
    expected_raw  <- paste0(var, lvl)
    expected_norm <- norm(expected_raw)
    
    # Keep only those that exist in wide_input_frame after normalisation
    present_norm <- intersect(expected_norm, wide_names_norm)
    
    if (length(present_norm) == 0) {
      # Fallback: try to find columns that start with var (normalised)
      # (helps when upstream used separators like '_' or ':' differently)
      var_norm <- norm(var)
      candidate_norm <- wide_names_norm[startsWith(wide_names_norm, var_norm)]
      
      # If still nothing, return an empty tibble with 0 cols
      if (length(candidate_norm) == 0) {
        return(dplyr::tibble())
      }
      
      present_norm <- candidate_norm
    }
    
    # Map normalised names back to the real column names in the data
    present_raw <- unname(norm_to_raw[present_norm])
    
    mask <- dplyr::select(wide_input_frame, dplyr::all_of(present_raw))
    
    # shap[, var] is the SHAP value for the *categorical variable as a whole*
    # replicate it across levels and then keep only the active one-hot via mask
    mat <- matrix(rep(shap[, var], ncol(mask)), byrow = FALSE, ncol = ncol(mask))
    out <- mat * as.matrix(mask)
    
    # Preserve the *actual* column names from the mask
    out <- as.data.frame(out)
    names(out) <- names(mask)
    out
  })
  
  cat_frame <- dplyr::bind_cols(cat_frame_list)
  
  # Combine:
  # - shap without categorical vars (since we expanded them)
  # - expanded categorical one-hots
  # - keep columns in the same order as wide_input_frame
  shap_noncat <- dplyr::select(shap, -dplyr::any_of(names(cat_frame)))
  
  shap_wide <- cbind(shap_noncat, cat_frame)
  
  # Reorder to match wide_input_frame columns where possible
  keep_order <- intersect(names(wide_input_frame), colnames(shap_wide))
  shap_wide  <- dplyr::select(shap_wide, dplyr::all_of(keep_order), dplyr::everything())
  
  # Add bias
  shap_wide <- dplyr::mutate(shap_wide, bias = shap$BIAS[1], .before = dplyr::everything())
  
  shap_wide
}

data_to_onehot_fix <- function(data, iblm_model, remove_target = TRUE) {
  check_iblm_model(iblm_model)
  
  coef_names_all <- iblm_model$coeff_names$all
  levels_all_cat <- iblm_model$cat_levels$all
  response_var   <- iblm_model$response_var
  
  # If no categoricals, just return (optionally drop target)
  if (length(iblm_model$predictor_vars$categorical) == 0) {
    out <- data
    if (remove_target) out <- dplyr::select(out, -dplyr::any_of(response_var))
    return(out)
  }
  
  # Start with a full zero matrix using the *exact* coefficient names
  output_frame <- as.data.frame(matrix(0, nrow = nrow(data), ncol = length(coef_names_all)))
  names(output_frame) <- coef_names_all
  
  # Intercept if present
  if ("(Intercept)" %in% names(output_frame)) {
    output_frame[["(Intercept)"]] <- 1
  }
  
  # Fill continuous predictors (works with spaces/special chars via [[ ]])
  cont_vars <- intersect(iblm_model$predictor_vars$continuous, names(data))
  for (v in cont_vars) {
    if (v %in% names(output_frame)) {
      output_frame[[v]] <- data[[v]]
    }
  }
  
  # Fill categorical one-hot columns using stored training levels
  # Columns are assumed to be named paste0(var, level), exactly as in your coeff_names$all_cat
  cat_vars <- intersect(names(levels_all_cat), names(data))
  for (v in cat_vars) {
    # compare on character to avoid factor level mismatches
    x <- as.character(data[[v]])
    lvls <- as.character(levels_all_cat[[v]])
    
    for (lvl in lvls) {
      col <- paste0(v, lvl)
      if (col %in% names(output_frame)) {
        output_frame[[col]] <- as.integer(x == lvl)
      }
    }
  }
  
  # Remove target if requested
  if (remove_target) {
    output_frame <- dplyr::select(output_frame, -dplyr::any_of(response_var))
  }
  
  output_frame
}
