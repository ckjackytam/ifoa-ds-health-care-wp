.build_ns <- function(x, df = NULL, knots = NULL, Boundary.knots = NULL, intercept = FALSE, col_prefix = "ns_", existing_colnames = NULL) {
  if (!is.null(knots) || !is.null(Boundary.knots)) {
    X <- ns(x, knots = knots, Boundary.knots = Boundary.knots, intercept = intercept)
  } else {
    if (is.null(df)) stop("Either df or (knots/Boundary.knots) must be provided for ns().")
    X <- ns(x, df = df, intercept = intercept)
  }
  if (is.null(existing_colnames)) {
    colnames(X) <- paste0(col_prefix, seq_len(ncol(X)))
  } else {
    colnames(X) <- existing_colnames
  }
  list(
    X = data.table(X),
    spec = list(
      type = "ns",
      knots = attr(X, "knots"),
      Boundary.knots = attr(X, "Boundary.knots"),
      intercept = intercept,
      colnames = colnames(X)
    )
  )
}

.build_bs <- function(x, df = NULL, knots = NULL, Boundary.knots = NULL, degree = 3, intercept = FALSE, col_prefix = "bs_", existing_colnames = NULL) {
  if (!is.null(knots) || !is.null(Boundary.knots)) {
    X <- bs(x, knots = knots, Boundary.knots = Boundary.knots, degree = degree, intercept = intercept)
  } else {
    if (is.null(df)) stop("Either df or (knots/Boundary.knots) must be provided for bs().")
    X <- bs(x, df = df, degree = degree, intercept = intercept)
  }
  if (is.null(existing_colnames)) {
    colnames(X) <- paste0(col_prefix, seq_len(ncol(X)))
  } else {
    colnames(X) <- existing_colnames
  }
  list(
    X = data.table(X),
    spec = list(
      type = "bs",
      knots = attr(X, "knots"),
      Boundary.knots = attr(X, "Boundary.knots"),
      degree = attr(X, "degree"),
      intercept = intercept,
      colnames = colnames(X)
    )
  )
}

gen_features <- function(dataset,
                         age_coefs = NULL, 
                         entry_age_coefs = NULL,
                         obs_year_coefs = NULL, 
                         issue_year_coefs = NULL,
                         attained_age_spline_spec = NULL,
                         duration_spline_spec    = NULL,
                         obs_year_spline_spec    = NULL,
                         factor_cols = c("Sex", "Smoker_Status", "Face_Amount_Band"),
                         map_unseen_factor_to_other = FALSE,
                         other_label = "Other") {
  
  dt <- as.data.table(dataset)
  agepoly <- if (!is.null(age_coefs)) {
    data.table(poly(dt$Attained_Age, degree = 3, coefs = age_coefs))
  } else {
    data.table(poly(dt$Attained_Age, degree = 3))
  }
  entry_agepoly <- if (!is.null(entry_age_coefs)) {
    data.table(poly(dt$Issue_Age, degree = 3, coefs = entry_age_coefs))
  } else {
    data.table(poly(dt$Issue_Age, degree = 3))
  }
  obs_yearpoly <- if (!is.null(obs_year_coefs)) {
    data.table(poly(dt$Observation_Year, degree = 3, coefs = obs_year_coefs))
  } else {
    data.table(poly(dt$Observation_Year, degree = 3))
  }
  issue_yearpoly <- if (!is.null(issue_year_coefs)) {
    data.table(poly(dt$Issue_Year, degree = 3, coefs = issue_year_coefs))
  } else {
    data.table(poly(dt$Issue_Year, degree = 3))
  }
  
  setnames(agepoly,        paste0("agepoly",         1:3))
  setnames(entry_agepoly,  paste0("issue_agepoly",   1:3))
  setnames(obs_yearpoly,   paste0("obs_yearpoly",    1:3))
  setnames(issue_yearpoly, paste0("issue_yearpoly",  1:3))
  
  if (is.null(age_coefs))        age_coefs        <- attr(poly(dt$Attained_Age, degree = 3), "coefs")
  if (is.null(entry_age_coefs))  entry_age_coefs  <- attr(poly(dt$Issue_Age, degree = 3), "coefs")
  if (is.null(obs_year_coefs))   obs_year_coefs   <- attr(poly(dt$Observation_Year, degree = 3), "coefs")
  if (is.null(issue_year_coefs)) issue_year_coefs <- attr(poly(dt$Issue_Year, degree = 3), "coefs")
  
  if (is.null(attained_age_spline_spec)) {
    ns_age <- .build_ns(
      x  = dt$Attained_Age,
      df = 5,
      col_prefix = "agespline"
    )
    agesplines <- ns_age$X
    attained_age_spline_spec <- ns_age$spec
  } else {
    ns_age <- .build_ns(
      x = dt$Attained_Age,
      knots = attained_age_spline_spec$knots,
      Boundary.knots = attained_age_spline_spec$Boundary.knots,
      intercept = isTRUE(attained_age_spline_spec$intercept),
      col_prefix = "agespline",
      existing_colnames = attained_age_spline_spec$colnames
    )
    agesplines <- ns_age$X
  }
  
  if (is.null(duration_spline_spec)) {
    bs_dur <- .build_bs(
      x  = dt$Duration,
      df = 5,
      degree = 3,
      col_prefix = "dur_bs_"
    )
    durationsplines <- bs_dur$X
    duration_spline_spec <- bs_dur$spec
  } else {
    bs_dur <- .build_bs(
      x = dt$Duration,
      knots = duration_spline_spec$knots,
      Boundary.knots = duration_spline_spec$Boundary.knots,
      degree = duration_spline_spec$degree %||% 3,
      intercept = isTRUE(duration_spline_spec$intercept),
      col_prefix = "dur_bs_",
      existing_colnames = duration_spline_spec$colnames
    )
    durationsplines <- bs_dur$X
  }
  
  if (is.null(obs_year_spline_spec)) {
    ns_obs <- .build_ns(
      x  = dt$Observation_Year,
      df = 4,
      col_prefix = "obs_ns_"
    )
    obssplines <- ns_obs$X
    obs_year_spline_spec <- ns_obs$spec
  } else {
    ns_obs <- .build_ns(
      x = dt$Observation_Year,
      knots = obs_year_spline_spec$knots,
      Boundary.knots = obs_year_spline_spec$Boundary.knots,
      intercept = isTRUE(obs_year_spline_spec$intercept),
      col_prefix = "obs_ns_",
      existing_colnames = obs_year_spline_spec$colnames
    )
    obssplines <- ns_obs$X
  }
  
  if (length(factor_cols) > 0) {
    for (cl in factor_cols) {
      if (!is.null(dt[[cl]])) {
        if (!is.factor(dt[[cl]])) dt[[cl]] <- factor(dt[[cl]])
        if (map_unseen_factor_to_other && !is.null(levels(dt[[cl]]))) {
          lv <- levels(dt[[cl]])
          if (!(other_label %in% lv)) {
            levels(dt[[cl]]) <- c(lv, other_label)
          }
          dt[[cl]][is.na(dt[[cl]])] <- other_label
        }
      }
    }
  }
  
  if ("Observation_Year" %in% names(dt)) {
    dt[, Observation_YearChar := as.character(Observation_Year)]
  }
  
  dt[, ':=' (dur0 = fifelse(Duration == 1, 1, 0),
                  invDur = 1/(Duration),
                  dur_sq = Duration ^ 2,
                  dur_cube = Duration ^ 3,
                  age_sq = Attained_Age^2,
                  age_cube = Attained_Age^3,
                  ageChar = as.character(Attained_Age),
                  issue_age_sq = Issue_Age^2,
                  issue_age_cube = Issue_Age^3,
                  issue_ageChar = as.character(Issue_Age),
                  durChar = as.character(Duration),
                  sexsmoker = paste0(Sex,Smoker_Status)
  )]
  out <- cbind(
    dt,
    agepoly, entry_agepoly, obs_yearpoly, issue_yearpoly,
    agesplines, durationsplines, obssplines
  )
  
  return(list(
    dataset = out,
    specs = list(
      poly = list(
        age_coefs        = age_coefs,
        entry_age_coefs  = entry_age_coefs,
        obs_year_coefs   = obs_year_coefs,
        issue_year_coefs = issue_year_coefs
      ),
      splines = list(
        attained_age_spline_spec = attained_age_spline_spec,
        duration_spline_spec     = duration_spline_spec,
        obs_year_spline_spec     = obs_year_spline_spec
      ),
      factors = list(
        factor_cols = factor_cols,
        map_unseen_factor_to_other = map_unseen_factor_to_other,
        other_label = other_label
      )
    )
  ))
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
