onehot_encode <- function(x_vars, formula) {
  ordered_factors <- sapply(x_vars, is.ordered)
  ordered_factor_names <- names(x_vars)[ordered_factors]
  
  ordinal_values <- list()
  for (ord_name in ordered_factor_names) {
    ordinal_values[[ord_name]] <- as.numeric(x_vars[[ord_name]]) - 1
  }
  
  x_vars[] <- lapply(x_vars, function(col) {
    if (is.character(col)) factor(col) else col
  })
  
  contrasts_arg <- lapply(names(x_vars), function(col_name) {
    col <- x_vars[[col_name]]
    if (is.factor(col) && !is.ordered(col)) {
      contr.treatment(levels(col), contrasts = FALSE)
    } else {
      NULL
    }
  })
  names(contrasts_arg) <- names(x_vars)
  contrasts_arg <- contrasts_arg[!sapply(contrasts_arg, is.null)]
  
  x_vars_encoded <- stats::model.matrix(
    stats::as.formula(formula),
    data = x_vars,
    contrasts.arg = contrasts_arg
  ) %>% data.table
  
  for (ord_name in ordered_factor_names) {
    poly_patterns <- c(
      paste0("^", ord_name, "\\.L$"),
      paste0("^", ord_name, "\\.Q$"),
      paste0("^", ord_name, "\\.C$"),
      paste0("^", ord_name, "\\^")
    )
    
    poly_cols <- character(0)
    for (pattern in poly_patterns) {
      poly_cols <- c(poly_cols, grep(pattern, names(x_vars_encoded), value = TRUE))
    }
    
    if (length(poly_cols) > 0) {
      x_vars_encoded[, (poly_cols) := NULL]
    }
    
    x_vars_encoded[, (ord_name) := ordinal_values[[ord_name]]]
  }
  
  return(x_vars_encoded)
}