onehot_encode <- function(x_vars, formula){
  x_vars[] <- lapply(x_vars, function(col) {
    if (is.character(col)) factor(col) else col
  })
  contrasts_arg <- lapply(x_vars, function(col) {
    if (is.factor(col)) contr.treatment(levels(col), contrasts = FALSE) else NULL
  })
  contrasts_arg <- contrasts_arg[!sapply(contrasts_arg, is.null)]
  x_vars <- stats::model.matrix(
    stats::as.formula(formula),
    data = x_vars,
    contrasts.arg = contrasts_arg
  ) %>% data.table
  
  return(x_vars)}