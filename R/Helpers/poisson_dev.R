poisson_dev <- function(y_true, y_pred, correction = 1e-9) {
  if (length(y_true) != length(y_pred)) {
    stop("y_true and y_pred must have the same length")
  }
  if (any(y_true < 0) || any(y_pred < 0)) {
    stop("y_true and y_pred must be non-negative")
  }
  
  dev <- numeric(length(y_true))
  
  non_zero <- y_true > 0
  dev[non_zero] <- y_pred[non_zero] - y_true[non_zero] - 
    y_true[non_zero] * log((y_pred[non_zero] + correction) / 
                             (y_true[non_zero] + correction))
  
  dev[!non_zero] <- y_pred[!non_zero]
  
  return(2 * mean(dev))
}