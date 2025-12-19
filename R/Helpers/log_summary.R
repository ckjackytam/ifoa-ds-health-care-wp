
log_detailed_summary <- function(data, filter_expr = NULL, reason = NULL) {
  if (!is.null(filter_expr)) {
    exclusion_vector <- eval(filter_expr, envir = data)
    dataout <- data[(!exclusion_vector), ]
    datain <- data[exclusion_vector, ]
    if((nrow(datain) + nrow(dataout)) != nrow(data) ){
      stop("Rows lost executing this operation")
    }
    summary <- dataout[, .(
      Excluded_Deaths = sum(Death_Count) %>% round(2)#,
      #Excluded_AtoE = (sum(Death_Count)/sum(expected_L)*100) %>% round(2),
      #Excluded_Avg_IssueYear = weighted.mean(IssueYear, expected_L) %>% round(2)
    )]
    summary$Excluded_nrow <- nrow(dataout)
    summary$operation <- paste(filter_expr, collapse = "")
    summary$reason <- reason
    
    if(!"filter_summary" %in% names(attributes(datain))){
      attributes(datain)$filter_summary <- summary
    } else {
      attributes(datain)$filter_summary <- rbind(
        attributes(datain)$filter_summary,
        summary
      )
    }
    if(nrow(datain) == nrow(data)){
      warning("No rows were excluded by the filter. Returning the full dataset.")
    }
    data <- datain
  } else {
    stop("Data filter required")
  }
  return(data)
}  