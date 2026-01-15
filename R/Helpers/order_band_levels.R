order_band_levels <- function(vals) {
  lv <- as.character(sort(unique(vals)))
  if (all(grepl("^\\s*\\d+\\s*:\\s*", lv))) {
    ord_key <- as.integer(sub("^\\s*(\\d+)\\s*:\\s*.*$", "\\1", lv))
    lv[order(ord_key)]
  } else {
    lower_num <- suppressWarnings(
      as.numeric(gsub(",", "", sub("^.*?(\\d[\\d,]*).*$", "\\1", lv)))
    )
    lv[order(lower_num, na.last = TRUE)]
  }
}