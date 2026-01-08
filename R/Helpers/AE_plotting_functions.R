ae_summary_by <- function(dt, by_col, actual = actual_col, pred = pred_col, conf = 0.95) {
  z <- qnorm(1 - (1 - conf) / 2)
  ans <- dt[, .(
    actual = sum(get(actual), na.rm = TRUE),
    pred   = sum(get(pred),   na.rm = TRUE)
  ), by = by_col]
  
  ans <- ans[ pred > 0 ]
  ans[, ae := actual / pred]
  
  ans[, se_log := fifelse(actual > 0, 1 / sqrt(actual), NA_real_) ]
  ans[, `:=`(
    lower = fifelse(is.na(se_log), 0, ae * exp(-z * se_log)),
    upper = fifelse(is.na(se_log), pmax(ae * exp(z * 10), ae * 5),  # very conservative when actual==0
                    ae * exp( z * se_log))
  )]
  
  setcolorder(ans, c(by_col, "actual", "pred", "ae", "lower", "upper"))
  ans[]
}

ae_plot <- function(summ_dt, by_col, top_n = 50) {
  plot_dt <- data.table::copy(summ_dt)
  orig_col    <- summ_dt[[by_col]]
  is_num      <- is.numeric(orig_col)
  is_ord_fact <- is.ordered(orig_col)
  
  if (is_num) {
    plot_dt[, (by_col) := as.numeric(get(by_col))]
    plot_dt <- plot_dt[order(get(by_col))]
    x_var <- plot_dt[[by_col]]
    title_suffix <- ""
  } else if (is_ord_fact) {
    levs <- levels(orig_col)
    plot_dt[, (by_col) := factor(as.character(get(by_col)),
                                 levels = levs, ordered = TRUE)]
    plot_dt[, ..ord_idx := as.integer(get(by_col))]
    plot_dt <- plot_dt[order(..ord_idx)]
    plot_dt[, ..ord_idx := NULL]
    x_var <- plot_dt[[by_col]]
    title_suffix <- ""
  } else {
    plot_dt <- plot_dt[order(-pred)]
    if (!is.null(top_n) && nrow(plot_dt) > top_n) plot_dt <- plot_dt[1:top_n]
    plot_dt[, (by_col) := factor(get(by_col), levels = plot_dt[order(ae)][[by_col]])]
    x_var <- plot_dt[[by_col]]
    title_suffix <- sprintf(" (top %s levels by exposure)", ifelse(is.null(top_n), "all", top_n))
  }
  
  eps <- .Machine$double.eps
  plot_dt[, lower_pos := pmax(lower, eps)]
  plot_dt[, upper_pos := pmax(upper, eps)]
  y_min <- max(eps, min(plot_dt$lower_pos, na.rm = TRUE) / 1.05)
  y_max <- max(plot_dt$upper_pos, na.rm = TRUE) * 1.05
  
  label_count <- if (is.numeric(x_var)) length(unique(x_var)) else nlevels(x_var)
  angle <- if (label_count > 10) 45 else 0
  hjust <- if (angle == 45) 1 else 0.5
  vjust <- if (angle == 45) 1 else 0.5
  
  ggplot2::ggplot(plot_dt, ggplot2::aes(x = x_var, y = ae)) +
    ggplot2::geom_hline(yintercept = 1, colour = "grey50", linetype = "dashed") +
    ggplot2::geom_point(size = 2, colour = "#1f77b4") +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = lower_pos, ymax = upper_pos),
                           width = 0.15, colour = "#1f77b4", alpha = 0.7) +
    ggplot2::scale_y_log10("Actual / Predicted (log scale)",
                           limits = c(y_min, y_max)) +
    ggplot2::xlab(by_col) +
    ggplot2::ggtitle(sprintf("A/E by %s%s", by_col, title_suffix)) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = angle, hjust = hjust, vjust = vjust),
      plot.title  = ggplot2::element_text(face = "bold")
    )
}
