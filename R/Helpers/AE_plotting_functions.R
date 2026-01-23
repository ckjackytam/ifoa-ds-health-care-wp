ae_summary_by <- function(dt_train, dt_test, by_col, actual = actual_col, pred = pred_col, conf = 0.95) {
  z <- qnorm(1 - (1 - conf) / 2)
  
  # Process training data
  ans_train <- dt_train[, .(
    actual = sum(get(actual), na.rm = TRUE),
    pred   = sum(get(pred),   na.rm = TRUE)
  ), by = by_col]
  
  ans_train <- ans_train[pred > 0]
  ans_train[, ae := actual / pred]
  ans_train[, se_log := fifelse(actual > 0, 1 / sqrt(actual), NA_real_)]
  ans_train[, `:=`(
    lower = fifelse(is.na(se_log), 0, ae * exp(-z * se_log)),
    upper = fifelse(is.na(se_log), pmax(ae * exp(z * 10), ae * 5),
                    ae * exp(z * se_log))
  )]
  ans_train[, dataset := "Train"]
  
  # Process test data
  ans_test <- dt_test[, .(
    actual = sum(get(actual), na.rm = TRUE),
    pred   = sum(get(pred),   na.rm = TRUE)
  ), by = by_col]
  
  ans_test <- ans_test[pred > 0]
  ans_test[, ae := actual / pred]
  ans_test[, se_log := fifelse(actual > 0, 1 / sqrt(actual), NA_real_)]
  ans_test[, `:=`(
    lower = fifelse(is.na(se_log), 0, ae * exp(-z * se_log)),
    upper = fifelse(is.na(se_log), pmax(ae * exp(z * 10), ae * 5),
                    ae * exp(z * se_log))
  )]
  ans_test[, dataset := "Test"]
  
  # Combine datasets
  ans <- rbindlist(list(ans_train, ans_test))
  
  setcolorder(ans, c(by_col, "dataset", "actual", "pred", "ae", "lower", "upper"))
  
  # Create combined plot
  library(ggplot2)
  
  p <- ggplot(ans, aes_string(x = by_col, y = "ae", color = "dataset")) +
    geom_point(size = 3, position = position_dodge(width = 0.3)) +
    geom_errorbar(aes(ymin = lower, ymax = upper), 
                  width = 0.2, 
                  position = position_dodge(width = 0.3)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
    scale_color_manual(values = c("Train" = "#2E86AB", "Test" = "#A23B72")) +
    labs(
      title = "A/E Ratios by Group: Train vs Test",
      x = by_col,
      y = "A/E Ratio",
      color = "Dataset"
    ) +
    theme_minimal() +
    theme(legend.position = "top")
  
  return(ans[])
}

ae_plot <- function(summ_dt, by_col, top_n = 50, max_integer_breaks = 25) {
  is_integer_like <- function(x) {
    if (!is.numeric(x)) return(FALSE)
    all(is.na(x) | (abs(x - round(x)) < .Machine$double.eps^0.5))
  }
  
  build_integer_x_scale <- function(x_vals, max_breaks = 25) {
    ux <- sort(unique(na.omit(as.numeric(x_vals))))
    if (length(ux) <= max_breaks) {
      ggplot2::scale_x_continuous(
        breaks = ux,
        labels = scales::number_format(accuracy = 1, big.mark = ""),
        expand = ggplot2::expansion(mult = c(0.01, 0.05))
      )
    } else {
      ggplot2::scale_x_continuous(
        breaks = scales::breaks_pretty(n = 8)(ux),
        labels = scales::number_format(accuracy = 1, big.mark = ""),
        expand = ggplot2::expansion(mult = c(0.01, 0.05))
      )
    }
  }
  
  plot_dt <- data.table::copy(summ_dt)
  orig_col    <- summ_dt[[by_col]]
  is_num      <- is.numeric(orig_col)
  is_ord_fact <- is.ordered(orig_col)
  
  has_dataset <- "dataset" %in% names(plot_dt)
  
  if (is_num) {
    plot_dt[, (by_col) := as.numeric(get(by_col))]
    plot_dt <- plot_dt[order(get(by_col))]
    x_var <- plot_dt[[by_col]]
    title_suffix <- ""
    
    unique_count <- length(unique(na.omit(x_var)))
    treat_as_discrete <- is_integer_like(x_var) && unique_count <= max_integer_breaks
    x_scale <- if (treat_as_discrete) {
      plot_dt[, (by_col) := factor(get(by_col), levels = unique(plot_dt[[by_col]]))]
      x_var <- plot_dt[[by_col]]
      ggplot2::scale_x_discrete(drop = FALSE)
    } else {
      build_integer_x_scale(x_var, max_integer_breaks)
    }
    
  } else if (is_ord_fact) {
    levs <- levels(orig_col)
    plot_dt[, (by_col) := factor(as.character(get(by_col)),
                                 levels = levs, ordered = TRUE)]
    plot_dt[, `.ord_idx` := as.integer(get(by_col))]
    plot_dt <- plot_dt[order(.ord_idx)]
    plot_dt[, `.ord_idx` := NULL]
    x_var <- plot_dt[[by_col]]
    title_suffix <- ""
    x_scale <- ggplot2::scale_x_discrete(drop = FALSE)
    
  } else {
    plot_dt <- plot_dt[order(-pred)]
    if (!is.null(top_n) && nrow(plot_dt) > top_n) plot_dt <- plot_dt[1:top_n]
    
    if (has_dataset) {
      level_order <- plot_dt[, .(mean_ae = mean(ae, na.rm = TRUE)), by = by_col][order(mean_ae)][[by_col]]
    } else {
      level_order <- unique(plot_dt[order(ae)][[by_col]])
    }
    
    plot_dt[, (by_col) := factor(get(by_col), levels = level_order)]
    x_var <- plot_dt[[by_col]]
    title_suffix <- sprintf(" (top %s levels by exposure)", ifelse(is.null(top_n), "all", top_n))
    x_scale <- ggplot2::scale_x_discrete(drop = FALSE)
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
  
  if (has_dataset) {
    p <- ggplot2::ggplot(plot_dt, ggplot2::aes(x = x_var, y = ae, color = dataset, group = dataset)) +
      ggplot2::geom_hline(yintercept = 1, colour = "grey50", linetype = "dashed") +
      ggplot2::geom_point(size = 2, position = ggplot2::position_dodge(width = 0.3)) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = lower_pos, ymax = upper_pos),
                             width = 0.15, alpha = 0.7, 
                             position = ggplot2::position_dodge(width = 0.3)) +
      ggplot2::scale_color_manual(values = c("Train" = "#1f77b4", "Test" = "#ff7f0e"),
                                  name = "Dataset") +
      ggplot2::scale_y_log10("Actual / Predicted (log scale)",
                             limits = c(y_min, y_max)) +
      x_scale +
      ggplot2::xlab(by_col) +
      ggplot2::ggtitle(sprintf("A/E by %s%s", by_col, title_suffix)) +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = angle, hjust = hjust, vjust = vjust),
        plot.title  = ggplot2::element_text(face = "bold"),
        legend.position = "top"
      )
  } else {
    p <- ggplot2::ggplot(plot_dt, ggplot2::aes(x = x_var, y = ae)) +
      ggplot2::geom_hline(yintercept = 1, colour = "grey50", linetype = "dashed") +
      ggplot2::geom_point(size = 2, colour = "#1f77b4") +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = lower_pos, ymax = upper_pos),
                             width = 0.15, colour = "#1f77b4", alpha = 0.7) +
      ggplot2::scale_y_log10("Actual / Predicted (log scale)",
                             limits = c(y_min, y_max)) +
      x_scale +
      ggplot2::xlab(by_col) +
      ggplot2::ggtitle(sprintf("A/E by %s%s", by_col, title_suffix)) +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = angle, hjust = hjust, vjust = vjust),
        plot.title  = ggplot2::element_text(face = "bold")
      )
  }
  
  return(p)
}