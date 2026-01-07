

library(DT)

glm_rr_table <- function(fit, conf_level = 0.95, digits = 3) {
  stopifnot(inherits(fit, "glm"))
  cf   <- coef(fit)
  vc   <- vcov(fit)
  se   <- sqrt(diag(vc))
  z    <- cf / se
  pval <- 2 * pnorm(-abs(z))
  
  alpha <- 1 - conf_level
  zcrit <- qnorm(1 - alpha/2)
  lo    <- cf - zcrit * se
  hi    <- cf + zcrit * se
  
  rr    <- exp(cf)
  rr_lo <- exp(lo)
  rr_hi <- exp(hi)
  
  mf <- model.frame(fit)
  term_labels <- names(cf)
  baselines <- character(length(cf))
  trms <- attr(terms(fit), "term.labels")
  for (nm in trms) {
    if (is.factor(mf[[nm]])) {
      bl <- levels(mf[[nm]])[1] 
      idx <- grepl(paste0("^", nm), term_labels)
      baselines[idx] <- paste0(nm, ": ", bl)
    }
  }
  baselines[baselines == ""] <- NA_character_
  
  out <- data.frame(
    Term         = names(cf),
    `Rate ratio` = rr,
    `Lower CI`   = rr_lo,
    `Upper CI`   = rr_hi,
    `Estimate`   = cf,
    `Std. Error` = se,
    `z value`    = z,
    `absolute z value`    = abs(z),
    #`Pr(>|z|)`   = pval,
    `Baseline (if factor)` = baselines,
    check.names  = FALSE
  )
  
  num_cols <- c("Rate ratio","Lower CI","Upper CI","Estimate","Std. Error","z value","absolute z value")
  out[num_cols] <- lapply(out[num_cols], function(x) signif(x, digits))
  out
}

glm_dt_options <- function() {
  list(
    dom = "Bfrtip",
    buttons = c("copy", "csv", "excel"),
    pageLength = 25,
    scrollX = TRUE,
    deferRender = TRUE,
    order = list(list(7, "desc")),
    columnDefs = list(
      list(targets = 0, className = "dt-left"),
      list(targets = 1:3, className = "dt-right"),
      list(targets = 4:7, className = "dt-right")
    )
  )
}

glm_dt_format <- function(dt) {
  dt %>%
#    formatStyle(
#      "Pr(>|z|)",
#      backgroundColor = styleInterval(c(0.001, 0.01, 0.05),
#                                      c("#e8f5e9", "#c8e6c9", "#fff9c4", "#ffebee")
#      )
#    ) %>%
    formatStyle(
      "Rate ratio",
      color = styleInterval(1, c("#1b5e20", "#b71c1c")) # <1 green, >1 red
    ) %>%
    formatRound(c("Rate ratio","Lower CI","Upper CI","Estimate","Std. Error","z value","absolute z value"), digits = 3) %>%
    htmlwidgets::onRender(
      "
      function(el, x) {
        var tbl = document.getElementById(el.id).getElementsByTagName('tbody')[0];
        for (var i = 0; i < tbl.rows.length; i++) {
          var baselineCell = tbl.rows[i].cells[8]; // column index of 'Baseline (if factor)'
          var termCell = tbl.rows[i].cells[0];     // 'Term'
          if (baselineCell && baselineCell.innerText.trim().length > 0) {
            termCell.title = baselineCell.innerText;
          }
        }
      }
      "
    )
}
