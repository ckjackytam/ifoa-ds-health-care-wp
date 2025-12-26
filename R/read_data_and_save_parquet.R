
start <- Sys.time()
ilec <- read_delim("Data/ILEC_2012_19 - 20240429.txt", 
                   delim = "\t", escape_double = FALSE, 
                   trim_ws = TRUE) %>% data.table
Sys.time() - start # 3 minutes

#drop_columns <- grep("^(ExpDth_|Cen)", names(ilec), value = TRUE)
#ilec[, (drop_columns) := NULL]

# retain TERM business
ilec <- ilec[Insurance_Plan == "Term"]

int_cols <- c("Attained_Age", "Issue_Age", "Duration", "Issue_Year", 
              "Preferred_Class", "Number_of_Pfd_Classes", "Observation_Year")
ilec[, (int_cols) := lapply(.SD, as.integer), .SDcols = int_cols]

arrow::write_parquet(ilec, "Data/ILEC_2012_19 - 20240429.parquet")

