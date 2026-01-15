gen_features <- function(dataset, age_coefs = NULL, entry_age_coefs = NULL){
  
  agepoly <- if (!is.null(age_coefs)) {
    poly(dataset$Attained_Age, degree = 3, coefs = age_coefs) %>% data.table
  } else {
    poly(dataset$Attained_Age, degree = 3) %>% data.table
  }
  entry_agepoly <- if (!is.null(entry_age_coefs)) {
    poly(dataset$Issue_Age, degree = 3, coefs = entry_age_coefs) %>% data.table
  } else {
    poly(dataset$Issue_Age, degree = 3) %>% data.table
  }
  agesplines <- ns(dataset$Attained_Age, knots = c(20, 30, 40, 50), Boundary.knots = c(18, 90)) %>% data.table
  
  names(agepoly) <- paste0("agepoly", seq(1,3,1))
  names(entry_agepoly) <- paste0("Issue_Agepoly", seq(1,3,1))
  names(agesplines) <- paste0("agespline", seq(1,5,1))
  
  dataset[, ':=' (dur0 = fifelse(Duration == 1, 1, 0),
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
  
  dataset <- cbind(
    dataset, agepoly, entry_agepoly, agesplines
  ) %>% data.table
  
  if("Observation_Year" %in% names(dataset)){
    dataset[, ':=' (Observation_YearChar = as.character(Observation_Year)
    )]
  }
  
  if("Issue_Year" %in% names(dataset)){
    
    min_uwYear <- min(dataset$Issue_Year)
    max_uwYear <- max(dataset$Issue_Year)
    dataset[, ':=' (uw_year_01 = (Issue_Year - min_uwYear) / (max_uwYear - min_uwYear))]
    
    dataset[, ':=' (uwYearChar = as.character(Issue_Year),
                    uw_year_01_sq = uw_year_01 ^2,
                    uw_year_01_cube = uw_year_01 ^3)]
  }
  
  return(dataset)}
