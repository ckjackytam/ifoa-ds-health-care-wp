gen_features <- function(dataset, age_coefs = NULL, entry_age_coefs = NULL, obs_year_coefs = NULL, issue_year_coefs = NULL){
  
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
  obs_yearpoly <- if (!is.null(obs_year_coefs)) {
    poly(dataset$Observation_Year, degree = 3, coefs = obs_year_coefs) %>% data.table
  } else {
    poly(dataset$Observation_Year, degree = 3) %>% data.table
  }
  issue_yearpoly <- if (!is.null(issue_year_coefs)) {
    poly(dataset$Issue_Year, degree = 3, coefs = issue_year_coefs) %>% data.table
  } else {
    poly(dataset$Issue_Year, degree = 3) %>% data.table
  }
  agesplines <- ns(dataset$Attained_Age, knots = c(20, 30, 40, 50), Boundary.knots = c(18, 90)) %>% data.table
  
  names(agepoly) <- paste0("agepoly", seq(1,3,1))
  names(entry_agepoly) <- paste0("issue_agepoly", seq(1,3,1))
  names(obs_yearpoly) <- paste0("obs_yearpoly", seq(1,3,1))
  names(issue_yearpoly) <- paste0("issue_yearpoly", seq(1,3,1))
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
    dataset, agepoly, entry_agepoly, obs_yearpoly, issue_yearpoly, agesplines
  ) %>% data.table
  
  if("Observation_Year" %in% names(dataset)){
    dataset[, ':=' (Observation_YearChar = as.character(Observation_Year)
    )]
  }
  
  return(dataset)}
