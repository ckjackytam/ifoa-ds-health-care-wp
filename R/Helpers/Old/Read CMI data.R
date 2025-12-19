# Read CMI data
files <- list.files(datadir)

# Do not read in the old data to save some time
files <- files[!grepl(paste0(seq(2007,2014,1), collapse = "|"), files)]

start <- Sys.time()
invisible(rm(expdata))
for(i in files){
  print(i)
  data <- fread(paste0(datadir,i), encoding = "Latin-1")
  names(data) <- gsub(" ", "", names(data)) %>% tolower
  if(exists("expdata") == TRUE){
    expdata <- rbind.fill(expdata, data) %>% data.table
    expdata$source[!complete.cases(expdata$source)] <- i
  }else{expdata <- data
  expdata$source <- i}}
Sys.time() - start
rm(data)

# Extract year and retain 2015-2019
expdata[, c("year") := gsub("CMI Term assurances DB | Datasheet v01 2019-07-19| Datasheet v01 2021-06-29|DB Datasheet |\\.csv", "", source) %>% as.numeric]
expdata <- expdata[year %in% seq(2015,2019,1)]
# Following transition to cloud, the pound sign got corrupted (perhaps because non-UK system?)
#expdata[, c("sumassuredband") := gsub("\xa3", "", sumassuredband)]

# Drop irrelevant cols
#expdata <- expdata[, -grepl("comparator|productcategory|source", names(expdata), ignore.case = T),
#                   with = F]

# Check unknown values in columns
# apply(expdata, 2, function(x) sum(is.na(x)))

# Set duration 10+ to 10
expdata[, c("duration") := fifelse(duration == "10+", "10", duration) %>% as.numeric]

# Set commencement 1999 or earlier to 1999
expdata[, c("commencementyear") := gsub(" or earlier", "", commencementyear) %>% as.numeric]

# Set exposure to 1 day for rows without exposure, but with claim
expdata[, c("livesexposure") := fifelse(livesexposure == 0 & incurredclaims != 0, 1/365.25, livesexposure)]
# Take out rows without exposure and no claims
expdata <- expdata[livesexposure != 0]
