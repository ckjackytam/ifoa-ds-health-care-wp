
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(readr)
ILEC_2012_19_20240429 <- read_delim("ILEC/ILEC_2012_19 - 20240429.txt", 
                                    delim = "\t", escape_double = FALSE, 
                                    trim_ws = TRUE)
View(ILEC_2012_19_20240429)

# filter only term rows - no need to run 
df_term <- ILEC_2012_19_20240429[grepl("Term", ILEC_2012_19_20240429$Insurance_Plan, ignore.case = TRUE), ]
df_term <- df_term[, 1:(ncol(df_term) - 5)]

df_term_clean <- df_term %>%
  filter(Issue_Age >=18 & Issue_Age <= 75
         & Issue_Year >= 1975
         & Duration <= 40
        )
   



# Function to create dual-axis chart
create_dual_axis_chart <- function(data, group_var, title) {
  df_ratio <- data %>%
    group_by({{ group_var }}) %>%
    summarise(
      Death_Count = sum(Death_Count, na.rm = TRUE),
      Expected = sum(ExpDth_VBT2015wMI_Cnt, na.rm = TRUE),
      Exposure = sum(Policies_Exposed, na.rm = TRUE)
    ) %>%
    mutate(AE_Ratio = Death_Count / Expected) %>%
    arrange({{ group_var }})
  
  # Plot with bars for exposure and line for A/E ratio
  ggplot(df_ratio, aes(x = {{ group_var }})) +
    geom_bar(aes(y = Exposure), stat = "identity", fill = "lightgray") +
    geom_line(aes(y = AE_Ratio * max(Exposure) / 1.3), color = "blue", size = 1) +
    geom_point(aes(y = AE_Ratio * max(Exposure) / 1.3), color = "blue", size = 3) +
    scale_y_continuous(
      name = "Exposure (Policies)",
      sec.axis = sec_axis(~ . / max(df_ratio$Exposure) * 1.3, name = "A/E Ratio", breaks = seq(0, 1.3, 0.1))
    ) +
    labs(title = title, x = as_label(enquo(group_var))) +
    theme_minimal()
}

# Create charts 
p_age    <- create_dual_axis_chart(df_term_clean, Issue_Age, "A/E Ratio and Exposure by Issue Age")
p_face   <- create_dual_axis_chart(df_term_clean, Face_Amount_Band, "A/E Ratio and Exposure by Face Amount Band")
p_year   <- create_dual_axis_chart(df_term_clean, Issue_Year, "A/E Ratio and Exposure by Issue Year")
p_smoker <- create_dual_axis_chart(df_term_clean, Smoker_Status, "A/E Ratio and Exposure by Smoker Status")
p_duration <- create_dual_axis_chart(df_term_clean, Duration, "A/E Ratio and Exposure by Duration")
p_preferred_class <-create_dual_axis_chart(df_term_clean, Preferred_Class, "A/E Ratio and Exposure by preferred class")
p_slct_ult <-create_dual_axis_chart(df_term_clean, Slct_Ult_Ind, "A/E Ratio and Exposure by Slct/Ult Indicator")
p_face_amount <-create_dual_axis_chart(df_term_clean, Face_Amount_Band, "A/E Ratio and Exposure by face amount band")
p_prederred_ind <-create_dual_axis_chart(df_term_clean, Preferred_Indicator, "A/E Ratio and Exposure by preferred indicator")
p_age_ind <-create_dual_axis_chart(df_term_clean, Age_Ind, "A/E Ratio and Exposure by age indicator")




# Display plots
p_age_ind
p_age
p_face
p_year
p_smoker
p_duration
p_preferred_class
p_slct_ult
p_face_amount
p_prederred_ind


#data cleaning - adjust age, duration, issue_age etc

#simple glm model 

#build model matrix
unique(df_term_clean$Age_Ind)
unique(df_term_clean$Preferred_Indicator)


