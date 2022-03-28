df <- read_csv("utils/openml_datasets_properties.csv") %>% select(-...1)

df_218 <- read_csv("openML_data/open_ml_attributes_218.csv") %>% select(-...1)

df <- df %>% bind_rows(df_218) %>% distinct()


df$id

(df %>% 
  filter(NumberOfInstances > 5000) %>% 
  filter(NumberOfClasses == 2) %>% 
  #filter(MinorityClassPercentage > 10) %>% 
  filter(PercentageOfInstancesWithMissingValues < 50) %>% 
  filter(PercentageOfNumericFeatures > 50))$id

ok_id <- c(219, 3904)

signal_ids <- c(9952, 168335)
#maybe 9952 is okay
#I think 168335 is okay but not 100% sure


simulated_id <- c(146606)

removed_for_now <- c(9977, 167120, 7592, 14965, 168337, 168338, 168908, 168912)
#think about it for later
#167120 are encrypted equity features
#not sure how they are mixed etc..

