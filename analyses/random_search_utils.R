library(tidyverse)
library(zoo)

rename <- function(df) {
  return(df %>% 
           mutate(model_name = case_when(
             model_name == "rf_c" ~ "RandomForest",
             model_name == "xgb_c" ~ "XGBoost",
             model_name == "gbt_c" ~ "GradientBoostingTree",
             model_name == "ft_transformer" ~ "FT Transformer",
             model_name == "rtdl_resnet" ~ "Resnet")))
}


normalize <- function(df, variable, group_on_variable=F, normalization_type="quantile", quantile=0.1){
  if (group_on_variable)
    df_normalized <- df %>% group_by(data__keyword, {{ variable }})
  else
    df_normalized <- df %>% group_by(data__keyword)
  
  
  if (normalization_type == "quantile") {
  df_normalized <- df_normalized %>% 
    mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
           mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T)))
  }
  
  if (normalization_type == "max") {
    df_normalized <- df_normalized %>% 
      mutate(mean_val_score = mean_val_score / max(mean_val_score, na.rm=T), 
             mean_test_score = mean_test_score / max(mean_test_score, na.rm=T))
  }
  
  
  
  return(
    df_normalized %>% 
      select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, {{ variable }}) %>% 
      ungroup()
    )
}

random_search <- function(df, variable, n_shuffles){
  res <- tibble()
  
  for (i in 1:n_shuffles) {
    new_df <- df %>% 
      add_column(random_number = runif(nrow(.))) %>% 
      group_by(model_name, data__keyword, {{variable}}) %>% 
      mutate(random_rank = rank(random_number))%>%
      select(-random_number) %>% 
      arrange(random_rank)
    
    res <- res %>%  bind_rows(new_df %>% 
                                                  
                                                  mutate(best_val_score_so_far = cummax(mean_val_score)) %>% 
                                                  mutate(mean_test_score = if_else(mean_val_score == best_val_score_so_far, mean_test_score, NA_real_)) %>% 
                                                  # Only keep the scores for the first time we beat valid score, put the other to NA
                                                  group_by(model_name, data__keyword, {{variable}}, mean_val_score) %>% 
                                                  mutate(mean_test_score = if_else(random_rank == min(random_rank), mean_test_score, NA_real_)) %>% 
                                                  ungroup(mean_val_score) %>% 
                                                  #for every dataset and {{variable}}, prolong the curve by the last best value when there is no value left
                                                  ungroup() %>% 
                                                  complete(model_name, data__keyword, {{variable}}, random_rank) %>%
                                                  group_by(model_name, {{variable}}, data__keyword) %>% 
                                                  mutate(mean_test_score = na.locf(mean_test_score, na.rm=F))%>% 
                                                  ungroup() %>% 
                                                  mutate(n_dataset = i))
  }
  return(res)
}

# df <- read_csv("results/sweeps/sweeps_classif/high_frequency/gbt_high_frequency.csv") %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/high_frequency/gbt_high_frequency_2.csv"))
# 
# a <- df %>% 
#   filter(!is.na(mean_time)) %>% 
#   rename() %>% 
#   normalize(variable = transform__2__cov_mult) %>% 
#   random_search(variable = transform__2__cov_mult, n_shuffles=30)
