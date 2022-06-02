library(tidyverse)
library(zoo)

rename <- function(df) {
  return(df %>% 
           mutate(model_name = case_when(
             model_name == "rf_c" ~ "RandomForest",
             model_name == "rf_r" ~ "RandomForest",
             model_name == "xgb_c" ~ "XGBoost",
             model_name == "xgb_r" ~ "XGBoost",
             model_name == "gbt_c" ~ "GradientBoostingTree",
             model_name == "gbt_r" ~ "GradientBoostingTree",
             model_name == "ft_transformer" ~ "FT Transformer",
             model_name == "ft_transformer_regressor" ~ "FT Transformer",
             model_name == "rtdl_resnet" ~ "Resnet",
             model_name == "rtdl_resnet_regressor" ~ "Resnet",
             model_name == "rtdl_mlp" ~ "MLP",
             model_name == "rtdl_mlp_regressor" ~ "MLP")))
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
  
  if ("hp" %in% colnames(df_normalized))
    return(
      df_normalized %>% 
        select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, {{ variable }}, hp) %>% 
        ungroup()
      )
  else
    return(
      df_normalized %>% 
        select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, {{ variable }}) %>% 
        ungroup()
    )
}

random_search <- function(df, variable, n_shuffles, default_first=F){
  res <- tibble()
  
  for (i in 1:n_shuffles) {
    if (default_first) {
    new_df <- df %>% 
      add_column(random_number = runif(nrow(.))) %>% 
      group_by(model_name, data__keyword, {{variable}}) %>% 
      mutate(random_rank = if_else(hp=="default", 0, rank(random_number)))%>% #default hp as first iter 
      select(-random_number) %>% 
      arrange(random_rank)
    }
    else {
      new_df <- df %>% 
        add_column(random_number = runif(nrow(.))) %>% 
        group_by(model_name, data__keyword, {{variable}}) %>% 
        mutate(random_rank = rank(random_number))%>%
        select(-random_number) %>% 
        arrange(random_rank)
    }
    
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


normalize_no_variable <- function(df, normalization_type="quantile", quantile=0.1){
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
  
  if ("hp" %in% colnames(df_normalized))
    return(
      df_normalized %>% 
        select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
        ungroup()
    )
  else
    return(
      df_normalized %>% 
        select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time) %>% 
        ungroup()
    )
}

random_search_no_variable <- function(df, n_shuffles, default_first=F){
  res <- tibble()
  
  for (i in 1:n_shuffles) {
    if (default_first) {
      new_df <- df %>% 
        add_column(random_number = runif(nrow(.))) %>% 
        group_by(model_name, data__keyword) %>% 
        mutate(random_rank = if_else(hp=="default", 0, rank(random_number)))%>% #default hp as first iter 
        select(-random_number) %>% 
        arrange(random_rank)
    }
    else {
      new_df <- df %>% 
        add_column(random_number = runif(nrow(.))) %>% 
        group_by(model_name, data__keyword) %>% 
        mutate(random_rank = rank(random_number))%>%
        select(-random_number) %>% 
        arrange(random_rank)
    }
    
    res <- res %>%  bind_rows(new_df %>% 
                                
                                mutate(best_val_score_so_far = cummax(mean_val_score)) %>% 
                                mutate(mean_test_score = if_else(mean_val_score == best_val_score_so_far, mean_test_score, NA_real_)) %>% 
                                # Only keep the scores for the first time we beat valid score, put the other to NA
                                group_by(model_name, data__keyword, mean_val_score) %>% 
                                mutate(mean_test_score = if_else(random_rank == min(random_rank), mean_test_score, NA_real_)) %>% 
                                ungroup(mean_val_score) %>% 
                                #for every dataset and {{variable}}, prolong the curve by the last best value when there is no value left
                                ungroup() %>% 
                                complete(model_name, data__keyword, random_rank) %>%
                                group_by(model_name, data__keyword) %>% 
                                mutate(mean_test_score = na.locf(mean_test_score, na.rm=F))%>% 
                                ungroup() %>% 
                                mutate(n_dataset = i))
  }
  return(res)
}

