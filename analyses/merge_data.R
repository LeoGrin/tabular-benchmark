library(tidyverse)


df <- read_csv("data/aggregates/all_datasets_categorical_classif.csv") %>% 
  bind_rows(read_csv("data/datasets_custom_categorical.csv") %>% 
              mutate(Remove = 0, Source = "Custom")) %>% 
  filter(Remove == 0, checks_passed==1) %>% 
  select(Source, dataset_name, dataset_id, n_samples, n_features, n_categorical_features) 

df %>% write_csv("test_csv.csv")




df <- read_csv("data/aggregates/all_datasets_numerical_classif.csv") %>% 
        filter(Remove == 0, checks_passed==1) %>% 
        select(Source, dataset_name, dataset_id, n_samples, n_features) %>% 
       mutate(link = str_c("https://openml.org/d/", dataset_id)) %>% 
  select(-dataset_id)

df %>% write_csv("cleaned_datasets_numerical_classif.csv")


df <- read_csv("data/aggregates/all_datasets_regression_numerical.csv") %>% 
  filter(Remove == 0, checks_passed==1) %>% 
  select(dataset_name, dataset_id, n_samples, n_features) %>% 
  mutate(link = str_c("https://openml.org/d/", dataset_id)) %>% 
  select(-dataset_id)

df %>% write_csv("cleaned_datasets_numerical_regression.csv")
