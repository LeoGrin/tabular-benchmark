source("analyses/plot_utils.R")

benchmark_numerical <- read_csv("analyses/results/random_search_benchmark_numerical.csv") 

##################
# Numerical classif

# df <- read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif.csv") %>%
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif.csv")) %>%  
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_classif.csv")) %>%
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_random_medium_checkpoint.csv") %>% 
#               select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time)) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_random_jannis.csv")) %>%
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_numerical_bonus.csv")) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_bonus_2.csv")) %>%
#   bind_rows(read_csv("results/sweeps/sweeps_classif/saint_classif.csv") %>% mutate(model_name = "SAINT")) %>% 
#   mutate(hp="random") %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>%
#               bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif_default.csv")) %>% 
#               bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_default_medium_numerical.csv")) %>%
#               bind_rows(read_csv("results/sweeps/sweeps_classif/saint_classif_default.csv") %>% 
#                           mutate(model_name = "SAINT")) %>% 
#               mutate(hp="default")) %>% 
#   filter(data__keyword != "poker") %>% 
#   select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
#   filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
#   rename()


df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_classif_medium")

plot_aggregated_results_time(df, y_inf=0.6)

ggsave("analyses/plots/benchmark_time_numerical_classif.pdf", width=7, height=6, bg="white")



#######################
# Numerical regression
# df <- read_csv("results/sweeps/sweeps_regression/benchmark_regression_random_medium_2.csv") %>% 
#   mutate(mean_r2_test = as.numeric(mean_r2_test),
#          mean_r2_val = as.numeric(mean_r2_val)) %>% 
#   select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_4.csv") %>%
#               bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_3.csv")) %>% 
#               select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>%
#               mutate(model_name = "SAINT")) %>% 
#   mutate(hp="random") %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_regression_default_medium_2.csv") %>% 
#               bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_default.csv") %>%
#                           bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_default_2.csv")) %>% 
#                           distinct(data__keyword, .keep_all = T) %>% 
#                           mutate(model_name = "SAINT")) %>% 
#               mutate(hp="default") %>% 
#               select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time, hp)) %>% 
#   filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
#   mutate(mean_test_score = mean_r2_test,
#          mean_val_score = mean_r2_val) %>% 
#   rename() %>% 
#   filter(data__keyword != "isolet") #Too many nans with saint

#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_regression_medium")

plot_aggregated_results_time(df, score="R2 score", quantile=0.5, truncate_scores = T, y_inf=0.6)

ggsave("analyses/plots/benchmark_time_numerical_regression.pdf", width=7, height=6, bg="white")


#######################

benchmark_categorical <- read_csv("analyses/results/random_search_benchmark_categorical.csv")

# Categorical regression
# df <-read_csv("results/sweeps/sweeps_regression/benchmark_categorical_regression_medium.csv") %>% 
#   mutate(mean_r2_test = as.numeric(mean_r2_test),
#          mean_r2_val = as.numeric(mean_r2_val)) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_regression/xgb_regression_categorical.csv")) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_regression/rf_regression_categorical_bonus.csv")) %>% 
#   mutate(hp = "random") %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_categorical_regression_medium_default.csv") %>% 
#               bind_rows(read_csv("results/sweeps/sweeps_regression/xgb_regression_categorical_default.csv")) %>% 
#               mutate(hp = "default")) %>% 
#   filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
#   mutate(mean_test_score = mean_r2_test,
#          mean_val_score = mean_r2_val) %>% 
#   rename() %>% 
#   filter(!(data__keyword %in% c("Allstate_Claims_Severity", "LoanDefaultPrediction")))

df <- benchmark_categorical %>% 
  filter(benchmark == "categorical_regression_medium")

plot_aggregated_results_time(df, score="R2 score", quantile=0.4, truncate_scores = T, y_inf=0.5)


ggsave("analyses/plots/benchmark_time_categorical_regression.pdf", width=7, height=6, bg="white")


#######################
# Categorical classif


# df <-  read_csv("results/sweeps/sweeps_classif/benchmark/ft_transformer_categorical_classif.csv") %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/resnet_categorical_classif.csv")) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/rf_categorical_classif.csv")) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/gbt_categorical_classif.csv")) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/hgbt_categorical_classif.csv")) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/xgb_categorical_classif.csv")) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/saint_categorical_classif.csv")) %>% 
#   mutate(hp = "random") %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_categorical_classif_medium_default.csv") %>% 
#               bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/xgb_categorical_classif_default.csv")) %>% 
#               bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/saint_categorical_classif_default.csv")) %>% 
#               mutate(hp = "default")) %>% 
#   filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
#   rename()

#df %>% group_by(model_name, data__keyword) %>% summarise(s = sum(mean_time))

df <- benchmark_categorical %>% 
  filter(benchmark == "categorical_classif_medium")

plot_aggregated_results_time(df, y_inf=0.4)

ggsave("analyses/plots/benchmark_time_categorical_classif.pdf", width=7, height=6, bg="white")

