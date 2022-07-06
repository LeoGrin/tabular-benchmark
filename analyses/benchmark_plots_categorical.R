source("analyses/plot_utils.R")

###################################
# Benchmark classif categorical medium


df <-  read_csv("results/sweeps/sweeps_classif/benchmark/ft_transformer_categorical_classif.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/resnet_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/rf_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/gbt_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/hgbt_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/xgb_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/saint_categorical_classif.csv")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_categorical_classif_medium_default.csv") %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/xgb_categorical_classif_default.csv")) %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/saint_categorical_classif_default.csv")) %>% 
              mutate(hp = "default")) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

# checks
checks(df)

# Dataset by dataset

plot_results_per_dataset(df, "accuracy")


ggsave("analyses/plots/benchmark_categorical_classif_datasets.jpg", width=15, height=10)


# Aggregated

plot_aggregated_results(df, y_inf=0.4, score="accuracy")


ggsave("analyses/plots/benchmark_categorical_classif.jpg", width=7, height=6)


###################################
# Benchmark classif categorical large


df <- read_csv("results/sweeps/sweeps_classif/benchmark/large/gbt_categorical_classif_large.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/hgbt_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/rf_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/xgb_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/saint_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/resnet_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/ft_categorical_classif_large.csv")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/benchmark_categorical_classif_large_default.csv") %>% 
              mutate(hp = "default")) %>% 
  #select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename() %>% 
  filter(data__keyword != "electricity")

# checks
checks(df)
View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

df <- df %>% select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time, hp) 

# Dataset by dataset

plot_results_per_dataset(df, "accuracy", legend_size=13)


ggsave("analyses/plots/benchmark_categorical_classif_large_datasets.jpg", width=8, height=6)


# Aggregated

plot_aggregated_results(df, y_inf=0.5, score="accuracy", quantile=0.2)


ggsave("analyses/plots/benchmark_categorical_classif_large.jpg", width=7, height=6)

######################
# Same datasets but medium size (for comparison)

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword


df <-  read_csv("results/sweeps/sweeps_classif/benchmark/ft_transformer_categorical_classif.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/resnet_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/rf_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/gbt_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/hgbt_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/xgb_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/saint_categorical_classif.csv")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_categorical_classif_medium_default.csv") %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/xgb_categorical_classif_default.csv")) %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/saint_categorical_classif_default.csv")) %>% 
              mutate(hp = "default")) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename() %>% 
  filter(data__keyword %in% datasets)

plot_results_per_dataset(df, "accuracy")


plot_aggregated_results(df, y_inf=0.5, score="accuracy")

ggsave("analyses/plots/benchmark_categorical_classif_medium_comparison.jpg", width=7, height=6)




############################################
############################################
# Benchmark regression categorical medium

df <-read_csv("results/sweeps/sweeps_regression/benchmark_categorical_regression_medium.csv") %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/xgb_regression_categorical.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/rf_regression_categorical_bonus.csv")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_categorical_regression_medium_default.csv") %>% 
              bind_rows(read_csv("results/sweeps/sweeps_regression/xgb_regression_categorical_default.csv")) %>% 
              mutate(hp = "default")) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename() %>% 
  filter(!(data__keyword %in% c("Allstate_Claims_Severity", "LoanDefaultPrediction")))

checks(df)

View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, "R2 score", truncate_scores = T)




ggsave("analyses/plots/benchmark_categorical_regression_datasets.jpg", width=15, height=10)


# Aggregated

plot_aggregated_results(df, y_inf=0.3, score="accuracy")


ggsave("analyses/plots/benchmark_categorical_classif.jpg", width=7, height=6)


############################################
# Benchmark regression categorical large

df <-read_csv("results/sweeps/sweeps_regression/resnet_categorical_regression_large.csv") %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/ft_categorical_regression_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/gbt_categorical_regression_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/gbt_regression_categorical_large_bonus.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/hgbt_regression_categorical_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/rf_regression_categorical_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/rf_categorical_regression_large_bonus.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/xgb_regression_categorical_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/saint_regression_categorical_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/benchmark_categorical_regression_large_default.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val)) %>% 
              mutate(hp = "default")) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename() %>% 
  filter(data__keyword != "Allstate_Claims_Severity",
         data__keyword != "LoanDefaultPrediction")

checks(df)

View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, score = "R2 score", truncate_scores = T, legend_size=13)#, normalize = T, quantile=0.5)

ggsave("analyses/plots/benchmark_regression_categorical_large_datasets.jpg", width=8, height=6)



plot_aggregated_results(df, score = "R2 score", quantile=0.5, truncate_scores = T)

ggsave("analyses/plots/benchmark_regression_categorical_large.jpg", width=7, height=6)


# Same datasets but medium sized

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword

df <-read_csv("results/sweeps/sweeps_regression/benchmark_categorical_regression_medium.csv") %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/xgb_regression_categorical.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/rf_regression_categorical_bonus.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_categorical_medium_comparison.csv")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_categorical_regression_medium_default.csv") %>% 
              bind_rows(read_csv("results/sweeps/sweeps_regression/xgb_regression_categorical_default.csv")) %>% 
              bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_categorical_medium_comparison_default.csv")) %>% 
              mutate(hp = "default")) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename() %>% 
  filter(data__keyword %in% datasets)

checks(df)

plot_results_per_dataset(df, score = "R2 score")

plot_aggregated_results(df, score = "R2 score", quantile=0.5, truncate_scores = T)

ggsave("analyses/plots/benchmark_regression_categorical_medium_comparison.jpg", width=7, height=6)


