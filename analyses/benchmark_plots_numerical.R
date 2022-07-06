source("analyses/plot_utils.R")


######################################################
# Benchmark regression numerical medium

df <- read_csv("results/sweeps/sweeps_regression/benchmark_regression_random_medium_2.csv") %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_4.csv") %>%
              bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_3.csv")) %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>%
              mutate(model_name = "SAINT")) %>% 
  mutate(hp="random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_regression_default_medium_2.csv") %>% 
              bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_default.csv") %>%
                          bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_default_2.csv")) %>% 
                          distinct(data__keyword, .keep_all = T) %>% 
                          mutate(model_name = "SAINT")) %>% 
              mutate(hp="default") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time, hp)) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename() #%>% 
  #filter(data__keyword != "isolet")

#CHECK DATA

# checks
checks(df)
View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

# Dataset by dataset

plot_results_per_dataset(df, "R2 score", truncate_scores = T)
  #geom_blank(aes(y = 1))


ggsave("analyses/plots/random_search_regression_numerical_datasets.jpg", width=15, height=10)


# Aggregated

plot_aggregated_results(df, y_inf=0.4, score="accuracy", quantile=0.5, truncate_scores = T)


ggsave("analyses/plots/random_search_regression_numerical.jpg", width=7, height=6)





######################################################
# Benchmark regression numerical large
df <- read_csv("results/sweeps/sweeps_regression/large/ft_numerical_regression_large.csv") %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/gbt_numerical_regression_large.csv") %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val)) %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/rf_numerical_regression_large.csv") %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val)) %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/xgb_numerical_regression_large.csv") %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val)) %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/resnet_numerical_regression_large.csv") %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val)) %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/saint_numerical_regression_large.csv") %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val)) %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time)) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/benchmark_numerical_regression_large_default.csv") %>% 
              mutate(hp = "default")) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename()

checks(df)

plot_results_per_dataset(df, "R2 score", truncate_scores = T, legend_size=16)

ggsave("analyses/plots/random_search_regression_numerical_large_datasets.jpg", width=8, height=6)



plot_aggregated_results(df, score = "R2 score", quantile = 0.5, truncate_scores = T, y_inf=0.6)

ggsave("analyses/plots/random_search_regression_numerical_large.jpg", width=7, height=6)



# Same datasets but medium size

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword


df <- read_csv("results/sweeps/sweeps_regression/benchmark_regression_random_medium_2.csv") %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_4.csv") %>%
              bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_3.csv")) %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>%
              mutate(model_name = "SAINT")) %>% 
  mutate(hp="random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_regression_default_medium_2.csv") %>% 
              bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_default.csv") %>%
                          bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_default_2.csv")) %>% 
                          distinct(data__keyword, .keep_all = T) %>% 
                          mutate(model_name = "SAINT")) %>% 
              mutate(hp="default") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time, hp)) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename() %>% 
  filter(data__keyword %in% datasets)

plot_aggregated_results(df, score = "R2 score", quantile = 0.5, truncate_scores = T, y_inf=0.6)
ggsave("analyses/plots/random_search_regression_numerical_medium_comparison.jpg", width=7, height=6)


########################################################
# Benchmark classif numerical medium


df <- read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif.csv") %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif.csv")) %>%  
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_classif.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_random_medium_checkpoint.csv")) %>% # %>% 
              #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_random_jannis.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_numerical_bonus.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_bonus_2.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/saint_classif.csv") %>% mutate(model_name = "SAINT")) %>% 
  mutate(hp="random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>%
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif_default.csv")) %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_default_medium_numerical.csv")) %>%
              bind_rows(read_csv("results/sweeps/sweeps_classif/saint_classif_default.csv") %>% 
                          mutate(model_name = "SAINT")) %>% 
              mutate(hp="default")) %>% 
  filter(data__keyword != "poker") %>% 
  #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

quantile = 0.1
df %>% 
  group_by(data__keyword) %>% 
  mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
         mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
  write_csv("analyses/benchmark_classif_numerical_medium_normalized.csv")

checks(df)

View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, "accuracy")
#geom_blank(aes(y = 1))


ggsave("analyses/plots/random_search_classif_numerical_datasets.jpg", width=15, height=10)


# Aggregated

plot_aggregated_results(df, y_inf=0.6, score="accuracy", quantile=0.1, truncate_scores = F)


ggsave("analyses/plots/random_search_classif_numerical.jpg", width=7, height=6)




########################################################
# Benchmark classif numerical large

df <-  read_csv("results/sweeps/sweeps_classif/benchmark_classif_random_large.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/saint_numerical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/saint_numerical_classif_large_2.csv") %>% 
              filter(data__keyword %in% c("covertype", "Higgs", "jannis", "MiniBooNE"))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/ft_classif_numeric_large_bonus.csv")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_default_large.csv") %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/saint_classif_numerical_large_default.csv")) %>% 
              mutate(hp="default")) %>% 
  filter(data__keyword != "poker") %>% 
  #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

checks(df)

View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, legend_size=13)

plot_aggregated_results(df, y_inf = 0.6)


ggsave("analyses/plots/random_search_classif_numerical_large_datasets.jpg", width=8, height=6)


ggsave("analyses/plots/random_search_classif_numerical_large.jpg", width=7, height=6)

# Same datasets but medium size (for comparison)

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword

df <- read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif.csv") %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif.csv")) %>%  
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_classif.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_random_medium_checkpoint.csv") %>% 
              select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_random_jannis.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_numerical_bonus.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_bonus_2.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/saint_classif.csv") %>% mutate(model_name = "SAINT")) %>% 
  mutate(hp="random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>%
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif_default.csv")) %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_default_medium_numerical.csv")) %>%
              bind_rows(read_csv("results/sweeps/sweeps_classif/saint_classif_default.csv") %>% 
                          mutate(model_name = "SAINT")) %>% 
              mutate(hp="default")) %>% 
  filter(data__keyword != "poker") %>% 
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename() %>% 
  filter(data__keyword %in% datasets)

plot_aggregated_results(df, y_inf=0.6)

ggsave("analyses/plots/random_search_classif_numerical_medium_comparison.jpg", width=7, height=6)


