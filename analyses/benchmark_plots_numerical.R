source("analyses/plot_utils.R")

benchmark_numerical <- read_csv("analyses/results/random_search_benchmark_numerical.csv")

######################################################
# Benchmark regression numerical medium

df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_regression_medium")

#CHECK DATA

# checks
checks(df)
#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

# Dataset by dataset

plot_results_per_dataset(df, "R2 score", truncate_scores = T)
  #geom_blank(aes(y = 1))


ggsave("analyses/plots/random_search_regression_numerical_datasets.jpg", width=15, height=10)


# Aggregated

plot_aggregated_results(df, y_inf=0.6, score="accuracy", quantile=0.5, truncate_scores = T)


ggsave("analyses/plots/random_search_regression_numerical.jpg", width=7, height=6)





######################################################
# Benchmark regression numerical large
df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_regression_large")

checks(df)

plot_results_per_dataset(df, "R2 score", truncate_scores = T, legend_size=16)

ggsave("analyses/plots/random_search_regression_numerical_large_datasets.jpg", width=8, height=6)



plot_aggregated_results(df, score = "R2 score", quantile = 0.5, truncate_scores = T, y_inf=0.6)

ggsave("analyses/plots/random_search_regression_numerical_large.jpg", width=7, height=6)



# Same datasets but medium size

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword


df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_regression_medium") %>% 
  filter(data__keyword %in% datasets)

plot_aggregated_results(df, score = "R2 score", quantile = 0.5, truncate_scores = T, y_inf=0.6)
ggsave("analyses/plots/random_search_regression_numerical_medium_comparison.jpg", width=7, height=6)


########################################################
# Benchmark classif numerical medium


df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_classif_medium")

# quantile = 0.1
# df %>% 
#   group_by(data__keyword) %>% 
#   mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
#          mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
#   write_csv("analyses/benchmark_classif_numerical_medium_normalized.csv")

checks(df)

#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, "accuracy")
#geom_blank(aes(y = 1))


ggsave("analyses/plots/random_search_classif_numerical_datasets.jpg", width=15, height=10)


# Aggregated

plot_aggregated_results(df, y_inf=0.6, score="accuracy", quantile=0.1, truncate_scores = F)


ggsave("analyses/plots/random_search_classif_numerical.jpg", width=7, height=6)




########################################################
# Benchmark classif numerical large

df <-  benchmark_numerical %>% 
  filter(benchmark == "numerical_classif_large")

checks(df)

#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, legend_size=13)


ggsave("analyses/plots/random_search_classif_numerical_large_datasets.jpg", width=8, height=6)


plot_aggregated_results(df, y_inf = 0.6)


ggsave("analyses/plots/random_search_classif_numerical_large.jpg", width=7, height=6)

# Same datasets but medium size (for comparison)

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword

df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_classif_medium") %>% 
  filter(data__keyword %in% datasets)

plot_aggregated_results(df, y_inf=0.6)

ggsave("analyses/plots/random_search_classif_numerical_medium_comparison.jpg", width=7, height=6)


