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


ggsave("analyses/plots/random_search_regression_numerical_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.5, score="R2 score", quantile=0.5, truncate_scores = T, text_size=9, theme_size=27)


ggsave("analyses/plots/random_search_regression_numerical.pdf", width=14, height=7.3, bg="white")




######################################################
# Benchmark regression numerical large
df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_regression_large")

checks(df)

plot_results_per_dataset(df, "R2 score", truncate_scores = T, legend_size=16)

ggsave("analyses/plots/random_search_regression_numerical_large_datasets.pdf", width=8, height=6, bg="white")



plot_aggregated_results(df, score = "R2 score", quantile = 0.4, truncate_scores = T, y_inf=0.6)

ggsave("analyses/plots/random_search_regression_numerical_large.pdf", width=7, height=6, bg="white")



# Same datasets but medium size

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword
model_names <- (df %>% select(model_name) %>% distinct())$model_name


df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_regression_medium") %>% 
  filter(model_name %in% model_names) %>%  #not available for large scale
  filter(data__keyword %in% datasets)

plot_aggregated_results(df, score = "R2 score", quantile = 0.4, truncate_scores = T, y_inf=0.6)
ggsave("analyses/plots/random_search_regression_numerical_medium_comparison.pdf", width=7, height=6, bg = "white")


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


ggsave("analyses/plots/random_search_classif_numerical_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.5, score="accuracy", quantile=0.1, truncate_scores = F, text_size=9, theme_size=27)


ggsave("analyses/plots/random_search_classif_numerical.pdf", width=14, height=7.3, bg="white")




########################################################
# Benchmark classif numerical large

df <-  benchmark_numerical %>% 
  filter(benchmark == "numerical_classif_large")

checks(df)

#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, legend_size=13)


ggsave("analyses/plots/random_search_classif_numerical_large_datasets.pdf", width=8, height=6, bg="white")


plot_aggregated_results(df, y_inf = 0.6, score="accuracy")


ggsave("analyses/plots/random_search_classif_numerical_large.pdf", width=7, height=6, bg="white")

# Same datasets but medium size (for comparison)

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword

df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_classif_medium") %>% 
  filter(data__keyword %in% datasets)

plot_aggregated_results(df, y_inf=0.6, score="accuracy")

ggsave("analyses/plots/random_search_classif_numerical_medium_comparison.pdf", width=7, height=6, bg="white")


