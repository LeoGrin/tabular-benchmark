source("analyses/plot_utils.R")

benchmark_categorical <- read_csv("analyses/results/random_search_benchmark_categorical.csv")

###################################
# Benchmark classif categorical medium

df <- benchmark_categorical %>% 
  filter(benchmark == "categorical_classif_medium")



# checks
checks(df)

# Dataset by dataset

plot_results_per_dataset(df, "accuracy")


ggsave("analyses/plots/benchmark_categorical_classif_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.4, score="accuracy", text_size=9, theme_size=27)


ggsave("analyses/plots/benchmark_categorical_classif.pdf", width=14, height=7.3, bg="white")


###################################
# Benchmark classif categorical large

df <- benchmark_categorical %>% 
  filter(benchmark == "categorical_classif_large")

# checks
checks(df)

df <- df %>% select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time, hp) 

# Dataset by dataset

plot_results_per_dataset(df, "accuracy", legend_size=13)


ggsave("analyses/plots/benchmark_categorical_classif_large_datasets.pdf", width=8, height=6, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.5, score="accuracy", quantile=0.2)


ggsave("analyses/plots/benchmark_categorical_classif_large.pdf", width=7, height=6, bg="white")

######################
# Same datasets but medium size (for comparison)

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword
model_names <- (df %>% select(model_name) %>% distinct())$model_name


df <- benchmark_categorical %>% 
  filter(benchmark == "categorical_classif_medium") %>% 
  filter(data__keyword %in% datasets) %>% 
  filter(model_name %in% model_names)

plot_results_per_dataset(df, "accuracy")


plot_aggregated_results(df, y_inf=0.5, score="accuracy")

ggsave("analyses/plots/benchmark_categorical_classif_medium_comparison.pdf", width=7, height=6, bg="white")




############################################
############################################
# Benchmark regression categorical medium

df <- benchmark_categorical %>% 
  filter(benchmark == "categorical_regression_medium")


checks(df)



plot_results_per_dataset(df, "R2 score", truncate_scores = T)


ggsave("analyses/plots/benchmark_categorical_regression_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, score = "R2 score", quantile=0.5, truncate_scores = T, y_inf=0.4,  text_size=9, theme_size=27)


ggsave("analyses/plots/benchmark_categorical_regression.pdf", width=14, height=7.3, bg="white")


############################################
# Benchmark regression categorical large

df <- benchmark_categorical %>% 
  filter(benchmark == "categorical_regression_large")

checks(df)

#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, score = "R2 score", truncate_scores = T, legend_size=13)#, normalize = T, quantile=0.5)

ggsave("analyses/plots/benchmark_regression_categorical_large_datasets.pdf", width=8, height=6, bg="white")



plot_aggregated_results(df, score = "R2 score", quantile=0.5, truncate_scores = T, y_inf=0.4)

ggsave("analyses/plots/benchmark_regression_categorical_large.pdf", width=7, height=6, bg="white")


# Same datasets but medium sized

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword
model_names <- (df %>% select(model_name) %>% distinct())$model_name

df <- benchmark_categorical %>% 
  filter(benchmark == "categorical_regression_medium") %>% 
  filter(data__keyword %in% datasets) %>% 
  filter(model_name %in% model_names)

checks(df)

plot_results_per_dataset(df, score = "R2 score")

plot_aggregated_results(df, score = "R2 score", quantile=0.5, truncate_scores = T, y_inf=0.4)

ggsave("analyses/plots/benchmark_regression_categorical_medium_comparison.pdf", width=7, height=6, bg="white")


