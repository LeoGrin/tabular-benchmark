library(tidyverse)


#Real data

names <- c("electricity", "california", "covtype", "churn", "credit", "shopping", "nomao", "spam")
n_samples <- c(10000, 10000, 10000, 4074, 10000, 3816,8194,3624)
n_features <- c(7, 8, 10, 6, 10, 10, 46, 57)

df_datasets <- tibble(names, n_samples, n_features)

colnames(df_datasets)

df <- read_csv("results/new/res_new_real_bonus_2_reverse.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names"))

df %>% 
  mutate(algo_hp = paste(method_name, max_depth, hidden_size, sep="")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__0__method_name == "no_transform", transform__1__method_name == "gaussienize", transform__1__type == "quantile") %>%
  filter(!method_name %in% c("mars")) %>% 
  filter(is.na(max_depth)) %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

df %>% 
  mutate(algo_hp = paste(method_name, max_depth, hidden_size, sep="")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__0__method_name == "random_rotation", transform__1__method_name == "gaussienize", transform__1__type == "quantile") %>%
  filter(!method_name %in% c("rotf", "mars")) %>% 
  filter(is.na(max_depth)) %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

df %>% 
  mutate(algo_hp = paste(method_name, max_depth, hidden_size, sep="")) %>% 
  filter(transform__1__method_name == "gaussienize", transform__1__type == "quantile") %>%
  filter(!method_name %in% c("mars")) %>% 
  filter(is.na(max_depth)) %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = test_scores, color = transform__0__method_name), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~algo_hp)


df <- read_csv("results/new/res_new_real_bonus_2.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names"))

df_2 <- read_csv("results/new/res_new_real_bonus_2_09_11.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names"))

df_3 <- read_csv("results/new/lounici.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names"))

total_df <- df %>% bind_rows(df_2) %>% bind_rows(df_3)

#TODO : relative score at the iter level
#relative score  = score / best_score for this dataset

total_df %>% 
  mutate(algo_hp = paste(method_name, max_depth, hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(!method_name %in% c("mars")) %>% 
  filter(is.na(max_depth)) %>% 
  group_by(algo_hp, method_name, data__keyword, score_type) %>% 
  summarise(mean_score_iter = mean(score)) %>% 
  ungroup() %>% 
  group_by(method_name, data__keyword, score_type) %>% 
  mutate(max_score_hp = max(mean_score_iter)) %>% 
  filter(max_score_hp == mean_score_iter) %>% 
  ungroup() %>% 
  group_by(data__keyword, score_type) %>% 
  mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = mean_score_iter, color = method_name), alpha=1, height=0.1, width=0) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)


total_df %>% 
  mutate(algo_hp = paste(method_name, max_depth, hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "random_rotation", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(!method_name %in% c("mars")) %>% 
  filter(is.na(max_depth)) %>% 
  group_by(algo_hp, method_name, data__keyword, score_type) %>% 
  summarise(mean_score_iter = mean(score)) %>% 
  ungroup() %>% 
  group_by(method_name, data__keyword, score_type) %>% 
  mutate(max_score_hp = max(mean_score_iter)) %>% 
  filter(max_score_hp == mean_score_iter) %>% 
  ungroup() %>% 
  group_by(data__keyword, score_type) %>% 
  mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = relative_score, color = method_name), alpha=0.6, height=0.1, width=0) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

total_df %>% 
  mutate(algo_hp = paste(method_name, max_depth, hidden_size, sep="")) %>% 
  filter(transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(!method_name %in% c("mars")) %>% 
  filter(is.na(max_depth)) %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = test_scores, color = transform__1__method_name), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~algo_hp)

#new

total_df <- read_csv("results/new/res_new_real_bonus_2_10_11.csv") %>% 
  bind_rows(read_csv("results/new/lounici_05_12.csv")) %>% 
  bind_rows(read_csv("results/new/nam.csv"))
  #bind_rows(read_csv("results/new/lounici.csv")) %>% 
  #bind_rows(read_csv("results/new/lounici2.csv") %>% mutate(method_name = "lounici2"))
  
total_df %>% 
  filter(method_name %in% c("mlp", "lounici", "nam", "auto_sklearn", "rf", "hgbt")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = method_name), alpha=1, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)


total_df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "random_rotation", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = mean_score_iter, color = method_name), alpha=1, height=0.1, width=0) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

total_df %>% 
  filter(transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = test_scores, color = transform__1__method_name), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~method_name)


# Tree taxonomy

df_trees_synth <- read_csv("results/new/res_tree_taxo_synth.csv")
df_trees_real <- read_csv("results/new/res_tree_taxo_real.csv")%>% 
  left_join(df_datasets, by = c("data__keyword" = "names"))

df_trees_real %>% 
  filter(transform__0__method_name == "no_transform") %>% 
  mutate(max_depth = as_factor(max_depth)) %>% 
  group_by(data__keyword, n_estimators, method_name, max_depth) %>% 
  mutate(mean_test_scores = mean(test_scores)) %>% 
  ungroup() %>% 
  ggplot() +
  #geom_jitter(aes(y = data__keyword, x = test_scores, color = max_depth),  alpha=0.5, width=0, height=0.1)+
  geom_point(aes(y = data__keyword, x = mean_test_scores, fill = max_depth),  size=3, alpha=0.6, color="black", shape=21)+
  facet_wrap(n_estimators~method_name)

df_trees_real %>% 
  filter(transform__0__method_name == "no_transform") %>% 
  filter(method_name == "rf") %>% 
  mutate(max_depth = as_factor(max_depth), n_estimators = as_factor(n_estimators)) %>% 
  group_by(data__keyword, method_name, n_estimators, max_depth) %>% 
  mutate(mean_test_scores = mean(test_scores)) %>% 
  ungroup() %>% 
  ggplot() +
  #geom_jitter(aes(y = data__keyword, x = test_scores, color = max_depth),  alpha=0.5, width=0, height=0.1)+
  geom_point(aes(y = data__keyword, x = mean_test_scores, fill = n_estimators),  size=3, alpha=0.6, color="black", shape=21)+
  facet_wrap(max_depth~method_name)

df_trees_real %>% 
  filter(transform__0__method_name == "no_transform") %>% 
  mutate(max_depth = as_factor(max_depth)) %>% 
  group_by(data__keyword, n_estimators, method_name, max_depth) %>% 
  mutate(mean_test_scores = mean(test_scores)) %>% 
  ungroup() %>% 
  ggplot() +
  #geom_jitter(aes(y = data__keyword, x = test_scores, color = max_depth),  alpha=0.5, width=0, height=0.1)+
  geom_point(aes(x = as_factor(n_features), y = mean_test_scores, fill = max_depth),  size=3, alpha=0.6, color="black", shape=21)+
  facet_wrap(n_estimators~method_name)


df_trees_real %>% 
  filter(transform__0__method_name == "no_transform") %>% 
  mutate(max_depth = as_factor(max_depth)) %>% 
  group_by(data__keyword, n_estimators, method_name, max_depth) %>% 
  mutate(mean_test_scores = mean(test_scores)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = test_scores, color = max_depth),  alpha=0.5, width=0, height=0.1)+
  #geom_point(aes(y = , x = mean_test_scores, fill = max_depth),  size=3, alpha=0.6, color="black", shape=21)+
  facet_wrap(n_estimators~method_name)


df_trees_real_features <- read_csv("results/new/res_tree_taxonomy_real_features.csv")%>% 
  left_join(df_datasets, by = c("data__keyword" = "names"))

df_trees_real_features %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = test_scores, color = max_features),  alpha=0.5, width=0, height=0.1)+
  facet_grid(max_depth~n_estimators)

df_trees_real_features %>% 
  filter(is.na(max_depth), n_estimators==100) %>% 
  group_by(data__keyword, max_features) %>% 
  mutate(mean_test_score = mean(test_scores)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(y = data__keyword, x = test_scores, color = max_features),  alpha=0.1, width=0, height=0.1)+
  geom_point(aes(y = data__keyword, x = mean_test_score, color = max_features), size=2)
  
df_trees_synth %>% 
  mutate(target__max_depth = as_factor(target__max_depth)) %>% 
  filter(data__num_samples == 10000, data__num_features==15) %>% 
  filter(target__n_trees == 5) %>% 
  filter(transform__0__method_name == "no_transform") %>% 
  group_by(target__max_depth, n_estimators, method_name, max_depth) %>% 
  mutate(mean_test_scores = mean(test_scores)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(y = target__max_depth, x = test_scores, color = max_depth),  alpha=0.5, width=0, height=0.1)+
  geom_point(aes(y = target__max_depth, x = mean_test_scores, fill = max_depth),  size=10, alpha=0.6, color="black", shape=21)+
  
  facet_wrap(n_estimators~method_name)

# Gaussian data

df_gaussian <- read_csv("results/new/res_new_gaussian.csv")

df_gaussian %>% 
  ggplot() +
  geom_jitter(aes(y = data__cov_matrix, x = test_scores, color = method_name),  alpha=0.5, width=0, height=0.1)

