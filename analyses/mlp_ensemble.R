library(tidyverse)


#Real data

names <- c("electricity", "california", "covtype", "churn", "credit", "shopping", "nomao", "spam")
n_samples <- c(10000, 10000, 10000, 4074, 10000, 3816,8194,3624)
n_features <- c(7, 8, 10, 6, 10, 10, 46, 57)

df_datasets <- tibble(names, n_samples, n_features)

colnames(df_datasets)


df <- read_csv("results/new/res_new_real_bonus_2.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names"))

#ensemble
df %>% 
  filter(method_name == "mlp_ensemble") %>% 
  mutate(algo_hp = paste(hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score, color = algo_hp)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

#ensemble
df %>% 
  filter(method_name == "mlp_ensemble") %>% 
  mutate(algo_hp = paste(hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score)) %>% 
  ungroup() %>% 
  filter(score_type == "test_scores") %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score, color = mlp_size)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(hidden_size ~ n_mlps) +
  ggtitle("Test scores")

#ensemble
df %>% 
  filter(method_name == "mlp_ensemble") %>% 
  mutate(algo_hp = paste(hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score)) %>% 
  ungroup() %>% 
  filter(score_type == "train_scores") %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score, color = n_mlps)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(hidden_size ~ mlp_size) +
  ggtitle("Test scores")

#ensemble
df %>% 
  filter(method_name == "mlp_ensemble") %>% 
  mutate(algo_hp = paste(hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score)) %>% 
  ungroup() %>% 
  filter(score_type == "train_scores") %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score, color = hidden_size)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(n_mlps ~ mlp_size) +
  ggtitle("Test scores")


#Real data

names <- c("electricity", "california", "covtype", "churn", "credit", "shopping", "nomao", "spam")
n_samples <- c(10000, 10000, 10000, 4074, 10000, 3816,8194,3624)
n_features <- c(7, 8, 10, 6, 10, 10, 46, 57)

df_datasets <- tibble(names, n_samples, n_features)

colnames(df_datasets)


df <- read_csv("results/new/res_mlp_ensemble.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names"))

#ensemble
df %>% 
  mutate(algo_hp = paste(train_on_different_batch, hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(score_type == "test_scores") %>% 
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score, color = algo_hp)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

#ensemble
df %>% 
  mutate(algo_hp = paste(train_on_different_batch, hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  filter(score_type == "test_scores") %>% 
  mutate(mlp_size = as_factor(mlp_size)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score, color = mlp_size)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(hidden_size ~ n_mlps) +
  ggtitle("Test scores")

#ensemble
df %>% 
  mutate(algo_hp = paste(train_on_different_batch, hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(score_type == "test_scores") %>% 
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score, color = n_mlps)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~mlp_size) +
  ggtitle("Test scores")

#ensemble
df %>% 
  mutate(algo_hp = paste(train_on_different_batch, hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  filter(score_type == "test_scores") %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score, color = train_on_different_batch)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~mlp_size) +
  ggtitle("Test scores")

#ensemble
df %>% 
  mutate(algo_hp = paste(train_on_different_batch, hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  group_by(data__keyword, score_type) %>% 
  summarise(best_score = max(mean_score, na.rm=T)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = best_score)) +
  # geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  ggtitle("Test scores")

total_df <- read_csv("results/new/res_new_real_bonus_2_10_11.csv")

total_df <- total_df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup()

df %>% 
  mutate(algo_hp = paste(train_on_different_batch, hidden_size, n_mlps, mlp_size, sep="/")) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(algo_hp, data__keyword, score_type) %>% 
  mutate(mean_score = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  group_by(data__keyword, score_type) %>% 
  summarise(mean_score_iter = max(mean_score, na.rm=T)) %>% #for same name
  mutate(method_name = "mlp_ensemble") %>% 
  bind_rows(total_df) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = method_name), alpha=1, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

  

