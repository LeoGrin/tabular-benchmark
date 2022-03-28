library(tidyverse)

df <- read_csv("results/new/small_data_mlp.csv")

df %>% 
  #filter(method_name %in% c("mlp", "auto_sklearn", "resnet")) %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str, data__max_num_samples) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  mutate(data__max_num_samples = as_factor(data__max_num_samples)) %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = method_name, shape = data__max_num_samples), alpha=1, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")

df %>% 
  filter(method_name %in% c("hgbt", "rf", "log_reg", "mlp")) %>% 
  group_by(data__keyword, iter, n_train) %>% 
  mutate(max_score_dataset = max(test_scores, na.rm=T)) %>% 
  mutate(relative_score = test_scores / max_score_dataset) %>% 
  ungroup() %>% 
  group_by(data__keyword, n_train, method_name) %>% 
  summarise(mean_relative_score = mean(relative_score)) %>% 
  #ungroup() %>% 
  #group_by(method_name, n_train) %>% 
  #summarise(mean_score = mean(relative_score, na.rm=T)) %>% 
  ggplot() +
  geom_point(aes(x = log10(n_train), y = mean_relative_score, color=method_name), alpha=0.5, size=2)


df %>% 
  filter(method_name %in% c("hgbt", "rf", "log_reg", "mlp")) %>% 
  filter(n_train %in% c(100, 1000, 10000)) %>% 
  group_by(data__keyword, iter, n_train) %>% 
  mutate(max_score_dataset = max(test_scores, na.rm=T)) %>% 
  mutate(relative_score = test_scores / max_score_dataset) %>% 
  ungroup() %>% 
  group_by(n_train, method_name) %>% 
  summarise(mean_relative_score = mean(relative_score)) %>% 
  #ungroup() %>% 
  #group_by(method_name, n_train) %>% 
  #summarise(mean_score = mean(relative_score, na.rm=T)) %>% 
  ggplot() +
  geom_line(aes(x = log10(n_train), y = mean_relative_score, color=method_name), alpha=0.5, size=2)
