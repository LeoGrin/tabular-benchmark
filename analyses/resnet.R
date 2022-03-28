library(tidyverse)

total_df <- read_csv("results/new/res_new_real_bonus_2_10_11.csv") %>% filter(method_name == "autosklearn") %>% 
  #bind_rows(read_csv("results/new/lounici_05_12.csv")) %>% 
  bind_rows(read_csv("results/new/resnet.csv")) 


total_df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(n_layers==3, hidden_size==256) %>% 
  group_by(method_name, data__keyword, score_type, lr) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = log10(lr)), alpha=1, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)


total_df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(n_layers==3, lr==0.001) %>% 
  group_by(method_name, data__keyword, score_type, hidden_size) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = hidden_size), alpha=1, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

total_df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(hidden_size==256, lr==0.001) %>% 
  group_by(method_name, data__keyword, score_type, n_layers) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = n_layers), alpha=1, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)



#how to achieve great train score
total_df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(hidden_size==256) %>% 
  group_by(method_name, data__keyword, score_type, hidden_size, lr, n_layers) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  filter(score_type == "train_scores") %>% 
  group_by(data__keyword,  hidden_size, lr, n_layers) %>% 
  mutate(score = median(mean_score_iter)) %>% 
  mutate(hidden_size = as_factor(hidden_size), lr=as_factor(lr), n_layers=as_factor(n_layers)) %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = lr, x = score, color = n_layers), alpha=0.7, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)


total_df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(n_layers==3) %>% 
  group_by(method_name, data__keyword, score_type, hidden_size, lr, n_layers) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  filter(score_type == "train_scores") %>% 
  group_by(data__keyword,  hidden_size, lr, n_layers) %>% 
  mutate(score = median(mean_score_iter)) %>% 
  mutate(hidden_size = as_factor(hidden_size), lr=as_factor(lr)) %>% 
  
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = lr, x = score, color = hidden_size), alpha=0.7, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)

total_df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  filter(lr==0.001) %>% 
  group_by(method_name, data__keyword, score_type, hidden_size, lr, n_layers) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  filter(score_type == "train_scores") %>% 
  group_by(data__keyword,  hidden_size, lr, n_layers) %>% 
  mutate(score = median(mean_score_iter)) %>% 
  mutate(hidden_size = as_factor(hidden_size), lr=as_factor(lr), n_layers=as_factor(n_layers)) %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = hidden_size, x = score, color = n_layers), alpha=0.7, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)
