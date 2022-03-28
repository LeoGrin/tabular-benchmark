library(tidyverse)

total_df <- read_csv("results/new/res_new_real_bonus_2_10_11.csv") %>% 
  #bind_rows(read_csv("results/new/lounici_05_12.csv")) %>% 
  bind_rows(read_csv("results/new/nam.csv")) %>% 
  bind_rows(read_csv("results/new/nam_embedding.csv"))
#bind_rows(read_csv("results/new/lounici.csv")) %>% 
#bind_rows(read_csv("results/new/lounici2.csv") %>% mutate(method_name = "lounici2"))

total_df %>% 
  filter(method_name %in% c("nam", "auto_sklearn")) %>% 
  mutate(module__embedding_size = if_else(is.na(module__embedding_size), 1, module__embedding_size)) %>% 
  mutate(module__mlp_hidden_sizes = if_else(is.na(module__mlp_hidden_sizes), mlp_hidden_sizes, module__mlp_hidden_sizes)) %>% 
  
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  mutate(module__embedding_size = as_factor(module__embedding_size)) %>% 
  group_by(method_name, data__keyword, score_type, module__mlp_hidden_sizes, module__embedding_size) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = module__mlp_hidden_sizes, shape=module__embedding_size), alpha=1, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type)



