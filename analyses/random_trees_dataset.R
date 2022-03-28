library(tidyverse)

#df <- read_csv("results/new/mlp_trees_03_01.csv") %>% 
#  bind_rows(read_csv("results/new/mlp_trees_04_01.csv"))

df <- read_csv("results/new/mlp_trees_04_01.csv")

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(module__dropout_prob==0.5) %>% 
  ggplot() +
  geom_point(aes(y = method_name, x = score, color = n_layers, shape = module__resnet), alpha=1, position=position_dodge2(width=0.5, reverse=T)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")



df <- read_csv("results/new/sparse_trees_05_01.csv")

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(module__n_w == 5, lr == 0.1) %>% 
  ggplot() +
  geom_point(aes(y = method_name, x = score, color = module__n_layers), alpha=1, position=position_dodge2(width=0.5, reverse=T)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")

df <- read_csv("results/new/sparse_10_01.csv")

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(lr == 0.01 | model_name ==  "mlp", 
         module__dropout_prob==0 | model_name ==  "mlp",
         module__batchnorm == TRUE,
         ) %>% 
  mutate(module__n_w = if_else(is.na(module__n_w), -1, module__n_w)) %>% 
  mutate(module__n_w = as_factor(module__n_w)) %>% 
  ggplot() +
  geom_point(aes(y = method_name, x = score, color = module__n_layers, shape=module__n_w), alpha=1, position=position_dodge2(width=0.5, reverse=T)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")


View(df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(lr == 0.01 | model_name ==  "mlp", 
         module__dropout_prob==0 | model_name ==  "mlp",
         module__batchnorm == TRUE,
  ) %>% 
  mutate(module__n_w = if_else(is.na(module__n_w), -1, module__n_w)) %>% 
  mutate(module__n_w = as_factor(module__n_w)) %>%  
  group_by(module__n_layers, module__n_w, method_name, score_type) %>% 
  summarise(mean_score = mean(score), count = n()))
