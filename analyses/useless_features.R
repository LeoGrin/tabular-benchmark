library(tidyverse)

df <- read_csv("results/useless_features.csv")


df %>% 
  filter(transform__0__num_features_to_remove < 1) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, 
           transform__0__method_name, transform__0__num_features_to_remove, data__keyword) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ggplot() +
  geom_line(aes(x = transform__0__num_features_to_remove, y = mean_score_iter, color=data__keyword)) +
  geom_point(aes(x = transform__0__num_features_to_remove, y = mean_score_iter, color=data__keyword)) +
  facet_wrap(method_name~score_type)


df %>% 
  ggplot() +
  geom_point(aes(x = transform__0__num_features_to_remove, y= n_features, color = data__keyword))
