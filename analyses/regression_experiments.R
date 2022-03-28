library(tidyverse)

df <- read_csv("results/regression_22_03.csv") %>%
  bind_rows(read_csv("results/regression_synthetic_25_03.csv")) #TODO: max epochs don't match
 # bind_rows( read_csv("results/regression_periods.csv"))


#View(read_csv("results/regression_synthetic_25_03.csv"))
# df %>% 
#        filter(target__period_size == 0.15, method_name == "mlp", n_layers == 3) %>% 
#        select(train_scores, data__num_samples)
# 
# df %>% 
#   #mutate(points_per_period = data__num_samples * (target__period_size / 4)))
#   pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
#   group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, target__period_size) %>% 
#   summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
#   #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
#   filter(score_type == "train_scores") %>% 
#   #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
#   mutate(n_layers = as.character(n_layers)) %>% 
#   mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
#   ggplot() +
#   geom_line(aes(x = target__period_size, y = mean_score_iter, color=method_name, linetype=n_layers)) +
#   geom_point(aes(x = target__period_size, y = mean_score_iter, color=method_name, shape=n_layers)) +
#   facet_wrap(~data__num_samples)
  
# df %>% 
#   mutate(points_per_period = data__num_samples * (target__period_size / 4)) %>% 
#   pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
#   group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, target__period_size, points_per_period) %>% 
#   summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
#   #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
#   filter(score_type == "train_scores", method_name == "mlp") %>% 
#   #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
#   mutate(n_layers = as.character(n_layers)) %>% 
#   #mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
#   ggplot() +
#   #geom_line(aes(x = points_per_period, y = mean_score_iter, color=method_name, linetype=n_layers)) +
#   geom_point(aes(x = points_per_period, y = mean_score_iter, color=target__period_size, shape=n_layers))#+
#   #facet_wrap(~data__num_samples)

df %>% 
  mutate(points_per_period = data__num_samples * (target__period_size / 4)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, target__period_size, points_per_period, target__period, batch_size) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  filter(score_type == "train_scores", method_name == "mlp") %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  mutate(n_layers = as.character(n_layers)) %>%
  #mutate(target__period_size = as.character(target__period_size)) %>%
  #filter(n_layers == 3) %>% 
  #filter(batch_size == 256) %>% 
  #mutate(a = 0.025 + 1 / points_per_period) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  #mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
  ggplot() +
  #geom_line(aes(x = points_per_period, y = mean_score_iter, color=method_name, linetype=n_layers)) +
  geom_point(aes(x = points_per_period, y = mean_score_iter, color=log10(data__num_samples), size=target__period_size), alpha=0.5) +#++
  #geom_line(aes(x = points_per_period, y = mean_score_iter, color=data__num_samples, size=target__period_size)) +
  scale_x_log10() +
  scale_y_log10() +
  #geom_line(aes(x = points_per_period, y = a)) +
  facet_wrap(n_layers~batch_size)
#facet_wrap(~data__num_samples)

df %>% 
  mutate(points_per_period = data__num_samples * (target__period_size / 4)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, target__period_size, points_per_period, target__period, batch_size) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  filter(score_type == "train_scores", method_name == "mlp") %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  mutate(n_layers = as.character(n_layers)) %>%
 # mutate(target__period_size = as.character(target__period_size)) %>%
  #filter(n_layers == 3) %>% 
  #filter(batch_size == 256) %>% 
  #mutate(a = 0.025 + 1 / points_per_period) %>% 
  mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  #mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
  ggplot() +
  #geom_line(aes(x = points_per_period, y = mean_score_iter, color=method_name, linetype=n_layers)) +
  geom_point(aes(x = target__period_size, y = mean_score_iter, color=data__num_samples)) +#++
  geom_line(aes(x = target__period_size, y = mean_score_iter, color=data__num_samples)) +
  scale_x_log10() +
  scale_y_log10() +
  #geom_line(aes(x = points_per_period, y = a)) +
  facet_wrap(n_layers~batch_size)
#facet_wrap(~data__num_samples)



###################################

df <- read_csv("results/regression_22_03_uniform.csv")# %>% 
# bind_rows( read_csv("results/regression_periods.csv"))


df %>% 
  mutate(points_per_period = data__num_samples * (data__period_size / 4)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, data__period_size, points_per_period, data__period, batch_size) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  filter(score_type == "train_scores", method_name == "mlp") %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  mutate(n_layers = as.character(n_layers)) %>%
  mutate(data__period_size = as.character(data__period_size)) %>%
  #filter(n_layers == 3) %>% 
  #filter(batch_size == 256) %>% 
  #mutate(a = 0.025 + 1 / points_per_period) %>% 
  mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  #mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
  ggplot() +
  #geom_line(aes(x = points_per_period, y = mean_score_iter, color=method_name, linetype=n_layers)) +
  geom_point(aes(x = points_per_period, y = mean_score_iter, color=data__num_samples, shape=data__period_size)) +#++
  geom_line(aes(x = points_per_period, y = mean_score_iter, color=data__num_samples, shape=data__period_size)) +
  scale_x_log10() +
  scale_y_log10() +
  #geom_line(aes(x = points_per_period, y = a)) +
  facet_wrap(n_layers~batch_size)
#facet_wrap(~data__num_samples)

df %>% 
  mutate(points_per_period = data__num_samples * (target__period_size / 4)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, target__period_size, points_per_period, target__period, batch_size) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  filter(score_type == "train_scores", method_name == "mlp") %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  mutate(n_layers = as.character(n_layers)) %>%
  # mutate(target__period_size = as.character(target__period_size)) %>%
  #filter(n_layers == 3) %>% 
  #filter(batch_size == 256) %>% 
  #mutate(a = 0.025 + 1 / points_per_period) %>% 
  mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  #mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
  ggplot() +
  #geom_line(aes(x = points_per_period, y = mean_score_iter, color=method_name, linetype=n_layers)) +
  geom_point(aes(x = target__period_size, y = mean_score_iter, color=data__num_samples)) +#++
  geom_line(aes(x = target__period_size, y = mean_score_iter, color=data__num_samples)) +
  scale_x_log10() +
  scale_y_log10() +
  #geom_line(aes(x = points_per_period, y = a)) +
  facet_wrap(n_layers~batch_size)
#facet_wrap(~data__num_samples)

######################


  
  df %>% 
    mutate(points_per_period = data__num_samples * (target__period_size / 4)) %>% 
    pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
    group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, target__period_size, points_per_period, target__period) %>% 
    summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
    #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
    filter(score_type == "train_scores", method_name == "mlp") %>% 
    #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
    mutate(n_layers = as.character(n_layers)) %>% 
    mutate(a = 0.025 + 1 / points_per_period) %>% 
    mutate(data__num_samples = as_factor(data__num_samples)) %>% 
    #mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
    ggplot() +
    #geom_line(aes(x = points_per_period, y = mean_score_iter, color=method_name, linetype=n_layers)) +
    geom_point(aes(x = points_per_period, y = mean_score_iter, color=target__period_size, shape=data__num_samples)) +#++
    geom_line(aes(x = points_per_period, y = mean_score_iter, color=target__period_size, group=target__period_size))
  #geom_line(aes(x = points_per_period, y = a)) +
  facet_wrap(~n_layers)


df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(method_name == "mlp") %>% 
  #filter(data__num_samples == 15000) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ggplot() +
  geom_line(aes(x = target__period_size, y = score, color=data__num_samples, group = data__num_samples)) +
  geom_point(aes(x = target__period_size, y = score, color=data__num_samples)) +
  facet_wrap(~score_type)

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(method_name == "rf") %>% 
  #filter(data__num_samples == 15000) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ggplot() +
  geom_line(aes(x = target__period_size, y = score, color=data__num_samples, group = data__num_samples)) +
  geom_point(aes(x = target__period_size, y = score, color=data__num_samples)) +
  facet_wrap(~score_type)

df <- read_csv("results/regression_tqt.csv")

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, target__period_size, transform__0__method_name) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  filter(data__num_samples %in% c(500, 1000, 2500)) %>% 
  filter(score_type == "train_scores") %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  mutate(n_layers = as.character(n_layers)) %>% 
  mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
  ggplot() +
  geom_line(aes(x = target__period_size, y = mean_score_iter, color=method_name, linetype=transform__0__method_name)) +
  geom_point(aes(x = target__period_size, y = mean_score_iter, color=method_name, shape=transform__0__method_name)) +
  facet_wrap(~data__num_samples)


df <- read_csv("results/initialization_regression.csv") %>% 
  bind_rows(read_csv("results/initialization_regression_mlp.csv"))

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, noise_std) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  filter(data__num_samples %in% c(500, 1000, 2500)) %>% 
  filter(score_type == "train_scores") %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  mutate(n_layers = as.character(n_layers)) %>% 
  mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
  ggplot() +
  geom_line(aes(x = noise_std, y = mean_score_iter, color=method_name)) +
  geom_point(aes(x = noise_std, y = mean_score_iter, color=method_name)) +
  geom_hline(color="red", yintercept = 0.156) +
  facet_wrap(~data__num_samples)


df <- read_csv("results/initialization_regression_zoom.csv") %>% 
  bind_rows(read_csv("results/initialization_regression.csv"))

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, noise_std) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  filter(data__num_samples %in% c(500, 1000, 2500)) %>% 
  filter(score_type == "train_scores") %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  mutate(n_layers = as.character(n_layers)) %>% 
  mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
  ggplot() +
  geom_line(aes(x = noise_std, y = mean_score_iter, color=method_name)) +
  geom_point(aes(x = noise_std, y = mean_score_iter, color=method_name)) +
  #geom_hline(aes(yintercept = mean_score_iter, color=method_name)) +
  facet_wrap(~data__num_samples) +
  scale_x_log10()


df <- read_csv("results/initialization_regression_uniform.csv")# 
 # bind_rows(read_csv("results/initialization_regression.csv"))

df %>% 
  filter(noise_std < 0.05) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__num_samples, n_layers, noise_std) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  filter(data__num_samples %in% c(500, 1000, 2500)) %>% 
  filter(score_type == "train_scores") %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  mutate(n_layers = as.character(n_layers)) %>% 
  mutate(n_layers = if_else(is.na(n_layers), "rf", n_layers)) %>% 
  ggplot() +
  geom_line(aes(x = noise_std, y = mean_score_iter, color=method_name)) +
  geom_point(aes(x = noise_std, y = mean_score_iter, color=method_name)) +
  #geom_hline(aes(yintercept = mean_score_iter, color=method_name)) +
  facet_wrap(~data__num_samples) +
  scale_x_log10()

