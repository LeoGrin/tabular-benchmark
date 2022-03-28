library(tidyverse)

df <- read_csv("results/new/rf_modified.csv") %>% 
  mutate(max_features="auto") %>% 
  bind_rows(read_csv("results/new/rf_modified_2.csv") %>% 
              mutate(max_features="auto")) %>% 
  #bind_rows(read_csv("results/new/rf_modified_3.csv"))
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type")

df_mean <- df %>% 
  group_by(method_name, data__keyword, score_type, model_params_str, n_features_per_tree, n_estimators, max_depth) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup()

df_mean %>% 
  filter(n_estimators==100, n_features_per_tree <= 1, max_depth==3) %>% 
  #mutate(n_features_per_tree = as_factor(n_features_per_tree)) %>% 
  ggplot()+
  geom_point(aes(y=data__keyword, x=mean_score_iter, color=n_features_per_tree, group=n_features_per_tree), position=position_dodge(width=0.5)) +
  facet_wrap(~score_type)

df_mean %>% 
  group_by(data__keyword, score_type, max_depth, n_estimators) %>% #relative score compared to same params with all features
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(n_estimators, max_depth, n_features_per_tree, score_type) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  filter(score_type == "test_scores", n_features_per_tree<=1) %>%
  ggplot() +
  geom_line(aes(y=relative_score, x =n_features_per_tree, color=data__keyword), alpha=0.4) +
  geom_line(aes(y=mean_score, x =n_features_per_tree), size=1) +
  geom_point(aes(y=mean_score, x =n_features_per_tree), size=1) +
  ylim(0.8, 1.0)+
  facet_grid(max_depth~n_estimators)


df_mean %>% 
  group_by(data__keyword, score_type, max_depth, n_estimators) %>% #relative score compared to same params with all features
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(n_estimators, max_depth, n_features_per_tree, score_type) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  filter(n_features_per_tree<=1) %>% 
  filter(n_estimators==100, is.na(max_depth)) %>% 
  ggplot() +
  geom_line(aes(y=relative_score, x =n_features_per_tree, color=data__keyword), alpha=0.4) +
  geom_line(aes(y=mean_score, x =n_features_per_tree), size=1) +
  geom_point(aes(y=mean_score, x =n_features_per_tree), size=1) +
  ylim(0.8, 1) +
  facet_wrap(~score_type)


df_mean %>% 
  group_by(data__keyword, score_type) %>%  #relative score compared to all
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(n_estimators, max_depth, n_features_per_tree, score_type) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  filter(n_features_per_tree<=1) %>% 
  filter(n_estimators==100) %>% 
  mutate(max_depth = as_factor(max_depth)) %>% 
  ggplot() +
  geom_line(aes(y=mean_score, x =n_features_per_tree, color=max_depth)) +
  geom_point(aes(y=mean_score, x =n_features_per_tree, color=max_depth)) +
  #ylim(0.80, 1) +
  facet_wrap(~score_type)


#with max_features

#df <- read_csv("results/new/rf_modified.csv") %>% 
#  mutate(max_features="auto") %>% 
#  bind_rows(read_csv("results/new/rf_modified_2.csv") %>% 
#              mutate(max_features="auto")) %>% 
##  bind_rows(read_csv("results/new/rf_modified_3.csv") %>% 
#              mutate(max_features="none")) %>%
#  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type")
df <- read_csv("results/new/rf_modified_min_2_09_01.csv") %>% 
  mutate(max_features = if_else(is.na(max_features), "none", max_features)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type")

df_mean <- df %>% 
  group_by(method_name, data__keyword, score_type, model_params_str, 
           n_features_per_tree, max_depth, max_features) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T), sd_score_iter = sd(score, na.rm=T)) %>% 
  ungroup()

df_mean %>% 
  #filter(max_depth) %>% 
  #mutate(n_features_per_tree = as_factor(n_features_per_tree)) %>% 
  ggplot()+
  geom_point(aes(y=data__keyword, x=mean_score_iter, color=n_features_per_tree, group=n_features_per_tree), position=position_dodge(width=0.5)) +
  facet_wrap(~score_type)

df_mean %>% 
  group_by(data__keyword, score_type, max_depth, max_features) %>% #relative score compared to same params with all features
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_depth, max_features, n_features_per_tree, score_type) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  filter(score_type == "test_scores", n_features_per_tree<=1) %>%
  ggplot() +
  geom_line(aes(y=relative_score, x =n_features_per_tree, color=data__keyword), alpha=0.4) +
  geom_line(aes(y=mean_score, x =n_features_per_tree), size=1) +
  geom_point(aes(y=mean_score, x =n_features_per_tree), size=1) +
  #ylim(0.8, 1.0)+
  facet_grid(max_depth~max_features)


df_mean %>% 
  group_by(data__keyword, score_type, max_depth, max_features) %>% #relative score compared to same params with all features
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_features, max_depth, n_features_per_tree, score_type) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  #filter(n_features_per_tree<=1) %>% 
  filter(is.na(max_depth), max_features=="auto") %>% 
  ggplot() +
  geom_line(aes(y=relative_score, x =n_features_per_tree, color=data__keyword), alpha=0.4) +
  geom_line(aes(y=mean_score, x =n_features_per_tree), size=1) +
  geom_point(aes(y=mean_score, x =n_features_per_tree), size=1) +
  #ylim(0.8, 1) +
  facet_wrap(~score_type)


df_mean %>% 
  group_by(data__keyword, score_type) %>%  #relative score compared to all
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_features, max_depth, n_features_per_tree, score_type) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  mutate(max_depth = as_factor(max_depth)) %>% 
  ggplot() +
  geom_line(aes(y=mean_score, x =n_features_per_tree, color=max_depth, linetype=max_features)) +
  geom_point(aes(y=mean_score, x =n_features_per_tree, color=max_depth, shape=max_features)) +
  ylim(0.8, 1) +
  facet_wrap(~score_type)

df_mean %>% 
  group_by(data__keyword, score_type) %>%  #relative score compared to all
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_features, max_depth, n_features_per_tree, score_type) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  filter(n_features_per_tree >= 1) %>% 
  mutate(n_features_per_tree = as_factor(n_features_per_tree)) %>% 
  mutate(n_features_per_tree = if_else(n_features_per_tree == 1, "all", as.character(n_features_per_tree))) %>% 
  mutate(max_depth = as_factor(max_depth)) %>% 
  ggplot() +
  geom_line(aes(y=mean_score, x =n_features_per_tree, color=max_depth, linetype=max_features)) +
  geom_point(aes(y=mean_score, x =n_features_per_tree, color=max_depth, shape=max_features)) +
  ylim(0.8, 1) +
  facet_wrap(~score_type)


# with oob score
df <- read_csv("results/new/rf_modified_13_01.csv")


df_mean <- df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, data__keyword, score_type, model_params_str, 
           n_features_per_tree, max_depth, max_features, oob_score_per_tree,
           n_estimators) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T), sd_score_iter = sd(score, na.rm=T)) %>% 
  ungroup()


df_mean %>% 
  group_by(data__keyword, score_type) %>%  #relative score compared to all
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_features, max_depth, oob_score_per_tree, score_type, n_estimators) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  mutate(max_depth = as_factor(max_depth)) %>%
  mutate(n_estimators = as_factor(n_estimators)) %>% 
  ggplot() +
  geom_line(aes(y=mean_score, x =oob_score_per_tree, color=max_depth, linetype=n_estimators)) +
  geom_point(aes(y=mean_score, x =oob_score_per_tree, color=max_depth, shape=n_estimators)) +
  ylim(0.8, 1) +
  facet_wrap(~score_type)

# with oob full
df <- read_csv("results/new/rf_modified_13_01_full.csv")


df_mean <- df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, data__keyword, score_type, model_params_str, 
           n_features_per_tree, max_depth, max_features, oob_score_per_tree,
           n_estimators, threshold_estimator_selection, n_features) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T), sd_score_iter = sd(score, na.rm=T)) %>% 
  ungroup()


df_mean %>% 
  group_by(data__keyword, score_type) %>%  #relative score compared to all
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_features, max_depth, oob_score_per_tree, score_type, n_estimators) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  mutate(max_depth = as_factor(max_depth)) %>%
  mutate(n_estimators = as_factor(n_estimators)) %>% 
  ggplot() +
  geom_line(aes(y=mean_score, x =oob_score_per_tree, color=max_depth, linetype=n_estimators)) +
  geom_point(aes(y=mean_score, x =oob_score_per_tree, color=max_depth, shape=n_estimators)) +
  ylim(0.8, 1) +
  facet_wrap(~score_type)

df_mean %>% 
  group_by(data__keyword, score_type) %>%  #relative score compared to all
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_features, max_depth, n_features_per_tree, score_type,
           n_estimators, oob_score_per_tree, threshold_estimator_selection) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  filter(n_estimators == 100, threshold_estimator_selection==0.9) %>% 
  mutate(max_depth = as_factor(max_depth)) %>% 
  ggplot() +
  geom_line(aes(y=mean_score, x =n_features_per_tree, color=max_depth, linetype=oob_score_per_tree)) +
  geom_point(aes(y=mean_score, x =n_features_per_tree, color=max_depth, shape=oob_score_per_tree)) +
  ylim(0.8, 1) +
  facet_wrap(~score_type)


df_mean %>% 
  group_by(data__keyword, score_type) %>%  #relative score compared to all
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_features, max_depth, n_features_per_tree, score_type,
           n_estimators, oob_score_per_tree, threshold_estimator_selection) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  filter(threshold_estimator_selection==0.9, is.na(max_depth)) %>% 
  mutate(max_depth = as_factor(max_depth)) %>%
  mutate(n_estimators = as_factor(n_estimators)) %>% 
  ggplot() +
  geom_line(aes(y=mean_score, x =n_features_per_tree, color=oob_score_per_tree, linetype=n_estimators)) +
  geom_point(aes(y=mean_score, x =n_features_per_tree, color=oob_score_per_tree, shape=n_estimators)) +
  ylim(0.8, 1) +
  facet_wrap(~score_type)

df_mean %>% 
  group_by(data__keyword, score_type,
           max_features, max_depth,
           n_estimators, oob_score_per_tree, threshold_estimator_selection) %>%  #relative score compared to same param with n_features_per_trees==1
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  #filter(n_features > 10) %>% 
  group_by(max_features, max_depth, n_features_per_tree, score_type,
           n_estimators, oob_score_per_tree, threshold_estimator_selection) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  filter(threshold_estimator_selection==0.9, is.na(max_depth)) %>% 
  mutate(max_depth = as_factor(max_depth)) %>%
  mutate(n_estimators = as_factor(n_estimators)) %>% 
  ggplot() +
  geom_line(aes(y=mean_score, x =n_features_per_tree, color=oob_score_per_tree, linetype=n_estimators)) +
  geom_point(aes(y=mean_score, x =n_features_per_tree, color=oob_score_per_tree, shape=n_estimators)) +
  ylim(0.8, 1) +
  facet_wrap(~score_type)

df_mean %>% 
  group_by(data__keyword, score_type) %>% #relative score compared to all
  mutate(relative_score = mean_score_iter / max(mean_score_iter)) %>% 
  ungroup() %>% 
  group_by(max_features, max_depth, n_features_per_tree, score_type) %>% 
  mutate(mean_score = mean(relative_score)) %>% 
  ungroup() %>% 
  #filter(score_type == "test_scores") %>%
  #filter(n_features_per_tree<=1) %>% 
  filter(is.na(max_depth), n_estimators==300, threshold_estimator_selection==0.9,
         oob_score_per_tree==T) %>% 
  ggplot() +
  #geom_line(aes(y=relative_score, x =n_features_per_tree, color=n_features, group=data__keyword), alpha=0.4) +
  geom_line(aes(y=relative_score, x =n_features_per_tree, color=data__keyword), alpha=0.4) +
  geom_line(aes(y=mean_score, x =n_features_per_tree), size=1) +
  geom_point(aes(y=mean_score, x =n_features_per_tree), size=1) +
  #ylim(0.8, 1) +
  facet_wrap(~score_type)



