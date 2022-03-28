library(tidyverse)

df <- read_csv("results/random_search/res_random_search_cpu_01_12.csv")

df_iter <- df %>% 
  filter(method_name %in% c("rf", "hgbt")) %>% 
  #bind_rows(df_default) %>% 
  group_by(data__keyword, transform__1__method_name, method_name, model_params_str) %>% 
  summarise(mean_score_iter = mean(test_scores, na.rm=T))%>% 
  ungroup() %>% 
  group_by(data__keyword, transform__1__method_name) %>% 
  mutate(relative_score = mean_score_iter / max(mean_score_iter, na.rm=T))

df_iter <- df_iter %>% 
  mutate(data__keyword = if_else(transform__1__method_name=="no_transform", data__keyword, map_chr(data__keyword, ~paste0(., "_rot", sep=""))))
         
#Best hp on other dataset
#For each model: take the best hp combination for the mean of all dataset except one and evaluate on loo dataset
# repeat for all datasets

best_hp_df <- df_iter %>% 
  filter(transform__1__method_name == "no_transform" & data__keyword != "mnist_1_7") %>% 
  group_by(data__keyword, method_name) %>% 
  mutate(best_score_iter = max(mean_score_iter, na.rm=T)) %>% 
  filter(best_score_iter == mean_score_iter) %>% 
  select(data__keyword, method_name, model_params_str, best_score_iter)

df_iter %>% 
       left_join(best_hp_df, by = c("method_name", "model_params_str")) %>% 
       filter(!is.na(best_score_iter)) %>% 
  filter(data__keyword.x != data__keyword.y) %>%  #remove best hp on same dataset
  ggplot() +
  geom_point(aes(x = relative_score, y=data__keyword.x, color=method_name, group=method_name), position = position_dodge(width=0.5)) +
  ggtitle("For each dataset, relative score of the best hps found on the other datasets") +
  xlab("Relative score of hp") +
  ylab("Dataset on which the hp is evaluated")



df_iter %>% 
  left_join(best_hp_df, by = c("method_name", "model_params_str")) %>% 
  filter(!is.na(best_score_iter)) %>% 
  filter(data__keyword.x != data__keyword.y) %>%  #remove best hp on same dataset
  group_by(transform__1__method_name) %>% 
  summarise(relative_score_mean = mean(relative_score))
  
df_iter %>% 
  group_by(transform__1__method_name) %>% 
  summarise(relative_score_mean = mean(relative_score))




#for each dataset, how the best hp on this dataset performs on all the other datasets

best_hp_df <- df_iter %>% 
  group_by(data__keyword, method_name) %>% 
  mutate(best_score_iter = max(mean_score_iter, na.rm=T)) %>% 
  filter(best_score_iter == mean_score_iter) %>% 
  select(data__keyword, method_name, model_params_str, best_score_iter, transform__1__method_name)

df_iter %>% 
  filter(transform__1__method_name == "no_transform" & data__keyword != "mnist_1_7") %>% 
  left_join(best_hp_df, by = c("method_name", "model_params_str")) %>% 
  filter(!is.na(best_score_iter)) %>% 
  filter(data__keyword.x != data__keyword.y) %>% 
  ggplot() +
  geom_point(aes(x = relative_score, y=data__keyword.y, color=method_name, group=method_name), position = position_dodge(width=0.5)) +
  ggtitle("For each dataset, relative score on the other datasets of the best hp found on this dataset") +
  xlab("Relative score of hp on others datasets") +
  ylab("Dataset on which the hp is found")

df_iter %>% 
  filter(transform__1__method_name == "no_transform" & data__keyword != "mnist_1_7") %>% 
  left_join(best_hp_df, by = c("method_name", "model_params_str")) %>% 
  filter(!is.na(best_score_iter)) %>% 
  filter(data__keyword.x != data__keyword.y) %>% 
  group_by(transform__1__method_name.y) %>% 
  summarise(relative_score_mean = mean(relative_score))


######################################@
# depth impact
df_depth <- read_csv("results/new/depth_impact.csv")

df_depth_iter <- df_depth %>% 
  mutate(data__keyword = if_else(transform__1__method_name=="no_transform", data__keyword, map_chr(data__keyword, ~paste0(., "_rot", sep="")))) %>% 
  group_by(data__keyword, method_name, max_depth, n_estimators, learning_rate, transform__1__method_name) %>% 
  summarise(mean_score_iter = mean(test_scores, na.rm=T))%>% 
  ungroup() %>% 
  group_by(data__keyword) %>% 
  mutate(relative_score = mean_score_iter / max(mean_score_iter, na.rm=T))

df_depth_iter %>% 
  filter(is.na(n_estimators) & is.na(learning_rate)) %>% 
  ggplot() +
  geom_point(aes(x = relative_score, y=data__keyword, color=max_depth, group=method_name, shape=method_name), position = position_dodge(width=0.5))
  
df_depth_iter %>% 
  filter(is.na(n_estimators) & is.na(learning_rate)) %>% 
  filter(data__keyword != "mnist_1_7") %>% 
  group_by(max_depth, method_name, transform__1__method_name) %>% 
  summarise(mean_relative_score = mean(relative_score), sd_relative_score = sd(relative_score), min_relative_score=min(relative_score), max_relative_score = max(relative_score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_point(aes(x = max_depth, y=mean_relative_score, color = transform__1__method_name)) +
  geom_line(aes(x = max_depth, y=mean_relative_score, color = transform__1__method_name)) +
  geom_ribbon(aes(x = max_depth, ymin=min_relative_score, ymax=max_relative_score, fill=transform__1__method_name), alpha=0.2)+
  facet_wrap(~method_name) +
  ggtitle("Impact of max depth on test score") +
  xlab("Max depth") +
  ylab("Mean relative score")
  

df_depth_iter %>% 
  filter(is.na(max_depth) & method_name=="rf") %>% 
  filter(data__keyword != "mnist_1_7") %>% 
  group_by(n_estimators, method_name, transform__1__method_name) %>% 
  summarise(mean_relative_score = mean(relative_score), sd_relative_score = sd(relative_score), min_relative_score=min(relative_score), max_relative_score = max(relative_score)) %>% 
  ggplot() +
  geom_point(aes(x = n_estimators, y=mean_relative_score, color = transform__1__method_name)) +
  geom_line(aes(x = n_estimators, y=mean_relative_score, color = transform__1__method_name)) +
  geom_ribbon(aes(x = n_estimators, ymin=mean_relative_score - 1 * sd_relative_score, ymax=mean_relative_score + 1 * sd_relative_score, fill=transform__1__method_name), alpha=0.2)+
  ggtitle("Impact of number of estimators on test score") +
  xlab("Number of estimators") +
  ylab("Mean relative score")

df_depth_iter %>% 
  filter(is.na(max_depth) & method_name=="hgbt") %>% 
  group_by(learning_rate, method_name, transform__1__method_name) %>% 
  summarise(mean_relative_score = mean(relative_score), sd_relative_score = sd(relative_score), min_relative_score=min(relative_score), max_relative_score = max(relative_score)) %>% 
  ggplot() +
  geom_point(aes(x = learning_rate, y=mean_relative_score, color = transform__1__method_name)) +
  geom_line(aes(x = learning_rate, y=mean_relative_score, color = transform__1__method_name)) +
  geom_ribbon(aes(x = learning_rate, ymin=mean_relative_score - 1 * sd_relative_score, ymax=mean_relative_score + 1 * sd_relative_score, fill=transform__1__method_name), alpha=0.2)+
  ggtitle("Impact of learning rate on hgbt test score") +
  xlab("Learning rate") +
  ylab("Mean relative score")

df_depth_iter %>% 
  filter(is.na(max_depth) & method_name=="rf") %>% 
  ggplot() +
  geom_point(aes(x = relative_score, y=data__keyword, color=n_estimators))

df_depth_iter %>% 
  filter(is.na(max_depth) & method_name=="hgbt") %>% 
  ggplot() +
  geom_point(aes(x = relative_score, y=data__keyword, color=learning_rate))


  