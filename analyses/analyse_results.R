library(tidyverse)


df_1 <- read_csv("results/res_gaussian_23_10.csv", col_types = cols("data__df"=col_number())) %>% 
  mutate(id = paste("gaussian", id, sep = ""))

df_2 <- read_csv("results/res_student_18_10.csv", col_types = cols("data__df"=col_number())) %>% 
  mutate(id = paste("student", id, sep="")) 

df_3 <- read_csv("results/res_real_18_10.csv") %>% 
  mutate(id = paste("real_18", id, sep=""))

df_4 <- read_csv("results/res_real_kmeans_22_10.csv") %>% 
  mutate(id = paste("real_kmeans", id, sep="")) %>% 
  mutate(data__max_num_samples = as.character(data__max_num_samples))

df_5 <- read_csv("results/res_real_22_10.csv") %>% 
  mutate(id = paste("real_22", id, sep="")) %>% 
  mutate(data__max_num_samples = as.character(data__max_num_samples))

df_6 <- read_csv("results/res_real_no_transform_27_10.csv") %>% 
  mutate(id = paste("real_27", id, sep="")) %>% 
  mutate(data__max_num_samples = as.character(data__max_num_samples))

df_7 <- read_csv("results/res_real_remove_27_10.csv") %>% 
  mutate(id = paste("real_remove", id, sep="")) %>% 
  mutate(data__max_num_samples = as.character(data__max_num_samples))

df_8 <- read_csv("results/res") %>% 
  mutate(id = paste("real_remove", id, sep="")) %>% 
  mutate(data__max_num_samples = as.character(data__max_num_samples))

df_rotation <- read_csv("results/res_rotation.csv") %>% bind_rows(read_csv("results/res_rotation_local.csv"))

df_rotation <- df_rotation %>% select(-...1, -iter, -id)


  
########################################

#diff with and without rotation
#test
#without gaussienization
df_rotation %>% 
  pivot_longer(cols=starts_with("test_score"), names_to = "model", values_to="test_score", names_prefix = "test_scores_") %>% 
  filter(transform__0__method_name %in% c("random_rotation", "no_transform"))%>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  ggplot() +
  geom_jitter(aes(x = test_score, y=data__keyword, color = transform__0__method_name), alpha=0.5, width=0, height=0.1) +
  facet_wrap(~model) +
  ggtitle("Without Gaussienization")

#diff with and without rotation
#test
#with gaussienization
df_rotation %>% 
  pivot_longer(cols=starts_with("test_score"), names_to = "model", values_to="test_score", names_prefix = "test_scores_") %>% 
  filter(transform__0__method_name == "gaussienize", transform__0__type=="quantile")%>% 
  #filter(transform__0__method_name %in% c("random_rotation", "no_transform"))%>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  mutate(transform__1__method_name = if_else(is.na(transform__1__method_name), "no_transform", transform__1__method_name)) %>% 
  ggplot() +
  geom_jitter(aes(x = test_score, y=data__keyword, color = transform__1__method_name), alpha=0.5, width=0, height=0.1) +
  facet_wrap(~model) +
  ggtitle("With Gaussienization (quantile)")

#diff with and without rotation
#train
#without gaussienization
df_rotation %>% 
  pivot_longer(cols=starts_with("train_score"), names_to = "model", values_to="train_score", names_prefix = "test_scores_") %>% 
  filter(transform__0__method_name %in% c("random_rotation", "no_transform"))%>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  ggplot() +
  geom_jitter(aes(x = train_score, y=data__keyword, color = transform__0__method_name), alpha=0.5, width=0, height=0.1) +
  facet_wrap(~model) +
  ggtitle("Without Gaussienization")

#diff with and without rotation
#train
#with gaussienization
df_rotation %>% 
  pivot_longer(cols=starts_with("train_score"), names_to = "model", values_to="train_score", names_prefix = "test_scores_") %>% 
  filter(transform__0__method_name == "gaussienize", transform__0__type=="quantile")%>% 
  #filter(transform__0__method_name %in% c("random_rotation", "no_transform"))%>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  mutate(transform__1__method_name = if_else(is.na(transform__1__method_name), "no_transform", transform__1__method_name)) %>% 
  ggplot() +
  geom_jitter(aes(x = train_score, y=data__keyword, color = transform__1__method_name), alpha=0.5, width=0, height=0.1) +
  facet_wrap(~model) +
  ggtitle("With Gaussienization (quantile)")

#model comparison
#test
df_rotation %>% 
  pivot_longer(cols=starts_with("test_score"), names_to = "model", values_to="test_score", names_prefix = "test_scores_") %>% 
  filter(transform__0__type=="quantile" | is.na(transform__0__type)) %>% 
  #filter(transform__0__method_name == "gaussienize", transform__0__type=="quantile")%>% 
  #filter(transform__1__method_name == "random_rotation") %>% 
  mutate(transform__1__method_name = if_else(is.na(transform__1__method_name), "no_transform", transform__1__method_name)) %>% 
  mutate(transform__1__method_name = if_else(transform__0__method_name == "random_rotation",transform__0__method_name,  transform__1__method_name)) %>% 
  mutate(transform__0__method_name = if_else(transform__1__method_name == "random_rotation" & transform__0__method_name == "random_rotation","no_transform",  transform__0__method_name)) %>%
  #filter(transform__0__method_name == "random_rotation") %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  ggplot() +
  geom_jitter(aes(x = test_score, y=data__keyword, color = model), alpha=0.5, width=0, height=0.1)+
  facet_grid(transform__1__method_name~transform__0__method_name)

#model comparison
#train
df_rotation %>% 
  pivot_longer(cols=starts_with("train_score"), names_to = "model", values_to="train_score", names_prefix = "train_scores_") %>% 
  filter(transform__0__type=="quantile" | is.na(transform__0__type)) %>% 
  #filter(transform__0__method_name == "gaussienize", transform__0__type=="quantile")%>% 
  #filter(transform__1__method_name == "random_rotation") %>% 
  mutate(transform__1__method_name = if_else(is.na(transform__1__method_name), "no_transform", transform__1__method_name)) %>% 
  mutate(transform__1__method_name = if_else(transform__0__method_name == "random_rotation",transform__0__method_name,  transform__1__method_name)) %>% 
  mutate(transform__0__method_name = if_else(transform__1__method_name == "random_rotation" & transform__0__method_name == "random_rotation","no_transform",  transform__0__method_name)) %>%
  #filter(transform__0__method_name == "random_rotation") %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  ggplot() +
  geom_jitter(aes(x = train_score, y=data__keyword, color = model), alpha=0.5, width=0, height=0.1)+
  facet_grid(transform__1__method_name~transform__0__method_name)

##############################################
df_rotation %>% 
  pivot_longer(cols=starts_with("test_score"), names_to = "model", values_to="test_score", names_prefix = "test_scores_") %>% 
  filter(model == "mlp_skorch") %>% 
  filter(transform__0__method_name != "random_rotation") %>% 
  filter(is.na(transform__1__method_name)) %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  group_by(data__keyword, transform__0__type) %>% 
  mutate(mean_test_score = mean(test_score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(x = test_score, y=data__keyword, color = transform__0__type), alpha=0.5, width=0, height=0.1) +
  geom_point(aes(x = mean_test_score, y=data__keyword, fill = transform__0__type), size=10, alpha=0.6, color="black", shape=21) +
  ggtitle("gaussienization comparison (test)")


df_rotation %>% 
  pivot_longer(cols=starts_with("train_score"), names_to = "model", values_to="train_score", names_prefix = "train_scores_") %>% 
  filter(model == "mlp_skorch") %>% 
  filter(transform__0__method_name != "random_rotation") %>% 
  filter(is.na(transform__1__method_name)) %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  group_by(data__keyword, transform__0__type) %>% 
  mutate(mean_train_score = mean(train_score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(x = train_score, y=data__keyword, color = transform__0__type), alpha=0.5, width=0, height=0.1) +
  geom_point(aes(x = mean_train_score, y=data__keyword, fill = transform__0__type), size=10, alpha=0.6, color="black", shape=21) +
  ggtitle("gaussienization comparison (train)")
#####################
df_rotation %>% 
  pivot_longer(cols=starts_with("test_score"), names_to = "model", values_to="test_score", names_prefix = "test_scores_") %>% 
  filter(transform__0__method_name != "random_rotation") %>% 
  filter(is.na(transform__1__method_name)) %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  group_by(data__keyword, transform__0__type, model) %>% 
  mutate(mean_test_score = mean(test_score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(x = test_score, y=data__keyword, color = transform__0__type), alpha=0.5, width=0, height=0.1) +
  geom_point(aes(x = mean_test_score, y=data__keyword, fill = transform__0__type), size=10, alpha=0.6, color="black", shape=21) +
  ggtitle("gaussienization comparison (test) (no rotation)") +
  facet_wrap(~model)


df_rotation %>% 
  pivot_longer(cols=starts_with("test_score"), names_to = "model", values_to="test_score", names_prefix = "test_scores_") %>% 
  filter(transform__1__method_name == "random_rotation" | transform__0__method_name == "random_rotation") %>% 
  mutate(transform__0__type = if_else(transform__0__method_name == "random_rotation", "no_gaussienization", transform__0__type)) %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  group_by(data__keyword, transform__0__type, model) %>% 
  mutate(mean_test_score = mean(test_score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(x = test_score, y=data__keyword, color = transform__0__type), alpha=0.5, width=0, height=0.1) +
  geom_point(aes(x = mean_test_score, y=data__keyword, fill = transform__0__type), size=10, alpha=0.6, color="black", shape=21) +
  ggtitle("gaussienization comparison (test) (random_rotation)") +
  facet_wrap(~model)


df_rotation %>% 
  pivot_longer(cols=starts_with("train_score"), names_to = "model", values_to="train_score", names_prefix = "train_scores_") %>% 
  filter(transform__0__method_name != "random_rotation") %>% 
  filter(is.na(transform__1__method_name)) %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  group_by(data__keyword, transform__0__type, model) %>% 
  mutate(mean_train_score = mean(train_score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(x = train_score, y=data__keyword, color = transform__0__type), alpha=0.5, width=0, height=0.1) +
  geom_point(aes(x = mean_train_score, y=data__keyword, fill = transform__0__type), size=10, alpha=0.6, color="black", shape=21) +
  ggtitle("gaussienization comparison (train) (no rotation after)") +
  facet_wrap(~model)


df_rotation %>% 
  pivot_longer(cols=starts_with("train_score"), names_to = "model", values_to="train_score", names_prefix = "train_scores_") %>% 
  filter(transform__1__method_name == "random_rotation" | transform__0__method_name == "random_rotation") %>% 
  mutate(transform__0__type = if_else(transform__0__method_name == "random_rotation", "no_gaussienization", transform__0__type)) %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  group_by(data__keyword, transform__0__type, model) %>% 
  mutate(mean_train_score = mean(train_score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(x = train_score, y=data__keyword, color = transform__0__type), alpha=0.5, width=0, height=0.1) +
  geom_point(aes(x = mean_train_score, y=data__keyword, fill = transform__0__type), size=10, alpha=0.6, color="black", shape=21) +
  ggtitle("gaussienization comparison (train) (random_rotation after)") +
  facet_wrap(~model)
############################################

df_rotation_test <- read_csv("results/res_rotation_test.csv") %>% select(-...1, -iter, -id)

df_rotation_test %>% 
  pivot_longer(cols=starts_with("test_score"), names_to = "model", values_to="test_score", names_prefix = "test_scores_") %>% 
  mutate(data__keyword = if_else(data__keyword == "219", "electricity", data__keyword)) %>% 
  mutate(data__keyword = if_else(data__keyword == "3904", "software", data__keyword)) %>% 
  group_by(data__keyword, transform__1__type, model) %>% 
  mutate(mean_test_score = mean(test_score)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_jitter(aes(x = test_score, y=data__keyword, color = transform__1__type), alpha=0.5, width=0, height=0.1) +
  geom_point(aes(x = mean_test_score, y=data__keyword, fill = transform__1__type), size=10, alpha=0.6, color="black", shape=21) +
  ggtitle("gaussienization comparison (test) (random_rotation before)") +
  facet_wrap(~model)

################################################

View(df_rotation)

total_df <- bind_rows(df_1, df_2, df_3, df_4, df_5, df_6) %>% select(-...1)

clean_total_df <- total_df %>% 
  group_by(across(-c(iter,id, starts_with("test_scores"), starts_with("train_scores")))) %>% 
  mutate(n_iter = n()) %>% 
  mutate(across(starts_with("test_scores"), list(mean=mean, sd=sd)), .keep="unused") %>%
  mutate(across(starts_with("train_scores"), list(mean=mean, sd=sd)), .keep="unused") %>% 
  select(-iter, -id, ) %>% 
  distinct() %>% 
  relocate(starts_with("train_scores")) %>% 
  relocate(starts_with("test_scores"))

clean_total_df %>% 
  filter(data__method_name == "open_ml") %>% 
  filter(target__method_name == "no_transform") %>% 
  filter(transform__0__method_name %in% c("add_noise", 'random_rotation')) %>% 
  filter(transform__0__scale == 0.1 | is.na(transform__0__scale)) %>% 
  mutate(data__openml_task_id = as_factor(data__openml_task_id)) %>% 
  #filter(data__max_num_samples == "10000") %>% 
  ggplot() +
  geom_point(aes(y = data__openml_task_id, x=test_scores_mlp_skorch_mean, color=transform__0__method_name))


clean_total_df %>% 
  ungroup() %>% 
  filter(data__method_name == "open_ml") %>% 
  filter(target__method_name == "no_transform") %>% 
  filter(transform__0__method_name %in% c("add_noise", 'random_rotation')) %>% 
  filter(transform__0__scale == 0.1 | is.na(transform__0__scale)) %>% 
  mutate(data__openml_task_id = as_factor(data__openml_task_id)) %>% 
  select(transform__0__method_name, data__openml_task_id, test_scores_gbt_mean, test_scores_rf_mean, test_scores_mlp_skorch_mean) %>% 
  pivot_wider(names_from=transform__0__method_name, values_from=c(test_scores_gbt_mean, test_scores_rf_mean, test_scores_mlp_skorch_mean)) %>% 
  summarise(mlp_rotation = mean(test_scores_mlp_skorch_mean_random_rotation),
            mlp_no_rotation = mean(test_scores_mlp_skorch_mean_add_noise),
            rf_rotation = mean(test_scores_rf_mean_random_rotation),
            rf_no_rotation = mean(test_scores_rf_mean_add_noise),
            gbt_rotation = mean(test_scores_gbt_mean_random_rotation),
            gbt_no_rotation = mean(test_scores_gbt_mean_add_noise),
            mlp_diff = mean(test_scores_mlp_skorch_mean_add_noise - test_scores_mlp_skorch_mean_random_rotation, ),
            rf_diff = mean(test_scores_rf_mean_add_noise - test_scores_rf_mean_random_rotation),
            gbt_diff = mean(test_scores_gbt_mean_add_noise - test_scores_gbt_mean_random_rotation))
  #filter(data__max_num_samples == "10000") %>% 

clean_total_df %>% 
  ungroup() %>% 
  filter(data__method_name == "open_ml") %>% 
  filter(target__method_name == "no_transform") %>% 
  filter(transform__0__method_name %in% c("add_noise", 'random_rotation')) %>% 
  filter(transform__0__scale == 0.1 | is.na(transform__0__scale)) %>% 
  mutate(data__openml_task_id = as_factor(data__openml_task_id)) %>% 
  select(transform__0__method_name, data__openml_task_id, train_scores_gbt_mean, train_scores_rf_mean, train_scores_mlp_skorch_mean) %>% 
  pivot_wider(names_from=transform__0__method_name, values_from=c(train_scores_gbt_mean, train_scores_rf_mean, train_scores_mlp_skorch_mean)) %>% 
  summarise(mlp_rotation = mean(train_scores_mlp_skorch_mean_random_rotation),
            mlp_no_rotation = mean(train_scores_mlp_skorch_mean_add_noise),
            rf_rotation = mean(train_scores_rf_mean_random_rotation),
            rf_no_rotation = mean(train_scores_rf_mean_add_noise),
            gbt_rotation = mean(train_scores_gbt_mean_random_rotation),
            gbt_no_rotation = mean(train_scores_gbt_mean_add_noise),
            mlp_diff = mean(train_scores_mlp_skorch_mean_add_noise - train_scores_mlp_skorch_mean_random_rotation, ),
            rf_diff = mean(train_scores_rf_mean_add_noise - train_scores_rf_mean_random_rotation),
            gbt_diff = mean(train_scores_gbt_mean_add_noise - train_scores_gbt_mean_random_rotation))


View(clean_total_df %>% 
  ungroup() %>% 
  filter(data__method_name == "open_ml") %>% 
  filter(target__method_name == "no_transform") %>% 
  filter(transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>% 
  mutate(transform__1__n_clusters = if_else(is.na(transform__1__n_clusters), 0, transform__1__n_clusters)) %>% 
  mutate(data__openml_task_id = as_factor(data__openml_task_id), transform__1__n_clusters = as_factor(transform__1__n_clusters)) %>% 
  select(transform__1__n_clusters, data__openml_task_id, test_scores_gbt_mean, test_scores_rf_mean, test_scores_mlp_skorch_mean)) %>% 
  pivot_wider(names_from=transform__1__n_clusters, values_from=c(test_scores_gbt_mean, test_scores_rf_mean, test_scores_mlp_skorch_mean))%>% 
  summarise(mlp_rotation = mean(test_scores_mlp_skorch_mean_random_rotation),
            mlp_no_rotation = mean(test_scores_mlp_skorch_mean_add_noise),
            rf_rotation = mean(test_scores_rf_mean_random_rotation),
            rf_no_rotation = mean(test_scores_rf_mean_add_noise),
            gbt_rotation = mean(test_scores_gbt_mean_random_rotation),
            gbt_no_rotation = mean(test_scores_gbt_mean_add_noise),
            mlp_diff = mean(test_scores_mlp_skorch_mean_add_noise - test_scores_mlp_skorch_mean_random_rotation, ),
            rf_diff = mean(test_scores_rf_mean_add_noise - test_scores_rf_mean_random_rotation),
            gbt_diff = mean(test_scores_gbt_mean_add_noise - test_scores_gbt_mean_random_rotation))
#filter(data__max_num_samples == "10000") %>% 


total_df %>% 
  group_by(across(-c(iter,id, starts_with("test_scores"), starts_with("train_scores")))) %>% 
  mutate(n_iter = n()) %>% 
  mutate(across(starts_with("test_scores"), list(mean=mean, sd=sd)), .keep="unused") %>%
  mutate(across(starts_with("train_scores"), list(mean=mean, sd=sd)), .keep="unused") %>% 
  select(-iter, -id, ) %>% 
  distinct() %>% 
  relocate(starts_with("train_scores")) %>% 
  relocate(starts_with("test_scores")) %>% 
  write_csv("total_res_clean.csv")

df_6 %>% 
  filter(transform__0__method_name == "no_transform") %>% 
  filter()

#df <- read_csv("utils/openml_datasets_properties.csv")

#(df %>% 
#  filter(NumberOfClasses == 2) %>% 
#  filter(MinorityClassPercentage > 10) %>% 
#  filter(PercentageOfInstancesWithMissingValues < 50) %>% 
#  filter(PercentageOfNumericFeatures > 50))$id


df_1 %>% 
  group_by(id) %>% 
  mutate(n_iter = n()) %>% 
  mutate(across(starts_with("test_scores"), list(mean=mean, sd=sd)), .keep="unused")%>%
  mutate(across(starts_with("train_scores"), list(mean=mean, sd=sd)), .keep="unused")%>%
  select(-...1, -iter) %>% 
  distinct() %>% 
  relocate(starts_with("train_scores")) %>% 
  relocate(starts_with("test_scores")) %>% 
  relocate(id) %>% 
  write_csv("clean_res_gaussian_23_10.csv")

bind_rows(read_csv("results/clean_res_gaussian_23_10.csv") %>% 
            mutate(id = paste("gaussian", id, sep="")),
          read_csv("results/clean_res_student_18_10.csv") %>% 
            mutate(id = paste("", id, sep="")),
          read_csv("results/clean_res_gaussian_23_10.csv") %>% 
            mutate(id = paste("gaussian", id, sep="")),
          read_csv("results/clean_res_gaussian_23_10.csv") %>% 
            mutate(id = paste("gaussian", id, sep="")),
          read_csv("results/clean_res_gaussian_23_10.csv") %>% 
            mutate(id = paste("gaussian", id, sep="")))

View(df %>% 
  group_by(id) %>% 
  mutate(n_iter = n()) %>% 
  mutate(across(starts_with("test_scores"), list(mean=mean, sd=sd)), .keep="unused")%>%
  mutate(across(starts_with("train_scores"), list(mean=mean, sd=sd)), .keep="unused")%>%
  select(-X1, -iter) %>% 
  distinct() %>% 
  relocate(starts_with("train_scores")) %>% 
  relocate(starts_with("test_scores")) %>% 
  relocate(id) %>% 
  filter(data__method_name == "student"))

clean_df <- df %>% 
  mutate_at(c("test_scores_mean_mlp_skorch", "test_scores_mean_rf", "test_scores_mean_gbt"), function(x) as.integer(100 * x)) %>% 
  select(!ends_with("sd")) %>% 
  select(!starts_with("train")) %>% 
  mutate(mlp_diff = as.integer(test_scores_mean_mlp_skorch - (test_scores_mean_rf + test_scores_mean_gbt) / 2)) %>% 
  relocate(id, mlp_diff, test_scores_mean_mlp_skorch, test_scores_mean_rf, test_scores_mean_gbt)

View(clean_df %>% arrange(mlp_diff))

clean_df %>% 
  filter(target__method_name != "linear") %>% 
  group_by(data__cov_matrix) %>% 
  summarise_at(c("mlp_diff", "test_scores_mean_mlp_skorch", "test_scores_mean_rf", "test_scores_mean_gbt"), mean)

clean_df %>% 
  filter(target__method_name != "linear") %>% 
  filter(data__method_name == "student" ) %>% 
  group_by(data__df) %>% 
  summarise_at(c("mlp_diff", "test_scores_mean_mlp_skorch", "test_scores_mean_rf", "test_scores_mean_gbt"), mean)
  
  aggregate("first")
  mutate(diff = test_scores_mean - mean(test_scores_mean)) %>% 
  mutate(max_diff_id = max(abs(diff))) %>% 
  summarise(first_non_missing = names[which(!is.na(names))[1]])
  
  arrange(max_diff_id))



clean_df <- df %>% 
  select(!ends_with("sd")) %>% 
  select(!starts_with("train")) %>% 
  mutate(test_scores_mean = test_scores_mean*100) %>% 
  pivot_wider(names_from=model, values_from = test_scores_mean, names_prefix = "test_score_") %>% 
  #pivot_wider(names_from=model, values_from = train_scores_mean, names_prefix = "train_score_") %>% 
  group_by(id) %>% 
  mutate(across(starts_with("test_score_"), mean, na.rm=T)) %>% 
  #mutate(across(starts_with("train_score_"), mean, na.rm=T)) %>% 
  select(!X1) %>% 
  distinct() %>% 
  mutate(mlp_diff = as.integer(test_score_mlp_skorch - (test_score_rf + test_score_gbt) / 2)) %>%
  #filter(target__method_name != "linear") %>% 
  arrange(mlp_diff)

