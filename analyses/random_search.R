library(tidyverse)

names <- c("electricity", "california", "covtype", "churn", "credit", "shopping", "nomao", "spam")
n_samples <- c(10000, 10000, 10000, 4074, 10000, 3816,8194,3624)
n_features <- c(7, 8, 10, 6, 10, 10, 46, 57) #TODO

df_datasets <- tibble(names, n_samples, n_features)

colnames(df_datasets)


df <- read_csv("results/random_search/res_random_search_cpu_2.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names")) %>% 
  add_column(dataset="random_search")



df_default <- read_csv("results/random_search/res_random_search_cpu_default.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names")) %>% 
  add_column(dataset="default")



# Comparison of best hp

df %>% 
  group_by(data__keyword, method_name, model_params_str) %>% 
  summarise(mean_score = mean(test_scores, na.rm=T)) %>%
  mutate(score = max(mean_score, na.rm=T)) %>% 
  filter(mean_score ==  score) %>% 
  ggplot() +
  #geom_point(aes(y = data__keyword, x = score, color = method_name))
  geom_point(aes(y = data__keyword, x = mean_score, color = method_name), position=position_dodge(width=0.5)) +
  ggtitle("Best hp comparison")

df %>% 
  group_by(data__keyword, method_name, model_params_str) %>% 
  summarise(mean_score = mean(test_scores, na.rm=T)) %>%
  ungroup() %>% 
  #group_by(data__keyword, method_name) %>% 
  #summarise(score = max(mean_score, na.rm=T)) %>% 
  #filter(mean_score ==  score) %>% 
  ggplot() +
  #geom_point(aes(y = data__keyword, x = score, color = method_name))
  geom_point(aes(y = data__keyword, x = mean_score, color = method_name), size=1, alpha=0.5, position=position_dodge(width=0.5)) +
  ggtitle("All hp comparison")

df %>% 
  bind_rows(df_default) %>% 
  group_by(data__keyword, method_name, model_params_str, dataset) %>% 
  summarise(mean_score = mean(test_scores, na.rm=T)) %>%
  ungroup() %>% 
  group_by(data__keyword, method_name, dataset) %>% 
  mutate(score = max(mean_score, na.rm=T)) %>% 
  filter(mean_score == score) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = score, color = method_name, shape=dataset, group=method_name), position=position_dodge(width=0.5)) +
  ggtitle("Default and best hp")


# Performance vs rank

#mean
df %>% 
  bind_rows(df_default) %>% 
  group_by(data__keyword, iter) %>% 
  mutate(max_score_dataset = max(test_scores, na.rm=T)) %>% 
  mutate(relative_score = test_scores / max_score_dataset) %>% 
  ungroup() %>% 
  group_by(method_name, model_params_str) %>% 
  summarise(mean_score = mean(relative_score, na.rm=T)) %>% #mean on both datasets and iter
  mutate(score_rank = rank(-mean_score)) %>% 
  filter(score_rank < 100) %>% 
  ungroup() %>% 
  ggplot() +
  geom_line(aes(x = score_rank , y = mean_score, color = method_name))

df %>% 
  bind_rows(df_default) %>% 
  group_by(data__keyword, iter) %>% 
  mutate(max_score_dataset = max(test_scores, na.rm=T)) %>% 
  mutate(relative_score = test_scores / max_score_dataset) %>% 
  ungroup() %>% 
  group_by(method_name, model_params_str) %>% 
  summarise(median_score = median(relative_score, na.rm=T)) %>%
  ungroup(model_params_str) %>% 
  mutate(score_rank = rank(-median_score)) %>% 
  filter(score_rank < 100) %>% 
  ggplot() +
  geom_line(aes(x = score_rank , y = median_score, color = method_name))
  
df %>% 
  bind_rows(df_default) %>% 
  group_by(data__keyword, method_name, model_params_str) %>% 
  summarise(mean_score = mean(test_scores, na.rm=T)) %>%
  mutate(score_rank = rank(-mean_score)) %>% 
  filter(score_rank < 100) %>% 
  ggplot() +
  geom_line(aes(x = score_rank , y = mean_score, color = method_name)) +
  facet_wrap(~data__keyword)

# Totally random search
df_2 <- df %>% 
  bind_rows(df_default) %>% 
  group_by(data__keyword, iter) %>% 
  mutate(max_score_dataset = max(test_scores, na.rm=T)) %>% 
  mutate(relative_score = test_scores / max_score_dataset) %>% 
  ungroup() %>% 
  group_by(method_name, model_params_str, data__keyword) %>% 
  summarise(mean_score_iter = mean(relative_score, na.rm=T), mean_time_iter = mean(time_elapsed, na.rm=T))%>% 
  mutate(mean_score_datasets = mean(mean_score_iter, na.rm=T)) %>% 
  ungroup()

new_df <- tibble()

for (i in 1:1000) {  
  new_df <- new_df %>% bind_rows(df_2 %>% mutate(n_dataset = i))   
}

new_df <- new_df %>% 
  add_column(random_number = runif(nrow(.))) %>% 
  group_by(method_name, data__keyword, n_dataset) %>% 
  mutate(random_rank = rank(random_number)) %>% 
  ungroup() %>% 
  select(-random_number)

res <- tibble()

for (i in 1:1000) {
  res <- res %>%
    bind_rows(new_df %>% 
    filter(random_rank <= i) %>% 
    group_by(method_name, data__keyword, n_dataset) %>% 
    mutate(max_score = max(mean_score_iter, na.rm=T), count = n(), cum_time = cumsum(mean_time_iter)) %>% #max on rank
    filter(max_score == mean_score_iter) %>% 
    ungroup() %>% 
    mutate(n_dataset = as_factor(n_dataset)) %>% 
    group_by(method_name, data__keyword) %>% #mean on reruns
    summarise(total_time = mean(cum_time, na.rm=T), mean_max_score = mean(max_score, na.rm=T), min_max_score = min(max_score, na.rm=T), max_max_score = max(max_score, na.rm=T)) %>% 
    mutate(max_rank = i)) #TODO: no mean for time
}

res %>% 
  ggplot() +
  geom_line(aes(x = max_rank, y = mean_max_score, color = method_name)) +
  #geom_line(aes(x = max_rank, y = min_max_score, color = method_name), linetype="dotted") +
  #geom_line(aes(x = max_rank, y = max_max_score, color = method_name), linetype="dotted") +
  facet_wrap(~data__keyword) +
  ylim(0.9, 1)

res %>% 
  group_by(method_name, max_rank) %>% 
  summarise(mean_score_datasets = mean(mean_max_score)) %>% 
  ggplot() +
  geom_line(aes(x = max_rank, y = mean_score_datasets, color = method_name)) +
  ylim(0.9, 1)

res %>% 
  group_by(method_name, max_rank) %>% 
  summarise(mean_score_datasets = mean(mean_max_score), mean_total_time = mean(total_time)) %>% 
  ggplot() +
  geom_line(aes(x = mean_total_time, y = mean_score_datasets, color = method_name)) +
  ylim(0.9, 1)


#Best hp on other dataset
#For each model: take the best hp combination for the mean of all dataset except one and evaluate on loo dataset
# repeat for all datasets

(df_iter %>% 
  filter(method_name == "rf") %>% 
  select(method_name, model_params_str) %>% 
  distinct())
  group_by(data__keyword, method_name) %>% 
  summarise(count = n())

df_rot <- read_csv("results/random_search/res_random_search_rot.csv") %>% 
  left_join(df_datasets, by = c("data__keyword" = "names")) %>% 
  add_column(dataset="random_search")

df_rot <- df_rot %>% 
  filter(data__keyword == "credit") %>% 
  mutate(data__keyword = "credit_rot")

df_iter <- df %>% 
  filter(method_name %in% c("rf", "hgbt")) %>% 
  bind_rows(df_rot) %>% 
  #bind_rows(df_default) %>% 
  group_by(data__keyword, method_name, model_params_str) %>% 
  summarise(mean_score_iter = mean(test_scores, na.rm=T))%>% 
  ungroup() %>% 
  group_by(data__keyword) %>% 
  mutate(relative_score = mean_score_iter / max(mean_score_iter, na.rm=T))

#df_iter %>% 
 # group_by(data__keyword, method_name) %>% 
#mutate(best_score_iter = max(mean_score_iter, na.rm=T)) %>% 
 # filter(best_score_iter == mean_score_iter) %>% 
  #write_csv("hp.csv")

hp_best_data_model <- ( df_iter %>% 
  group_by(data__keyword, method_name) %>% 
  mutate(best_score_iter = max(mean_score_iter, na.rm=T)) %>% 
  filter(best_score_iter == mean_score_iter))$model_params_str

best_hp_df <- df_iter %>% 
  group_by(data__keyword, method_name) %>% 
  mutate(best_score_iter = max(mean_score_iter, na.rm=T)) %>% 
  filter(best_score_iter == mean_score_iter) %>% 
  select(data__keyword, method_name, model_params_str, best_score_iter)

View(df_iter %>% 
  left_join(best_hp_df, by = c("method_name", "model_params_str")) %>% 
  filter(!is.na(best_score_iter))) %>% 
  ggplot() +
  geom_point(aes(x = relative_score, y=data__keyword.y, color=method_name, group=method_name), position = position_dodge(width=0.5))
  
  


df_iter %>% 
  filter(model_params_str %in% hp_best_data_model) %>% 
  ggplot() +
  geom_point(aes(y=data__keyword, x = mean_score_iter, color = method_name, group=method_name), position=position_dodge(width=0.5)) +
  ggtitle("Best hp combination for each dataset, evaluated on all other datasets")

#for each dataset, take best hp and evaluate it on all other datasets
df_iter %>% 
  filter(model_params_str %in% hp_best_data_model) %>% 

datasets <- (df %>% select(data__keyword) %>% distinct())$data__keyword

df_loo <- tibble()

for (dataset in datasets){
  hp <- (df_iter %>% 
           filter(data__keyword != dataset) %>% 
           group_by(method_name, model_params_str) %>% 
           summarise(mean_score_datasets = mean(mean_score_iter)) %>% 
           mutate(best_score_datasets = max(mean_score_datasets, na.rm=T)) %>% 
           filter(best_score_datasets == mean_score_datasets))$model_params_str
  df_loo <- df_loo %>% 
            bind_rows(df_iter %>% 
                        filter(data__keyword == dataset) %>% 
                        filter(model_params_str %in% hp))
}

df_best <- df_iter %>% 
  group_by(method_name, data__keyword) %>% 
  summarise(mean_score_iter = max(mean_score_iter, na.rm=T)) %>% 
  mutate(dataset = "best")

df_loo <- df_loo %>% mutate(dataset = "loo")

df_loo %>% 
  bind_rows(df_best) %>% 
  ggplot() +
  geom_point(aes(y=data__keyword, x = mean_score_iter, color = method_name, group=method_name, shape=dataset), position=position_dodge(width=0.5)) +
  ggtitle("Best hp combination for each dataset, evaluated on all other datasets")


# Optimal hp search path
#TODO relative score
compute_best_hp_1 <- function(data) {
  to_join <- data %>%
    group_by(method_name, data__keyword) %>% 
    mutate(best_score_datasets = max(mean_score_datasets, na.rm=T)) %>% 
    filter(best_score_datasets == mean_score_datasets) %>% 
    mutate(best_score_iter = mean_score_iter, .keep="unused") %>% 
    select(-mean_score_datasets, -model_params_str)
  data %>% 
    left_join(to_join, by=c("method_name", "data__keyword"))
}

compute_good_hps <- function(data, n) {
  to_join <- data %>%
    group_by(method_name, data__keyword) %>% 
    mutate(rank_score_dataset = rank(-mean_score_datasets, ties.method="min")) %>% 
    filter(rank_score_dataset == n) %>% 
    mutate(best_score_datasets = mean_score_datasets) %>% 
    mutate(best_score_iter = mean_score_iter, .keep="unused") %>% 
    select(-mean_score_datasets, -model_params_str)
  data %>% 
    left_join(to_join, by=c("method_name", "data__keyword")) %>% 
    select(method_name, data__keyword, best_score_iter) %>% 
    distinct()
}



#TODO relative score

df_1 <- df %>% 
  bind_rows(df_default) %>% 
  group_by(data__keyword, iter) %>% 
  mutate(max_score_dataset = max(test_scores, na.rm=T)) %>% 
  mutate(relative_score = test_scores / max_score_dataset) %>% 
  ungroup() %>% 
  group_by(method_name, model_params_str, data__keyword) %>% 
  summarise(mean_score_iter = mean(relative_score, na.rm=T)) %>% 
  ungroup() %>% 
  group_by(method_name, model_params_str) %>% 
  mutate(mean_score_datasets = mean(mean_score_iter, na.rm=T)) %>% 
  ungroup() 

df_good_hp_list <- map(1:10, ~compute_good_hps(df_1, .))  

df_1 <- df_1 %>% 
  compute_best_hp_1()

df_1_final <- df_1 %>% 
  select(method_name, best_score_datasets) %>% 
  distinct()

df_1_to_use <- df_1 %>% 
  select(method_name, data__keyword, best_score_iter, best_score_datasets) %>% 
  distinct()


remove_useless_hp <- function(data_original, data_previous_list){
  new_df <- data_original
  for (data_previous in data_previous_list) {
    admissible_hp <- (new_df %>% 
                        select(-best_score_iter) %>% 
                        left_join(data_previous %>% select(method_name, data__keyword, best_score_iter), by=c("method_name", "data__keyword")) %>% 
                        mutate(diff = mean_score_iter - best_score_iter) %>% 
                        filter(diff >= 1e-5 | is.na(best_score_iter)) %>% 
                        select(model_params_str) %>% 
                        distinct())$model_params_str
    print("num admissible hp")
    print(length(admissible_hp))
    new_df <- new_df %>% 
      filter(model_params_str %in% admissible_hp)
  }
  return(new_df)
}

compute_temp <- function(new_df, n){
  temp <- new_df %>% 
    select(-best_score_iter) %>% 
    group_by(method_name) %>% 
    mutate(count = n()) %>% 
    filter(count >= n * 10) %>% 
    ungroup() %>% 
    select(-count)
  
  for (i in 1:n) {  
    temp <- temp %>% 
      mutate(!!sym(paste("model_params_str_", i, sep="")) := model_params_str) 
  }
  var_list <- map(1:n, ~paste("model_params_str_", ., sep=""))
  temp <-  temp %>% 
    group_by(method_name, data__keyword) %>%
    complete(!!!syms(var_list)) %>% 
    select(-model_params_str, -mean_score_iter)
  
  for (i in 2:n) {  
    temp <- temp %>% filter(!!sym(paste("model_params_str_", i, sep="")) > !!sym(paste("model_params_str_", i-1, sep="")))
  }
  
  print(temp %>% group_by(method_name) %>% summarise(count = n()))
  
  print(nrow(temp))
  
  for (i in 1:n) {  
    var_name_score <- paste("mean_score_iter_", i, sep="")
    var_name_params <- paste("model_params_str_", i, sep="")
    join_cols <- c("method_name", "data__keyword", "model_params_str")
    names(join_cols) <- c("method_name", "data__keyword", var_name_params)
    temp <- temp %>% 
      left_join(new_df %>% mutate(!!sym(var_name_score) := mean_score_iter, .keep="unused") %>% select(-mean_score_datasets, -best_score_datasets, -best_score_iter),  by=join_cols)
  }
  temp
}

compute_best_hp_n <- function(temp, rank_best){
  new_df_to_use <- temp %>% 
    pivot_longer(starts_with("mean_score_iter"), names_prefix="mean_score_iter_", values_to = "mean_score_iter") %>% 
    group_by(across(c(-name, -mean_score_iter))) %>% 
    summarise(best_score_iter = max(mean_score_iter)) %>% 
    ungroup() %>% 
    group_by(across(c(-data__keyword, -best_score_iter))) %>% 
    mutate(mean_score_datasets = mean(best_score_iter)) %>% 
    ungroup() %>% 
    group_by(method_name, data__keyword) %>% 
    mutate(rank_score_datasets = rank(-mean_score_datasets, ties.method="first")) %>% 
    filter(rank_score_datasets == rank_best) %>%
    mutate(best_score_datasets = mean_score_datasets) %>% 
    select(method_name, data__keyword, best_score_iter, best_score_datasets) %>% 
    ungroup()
  
  new_df_to_use
}

compute_hp_best_dataset <- function(temp, rank_best, dataset){
  temp %>% 
    pivot_longer(starts_with("mean_score_iter"), names_prefix="mean_score_iter_", values_to = "mean_score_iter") %>% 
    group_by(across(c(-name, -mean_score_iter))) %>% 
    summarise(best_score_iter = max(mean_score_iter)) %>%
    ungroup(data__keyword) %>%
    mutate(best_score_datasets = sum((data__keyword == dataset) * best_score_iter, na.rm=T)) %>% 
    ungroup() %>% 
    group_by(method_name, data__keyword) %>% 
    mutate(rank_score_datasets = rank(-best_score_datasets, ties.method="first")) %>% 
    filter(rank_score_datasets == rank_best) %>%
    select(method_name, data__keyword, best_score_iter) %>% 
    ungroup()
}

datasets <- (df_1 %>% group_by(data__keyword) %>% summarise(count = n()))$data__keyword

new_df <- df_1

final_df_list <- list()

df_list <- df_good_hp_list

new_df <- remove_useless_hp(new_df, df_list)

for (i in 2:10){
  print(i)
  temp <- compute_temp(new_df, i)
  df_list <- map(1:10, ~compute_best_hp_n(temp, .))
  df_list_2 <- map(datasets, ~compute_hp_best_dataset(temp, 1, .))
  #df_list_2 <- list(compute_hp_best_dataset(temp, 1, "covtype"))
  new_df <- remove_useless_hp(new_df, df_list)
  new_df <- remove_useless_hp(new_df, df_list_2)
  final_df_list <- append(final_df_list, list(df_list[[1]]))
}

final_df <- df_1_to_use %>% mutate(set_size = 1) 

for (i in 2:length(final_df_list)){
  final_df <- final_df %>% bind_rows(final_df_list[[i]] %>% mutate(set_size=i))
}


final_df %>%
 complete(method_name, data__keyword, set_size) %>% 
 group_by(method_name, data__keyword) %>% 
  mutate(best_score_datasets = if_else(is.na(best_score_datasets), max(best_score_datasets, na.rm=T), best_score_datasets),
         best_score_iter = if_else(is.na(best_score_iter), max(best_score_iter, na.rm=T), best_score_iter)) %>%  #if all hp have been eliminated before the end, we take the last (=max) for all subsequent set_sizes
ungroup() %>% 
  select(method_name, best_score_datasets, set_size) %>% 
  distinct() %>% 
  ggplot() +
  geom_line(aes(x = set_size, y=best_score_datasets, color=method_name))



df_1_to_use %>% 
  mutate(set_size = 1) %>% 
  bind_rows(df_2_list[[1]] %>%
              select(method_name, data__keyword, best_score_iter, best_score_datasets) %>% 
              mutate(set_size=2)) %>% 
  bind_rows(df_3_list[[1]]  %>%
              select(method_name, data__keyword, best_score_iter, best_score_datasets)%>% 
              mutate(set_size=3)) %>% 
  bind_rows(df_4_list[[1]]  %>%
              select(method_name, data__keyword, best_score_iter, best_score_datasets) %>% 
              mutate(set_size=4)) %>% 
  bind_rows(df_5_list[[1]]  %>%
              select(method_name, data__keyword, best_score_iter, best_score_datasets) %>% 
              mutate(set_size=5)) %>% 
  bind_rows(df_6_list[[1]]  %>%
              select(method_name, data__keyword, best_score_iter, best_score_datasets) %>% 
              mutate(set_size=6)) %>% 
  bind_rows(df_7_list[[1]]  %>%
              select(method_name, data__keyword, best_score_iter, best_score_datasets) %>% 
              mutate(set_size=7)) %>%
  complete(method_name, data__keyword, set_size) %>% 
  group_by(method_name, data__keyword) %>% 
  mutate(best_score_datasets = if_else(is.na(best_score_datasets), max(best_score_datasets, na.rm=T), best_score_datasets),
         best_score_iter = if_else(is.na(best_score_iter), max(best_score_iter, na.rm=T), best_score_iter)) %>%  #if all hp have been eliminated before the end, we take the last (=max) for all subsequent set_sizes
  ungroup() %>%  
  ggplot() +
  geom_line(aes(x = set_size, y=best_score_iter, color=method_name))+
  facet_wrap(~data__keyword)



 # make a table with id --> model_params_str
# for n, make a combinaison


