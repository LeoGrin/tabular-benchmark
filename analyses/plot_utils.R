source("analyses/random_search_utils.R")
library(directlabels)
library(shadowtext)
library(ggrepel)
library(glue)

checks <- function(df) {
  print(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()) %>% summarise(count = sum(count > 100)))
  nrow(df %>% filter(n_train > 10000))
}

plot_results_per_dataset <- function(df, score="accuracy", truncate_scores=F, normalize=F, quantile=0.1, legend_size=22, equalize_n_iteration=T, default_colscale=T) {
  res_datasets <- 
    df %>% 
    filter(!is.na(mean_time)) %>% 
    mutate_if(is.numeric, list(~na_if(., Inf))) %>% 
    mutate_if(is.numeric, list(~na_if(., -Inf)))
  
  if (normalize)
    res_datasets <- res_datasets %>% normalize_no_variable(normalization_type = "quantile", quantile = quantile)
  if (truncate_scores)
    res_datasets <- res_datasets %>% mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score)) 
  
  
  #normalize_no_variable(normalization_type = "quantile", quantile = 0.1) %>% 
  res_datasets <- res_datasets %>% random_search_no_variable(n_shuffles=15, default_first = T, equalize_n_iteration = equalize_n_iteration) 
  
  if (equalize_n_iteration)
    res_datasets <- res_datasets %>% 
    # Restrict to the min of iter for all model for each datasets
    left_join(df %>% group_by(data__keyword, model_name) %>% summarise(count = n()) %>% summarise(min_count = min(count)), by="data__keyword") %>% 
    filter(random_rank < min_count) %>% 
    ungroup()
  

  res_datasets <- res_datasets %>% 
    # Aggregate across shuffles
    group_by(random_rank, data__keyword, model_name) %>% 
    summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
              max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
              mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
    ungroup() %>% 
    filter(min_test_score != Inf, max_test_score != -Inf) %>% 
    # prevent the ribbon from shifting the scale
    group_by(data__keyword) %>% 
    mutate(plot_min = min(mean_test_score, na.rm=T)) %>% # lowest line
    ungroup() %>% 
    rowwise() %>%
    mutate(min_test_score = max(min_test_score, plot_min)) %>% 
    ungroup()
  
  
  
  
  plot <- res_datasets %>% 
    ggplot() +
    geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets %>% filter(random_rank == 0) %>% group_by(model_name, data__keyword) %>% summarise(mean_test_score = mean(mean_test_score, na.rm=T)), linetype="dotted", size=1.5, alpha=0.7) +
    geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
    geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
    #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
    #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
    facet_wrap(~data__keyword, scales="free") +
    #geom_blank(aes(y = y_min)) +
    coord_cartesian(ylim=c(res_datasets$y_min, res_datasets$y_max)) + 
    scale_x_log10() +#limits=c(1, 1000)) +
    xlab("Number of random search iterations") +
    ylab(str_interp(glue("Test {score} of best model \n (on valid set) up to this iteration"))) +
    theme_minimal(base_size=22) +
    theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=legend_size))
  
  if (default_colscale)
    plot + colScale
  else
    plot
}

plot_aggregated_results <- function(df, score="accuracy", quantile=0.1, truncate_scores=F, y_inf = 0.5, equalize_n_iteration=T, default_colscale=T, text_size=6.5, theme_size=22) {
  
  res_datasets <- 
    df %>% 
    filter(!is.na(mean_time)) %>% 
    normalize_no_variable(normalization_type = "quantile", quantile =quantile)
  
  if (truncate_scores)
    res_datasets <- res_datasets %>% mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score)) 
  
  
  res_datasets <- res_datasets %>% 
    random_search_no_variable(n_shuffles=15, default_first = T, equalize_n_iteration=equalize_n_iteration) 
  
  
  res_datasets_ <- res_datasets %>% 
    group_by(random_rank, model_name, n_dataset) %>% 
    summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% #summarise on datasets
    summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
              max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
              mean_test_score = mean(mean_test_score, na.rm=T))
  
  
  plot <- res_datasets_ %>% 
    ggplot() +
    geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
    geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
    geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
    #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
    #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
    geom_text_repel(aes(label=model_name, 
                        color = model_name,
                        x = random_rank,
                        y =  mean_test_score),
                    data= (res_datasets_ %>% 
                             filter(random_rank == 1)),
                    bg.color='white', size = text_size, bg.r=0.15,
                    nudge_y = 0., nudge_x = 0.3, min.segment.length=100)+
    coord_cartesian(ylim=c(y_inf, 1.0)) + 
    #scale_x_log10(limits=c(1, 700)) +
    scale_x_log10() +
    xlab("Number of random search iterations") +
    ylab(glue("Normalized test {score} of best  \n model (on valid set) up to this iteration")) +
    theme_minimal(base_size=theme_size) +
    theme(legend.position="none")
  
  if (default_colscale)
    plot + colScale
  else
    plot
}

plot_aggregated_results_time <- function(df, score="accuracy", quantile=0.1, truncate_scores=F, y_inf = 0.5) {
  
  res_datasets <- 
    df %>% 
    filter(!is.na(mean_time)) %>% 
    normalize_no_variable(normalization_type = "quantile", quantile =quantile)
  
  if (truncate_scores)
    res_datasets <- res_datasets %>% mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score)) 
  
  
  res_datasets <- res_datasets %>% 
    random_search_no_variable_time(n_shuffles=15, default_first = T) 

  
  
  res_time_ <- res_datasets %>% #select(model_name, data__keyword, log_cum_time_factor,log_cum_time, mean_val_score,best_val_score_so_far, mean_test_score) %>% 
    group_by(model_name, cum_time_factor, n_dataset) %>% 
    # mean on dataset (should be nan if a dataset is missing)
    summarise(mean_test_score = mean(mean_test_score), cum_time = median(cum_time, na.rm=T)) %>% 
    # mean on shuffles
    summarise(max_test_score = max(mean_test_score),
              min_test_score = min(mean_test_score),
              mean_test_score = mean(mean_test_score),
              cum_time = mean(cum_time))
  
  
  res_time_ %>% 
    filter(!is.na(mean_test_score)) %>% 
    ggplot() +
    #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
    geom_line(aes(x = cum_time, y = mean_test_score, color = model_name), size=2) +
    geom_ribbon(aes(x=cum_time, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
    #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
    #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
    geom_text_repel(aes(label=model_name, 
                        color = model_name,
                        x = cum_time,
                        y =  mean_test_score),
                    data= (res_time_ %>% 
                             filter(!is.na(mean_test_score)) %>% 
                             group_by(model_name) %>% 
                             filter(cum_time_factor == min(cum_time_factor, na.rm=T))),
                    bg.color='white', size = 6.5, bg.r=0.15,
                    nudge_y = 0., nudge_x = 0.3, min.segment.length=100)+
    coord_cartesian(ylim=c(y_inf, 1.0)) + 
    #scale_x_log10(limits=c(1, 700)) +
    scale_x_log10() +
    xlab("Random search time (seconds)") +
    ylab(glue("Normalized test {score} of best  \n model (on valid set) up to this iteration")) +
    theme_minimal(base_size=22) +
    theme(legend.position="none") +
    colScale
}
