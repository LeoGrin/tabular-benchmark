source("analyses/random_search_utils.R")

library(RColorBrewer)
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
myColors <- gg_color_hue(6)
names(myColors) <- c("GradientBoostingTree", "RandomForest", "XGBoost", "FT Transformer", "Resnet", "MLP")
colScale <- list(scale_colour_manual(name = "grp",values = myColors),
                 scale_fill_manual(name = "grp",values = myColors))


df <- read_csv("results/sweeps/sweeps_classif/random_rotation.csv") %>% 
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, transform__2__deactivated, transform__0__max_rel_decrease) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/random_rotation_2.csv") %>% 
              select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, transform__2__deactivated, transform__0__max_rel_decrease)) %>%
  mutate(transform__2__deactivated = F) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/random_rotation_3.csv") %>% 
              select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, transform__2__deactivated, transform__0__max_rel_decrease)) %>% 
  mutate(hp = "random") %>% 
  filter(!is.na(mean_test_score))
  #filter(transform__0__max_rel_decrease == 0.05)
    

res_datasets <- tibble()

for (max_rel_decrease in c(0, 0.01, 0.02, 0.05)) {
  res_datasets <- res_datasets %>% 
    bind_rows(
      df %>% 
        filter(transform__0__max_rel_decrease == max_rel_decrease) %>% 
        filter(!is.na(mean_time)) %>% 
        rename() %>% 
        normalize(variable=transform__2__deactivated, normalization_type = "quantile", quantile = 0.05) %>% 
        random_search(variable =transform__2__deactivated, n_shuffles=30, default_first = T) %>% 
        mutate(transform__0__max_rel_decrease = max_rel_decrease)
    )
}


res_datasets %>% 
  filter(random_rank == 20) %>%
  #filter(model_name == "GradientBoostingTree") %>%
  filter(data__keyword != "poker") %>% #, data__keyword %in% c("electricity", "california")) %>% 
  #filter(data__keyword == "wine") %>% 
  group_by(model_name, n_dataset, transform__2__deactivated, transform__0__max_rel_decrease) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  mutate(transform__2__deactivated = as_factor(transform__2__deactivated),
         transform__0__max_rel_decrease = as_factor(transform__0__max_rel_decrease)) %>% 
  #mutate(transform__2__cov_mult = as_factor(100 * transform__2__cov_mult)) %>% 
  #mutate(transform__2__cov_mult = paste0(transform__2__cov_mult, "%")) %>% 
  #mutate(transform__2__cov_mult = as.character(transform__2__cov_mult)) %>% 
  #mutate(model_name = fct_relevel(as_factor(model_name), "GradientBoostingTree", "RandomForest", "FT_Transformer", "Resnet"))%>%
  mutate(model_name = factor(model_name, levels = c("GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"))) %>% 
  mutate(random_rotation = if_else(transform__2__deactivated == 0, T, F)) %>% 
  ggplot() +
  geom_boxplot(aes(x = random_rotation, y=mean_test_score, fill=model_name))  +
  facet_wrap(~transform__0__max_rel_decrease) +
  #geom_jitter(aes(x = transform__1__cov_mult, y=mean_test_score, color=model_name), height = 0, width=0.1)+
  #geom_dl(aes(label = model_name, x=transform__2__cov_mult, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  xlab("Random rotation") +
  ylab("Normalized test score of best model \n (on valid set) after 20 random search iterations") +
  theme_minimal(base_size=22) +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  colScale


res_datasets %>% 
  filter(random_rank == 20) %>%
  #filter(model_name == "GradientBoostingTree") %>%
  filter(data__keyword != "poker") %>% #, data__keyword %in% c("electricity", "california")) %>% 
  #filter(data__keyword == "wine") %>% 
  group_by(model_name, n_dataset, transform__2__deactivated, transform__0__max_rel_decrease) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ungroup() %>% 
  group_by(model_name, transform__2__deactivated, transform__0__max_rel_decrease) %>% 
  summarise(high_test_score = max(mean_test_score, na.rm=T),
            low_test_score = min(mean_test_score, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ungroup() %>% 
  mutate(transform__2__deactivated = as_factor(transform__2__deactivated)) %>% 
  #mutate(transform__2__cov_mult = as_factor(100 * transform__2__cov_mult)) %>% 
  #mutate(transform__2__cov_mult = paste0(transform__2__cov_mult, "%")) %>% 
  #mutate(transform__2__cov_mult = as.character(transform__2__cov_mult)) %>% 
  #mutate(model_name = fct_relevel(as_factor(model_name), "GradientBoostingTree", "RandomForest", "FT_Transformer", "Resnet"))%>%
  mutate(model_name = factor(model_name, levels = c("GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"))) %>% 
  mutate(random_rotation = if_else(transform__2__deactivated == 0, T, F)) %>% 
  ggplot() +
  geom_ribbon(aes(x = transform__0__max_rel_decrease, 
                  ymin = low_test_score,
                  ymax = high_test_score,
                  fill = model_name,
                  linetype=transform__2__deactivated),
                  alpha=0.5) + 
  geom_line(aes(x = transform__0__max_rel_decrease, y=mean_test_score,linetype=transform__2__deactivated, color=model_name))  +
  #facet_wrap(~transform__2__deactivated) +
  #geom_jitter(aes(x = transform__1__cov_mult, y=mean_test_score, color=model_name), height = 0, width=0.1)+
  #geom_dl(aes(label = model_name, x=transform__2__cov_mult, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  xlab("Random rotation") +
  ylab("Normalized test score of best model \n (on valid set) after 20 random search iterations") +
  theme_minimal(base_size=22) +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  colScale
