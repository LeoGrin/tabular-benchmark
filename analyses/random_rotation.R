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


# df <- read_csv("results/sweeps/sweeps_classif/random_rotation.csv") %>% 
#   select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, transform__2__deactivated, transform__0__max_rel_decrease) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/random_rotation_2.csv") %>% 
#               select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, transform__2__deactivated, transform__0__max_rel_decrease)) %>%
#   mutate(transform__2__deactivated = F) %>% 
#   bind_rows(read_csv("results/sweeps/sweeps_classif/random_rotation_3.csv") %>% 
#               select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, transform__2__deactivated, transform__0__max_rel_decrease)) %>% 
#   mutate(hp = "random") %>% 
#   filter(!is.na(mean_test_score))
#   #filter(transform__0__max_rel_decrease == 0.05)

df <- read_csv("results/sweeps/sweeps_classif/random_rotation/ft_random_rotation_2.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/random_rotation/resnet_random_rotation_2.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/random_rotation/rf_random_rotation_2.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/random_rotation/gbt_random_rotation_2.csv")) %>% 
  mutate(hp = "random") %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score))

res_datasets <- tibble()


for (num_features_to_remove in c(0, 0.5)) {
  res_datasets <- res_datasets %>%
    bind_rows(
      df %>%
        filter(transform__0__num_features_to_remove == num_features_to_remove) %>%
        filter(!is.na(mean_time)) %>%
        rename() %>%
        normalize(variable=transform__2__deactivated, normalization_type = "quantile", quantile = 0.1) %>%
        random_search(variable =transform__2__deactivated, n_shuffles=30, default_first = T) %>%
        mutate(transform__0__num_features_to_remove = num_features_to_remove)
    )
}


res_datasets_ <-res_datasets  %>% 
  filter(random_rank == 20) %>%
  filter(data__keyword != "poker") %>% #, data__keyword %in% c("electricity", "california")) %>% 
  group_by(model_name, n_dataset, transform__2__deactivated, transform__0__num_features_to_remove) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  mutate(transform__2__deactivated = as_factor(transform__2__deactivated),
         #transform__0__max_rel_decrease = as_factor(transform__0__max_rel_decrease)) %>%
         transform__0__num_features_to_remove = as_factor(transform__0__num_features_to_remove)) %>% 
  mutate(model_name = factor(model_name, levels = c("GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"))) %>% 
  mutate(random_rotation = if_else(transform__2__deactivated == T, "No Rotation", "Rotation")) %>% 
  filter(transform__0__num_features_to_remove == 0) #CHANGE HERE TO REMOVE FEATURES

res_datasets_2 <- res_datasets_ %>% mutate(random_rotation_num = 1 - as.numeric(transform__2__deactivated)) %>% #to invert order
  group_by(model_name, n_dataset, data__keyword) %>% 
  mutate(r = rank(random_rotation_num), background_color = if_else(r %%2 == 0, 0.01, 0.0)) %>% 
  ungroup()

small_df <- res_datasets_2 %>% filter(n_dataset==1, model_name == "FT Transformer")

  
res_datasets_2 %>% 
  droplevels() %>% 
  #mutate(model_name = fct_drop(model_name)) %>% 
  ggplot() +
  geom_rect(data = small_df, aes(ymin=-Inf, ymax=Inf, xmin=random_rotation_num - 0.55, xmax=random_rotation_num + 0.55, alpha=background_color))+
  geom_boxplot(aes(x = random_rotation_num, y=mean_test_score, color=model_name, group = interaction(model_name, random_rotation_num)))  +
  #facet_wrap(~transform__0__num_features_to_remove) +
  # geom_text_repel(aes(label=model_name,
  #                     color = model_name,
  #                     x = random_rotation_num,
  #                     y =  mean_test_score),
  #                 data= (res_datasets_2 %>%
  #                          filter(random_rotation == "No Rotation", n_dataset==1)),
  #                 bg.color='white', size =6.5,bg.r=0.15,
  #                 nudge_y = 0.01, nudge_x = 0.6
  #                 , min.segment.length=100)+
  #geom_jitter(aes(x = transform__1__cov_mult, y=mean_test_score, color=model_name), height = 0, width=0.1)+
  #geom_dl(aes(label = model_name, x=transform__2__cov_mult, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  facet_wrap(~data__keyword, scales="free") +
  scale_alpha(range = c(0, 0.13)) +  
  scale_x_continuous(breaks=small_df$random_rotation_num, labels=small_df$random_rotation) + 
  #ylim(c(0.6, 0.97)) +
  xlab("") +
  ylab("Normalized test score of best \n model (on valid set) after 20 \n random search iterations") +
  theme_minimal(base_size=22) +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  #theme(legend.position="none") +
  #scale_x_discrete(guide = guide_axis(check.overlap = TRUE))+
  colScale +
  guides(alpha="none")

ggsave("analyses/plots/random_rotation_datasets.jpg", width=15, height=9)




res_datasets_ %>% 
  ggplot() +
  geom_boxplot(aes(x = model_name, y=mean_test_score, fill=model_name))  +
  #facet_wrap(~transform__0__num_features_to_remove) +
  geom_text_repel(aes(label=model_name,
                      color = model_name,
                      x = random_rotation,
                      y =  mean_test_score),
                  data= (res_datasets_ %>%
                           filter(random_rotation == "No Rotation", n_dataset==1)),
                  bg.color='white', size =6.5,bg.r=0.15,
                  nudge_y = 0.01, nudge_x = 0.6
                  , min.segment.length=100)+
  #geom_jitter(aes(x = transform__1__cov_mult, y=mean_test_score, color=model_name), height = 0, width=0.1)+
  #geom_dl(aes(label = model_name, x=transform__2__cov_mult, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  facet_wrap(~random_rotation, scales="free") +
  ylim(c(0.6, 0.97)) +
  xlab("") +
  ylab("Normalized test score of best \n model (on valid set) after 20 \n random search iterations") +
  theme_minimal(base_size=22) +
  #theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  theme(legend.position="none") +
  colScale

ggsave("analyses/plots/random_rotation_features_removed.jpg", width=15, height=9)






################
# dataset by dataset




res_datasets <- tibble()


for (num_features_to_remove in c(0, 0.5)) {
  res_datasets <- res_datasets %>%
    bind_rows(
      df %>%
        filter(transform__0__num_features_to_remove == num_features_to_remove) %>%
        filter(!is.na(mean_time)) %>%
        rename() %>%
        #normalize(variable=transform__2__deactivated, normalization_type = "quantile", quantile = 0.1) %>%
        random_search(variable =transform__2__deactivated, n_shuffles=30, default_first = T) %>%
        mutate(transform__0__num_features_to_remove = num_features_to_remove)
    )
}


res_datasets_ <-res_datasets  %>% 
  filter(random_rank == 20) %>%
  filter(data__keyword != "poker") %>% #, data__keyword %in% c("electricity", "california")) %>% 
  #group_by(model_name, n_dataset, transform__2__deactivated, transform__0__num_features_to_remove) %>% 
  #summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  mutate(transform__2__deactivated = as_factor(transform__2__deactivated),
         #transform__0__max_rel_decrease = as_factor(transform__0__max_rel_decrease)) %>%
         transform__0__num_features_to_remove = as_factor(transform__0__num_features_to_remove)) %>% 
  mutate(model_name = factor(model_name, levels = c("GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"))) %>% 
  mutate(random_rotation = if_else(transform__2__deactivated == T, "No Rotation", "Rotation")) %>% 
  filter(transform__0__num_features_to_remove == 0) #CHANGE HERE TO REMOVE FEATURES

res_datasets_2 <- res_datasets_ %>% mutate(random_rotation_num = 1 - as.numeric(transform__2__deactivated)) %>% #to invert order
  group_by(model_name, n_dataset, data__keyword) %>% 
  mutate(r = rank(random_rotation_num), background_color = if_else(r %%2 == 0, 0.01, 0.0)) %>% 
  ungroup()

small_df <- res_datasets_2 %>% filter(n_dataset==1, model_name == "FT Transformer")

small_df_2 <- res_datasets_2 %>% filter(n_dataset==1, model_name == "FT Transformer", data__keyword=="electricity") #prevent label aliasing



res_datasets_2 %>% 
  droplevels() %>% 
  #mutate(model_name = fct_drop(model_name)) %>% 
  ggplot() +
  geom_rect(data = small_df, aes(ymin=-Inf, ymax=Inf, xmin=random_rotation_num - 0.55, xmax=random_rotation_num + 0.55, alpha=background_color))+
  geom_boxplot(aes(x = random_rotation_num, y=mean_test_score, color=model_name, group = interaction(model_name, random_rotation_num)))  +
  #facet_wrap(~transform__0__num_features_to_remove) +
  # geom_text_repel(aes(label=model_name,
  #                     color = model_name,
  #                     x = random_rotation_num,
  #                     y =  mean_test_score),
  #                 data= (res_datasets_2 %>%
  #                          filter(random_rotation == "No Rotation", n_dataset==1)),
  #                 bg.color='white', size =6.5,bg.r=0.15,
  #                 nudge_y = 0.01, nudge_x = 0.6
  #                 , min.segment.length=100)+
  #geom_jitter(aes(x = transform__1__cov_mult, y=mean_test_score, color=model_name), height = 0, width=0.1)+
#geom_dl(aes(label = model_name, x=transform__2__cov_mult, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
facet_wrap(~data__keyword, scales="free") +
  scale_alpha(range = c(0, 0.13)) +  
  scale_x_continuous(breaks=small_df_2$random_rotation_num, labels=small_df_2$random_rotation) + 
  #ylim(c(0.6, 0.97)) +
  xlab("") +
  ylab("Test accuracy of best model \n (on valid set) after 20 \n random search iterations") +
  theme_minimal(base_size=22) +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  #theme(legend.position="none") +
  #scale_x_discrete(guide = guide_axis(check.overlap = TRUE))+
  colScale +
  guides(alpha="none")

ggsave("analyses/plots/random_rotation_datasets.jpg", width=15, height=9)


# 
# 
# res_datasets %>% 
#   filter(random_rank == 20) %>%
#   #filter(model_name == "GradientBoostingTree") %>%
#   filter(data__keyword != "poker") %>% #, data__keyword %in% c("electricity", "california")) %>% 
#   #filter(data__keyword == "wine") %>% 
#   group_by(model_name, n_dataset, transform__2__deactivated, transform__0__max_rel_decrease) %>% 
#   summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
#   ungroup() %>% 
#   group_by(model_name, transform__2__deactivated, transform__0__max_rel_decrease) %>% 
#   summarise(high_test_score = max(mean_test_score, na.rm=T),
#             low_test_score = min(mean_test_score, na.rm=T),
#             mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
#   ungroup() %>% 
#   mutate(transform__2__deactivated = as_factor(transform__2__deactivated)) %>% 
#   #mutate(transform__2__cov_mult = as_factor(100 * transform__2__cov_mult)) %>% 
#   #mutate(transform__2__cov_mult = paste0(transform__2__cov_mult, "%")) %>% 
#   #mutate(transform__2__cov_mult = as.character(transform__2__cov_mult)) %>% 
#   #mutate(model_name = fct_relevel(as_factor(model_name), "GradientBoostingTree", "RandomForest", "FT_Transformer", "Resnet"))%>%
#   mutate(model_name = factor(model_name, levels = c("GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"))) %>% 
#   mutate(random_rotation = if_else(transform__2__deactivated == 0, T, F)) %>% 
#   ggplot() +
#   geom_ribbon(aes(x = transform__0__max_rel_decrease, 
#                   ymin = low_test_score,
#                   ymax = high_test_score,
#                   fill = model_name,
#                   linetype=transform__2__deactivated),
#                   alpha=0.5) + 
#   geom_line(aes(x = transform__0__max_rel_decrease, y=mean_test_score,linetype=transform__2__deactivated, color=model_name))  +
#   #facet_wrap(~transform__2__deactivated) +
#   #geom_jitter(aes(x = transform__1__cov_mult, y=mean_test_score, color=model_name), height = 0, width=0.1)+
#   #geom_dl(aes(label = model_name, x=transform__2__cov_mult, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
#   xlab("Random rotation") +
#   ylab("Normalized test score of best model \n (on valid set) after 20 random search iterations") +
#   theme_minimal(base_size=22) +
#   theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
#   colScale
