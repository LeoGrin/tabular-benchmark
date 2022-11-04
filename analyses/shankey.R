df <- read_csv("data/data_metadata/cc18_cleaned.csv")

df %>% 
 mutate(used = benchmark_categorical | benchmark_numerical) %>% 
  mutate(`Too small before processing` = original_n_samples < 3000) %>% 
  mutate(`Too small after preprocessing` = if_else(`Too small before processing`, F, `Too small`)) %>% 
  mutate(`Too easy` = too_easy) %>% 
  mutate(`Used` = used) %>% 
 mutate(`Not heterogeneous` = !heterogeneous) %>% 
 mutate(`High dimensional` = !ratio_n_p) %>% 
 pivot_longer(cols = c("Used", "Too easy", "High dimensional", "Not heterogeneous", "Too small before processing", "Artificial", "Deterministic", "Too small after preprocessing", "Not enough features"), names_to = "Criterion") %>% 
 select(dataset_name, Criterion, value) %>% 
 drop_na(value) %>% 
 filter(value == 1) %>% 
  distinct(dataset_name, .keep_all = T) %>% 
  group_by(Criterion) %>% 
  summarise(count = n()) %>% 
  write_csv("cc18_decomposition.csv")



df <- read_csv("data/data_metadata/automl_classif_cleaned.csv")

df %>% 
  mutate(used = benchmark_categorical | benchmark_numerical) %>% 
  mutate(`Too small before processing` = original_n_samples < 3000) %>% 
  mutate(`Too small after preprocessing` = if_else(`Too small before processing`, F, too_small)) %>% 
  mutate(`Not enough features` = not_enough_features) %>% 
  mutate(`Too easy` = too_easy) %>% 
  mutate(`Used` = used) %>% 
  mutate(`Not heterogeneous` = !heterogeneous) %>% 
  mutate(`High dimensional` = !ratio_n_p) %>% 
  pivot_longer(cols = c("Used", "Too easy", "High dimensional", "Not heterogeneous", "Artificial", "Deterministic", "Too small before processing", "Too small after preprocessing", "Not enough features", "Other"), names_to = "Criterion") %>% 
  select(dataset_name, Criterion, value) %>% 
  drop_na(value) %>% 
  filter(value == 1) %>% 
  distinct(dataset_name, .keep_all = T) %>% 
  group_by(Criterion) %>% 
  summarise(count = n()) %>% 
  write_csv("automl_classif_decomposition.csv")

df <- read_csv("data/data_metadata/automl_regression_cleaned.csv")

df %>% 
  mutate(`Too small before processing` = original_n_samples < 3000) %>% 
  mutate(`Too small after preprocessing` = if_else(`Too small before processing`, F, too_small)) %>% 
  mutate(`Not enough features` = not_enough_features) %>% 
  mutate(`Too easy` = too_easy) %>% 
  mutate(`High dimensional` = !ratio_n_p) %>% 
  pivot_longer(cols = c("Used", "Too easy", "High dimensional", "Artificial", "Deterministic", "Too small before processing", "Too small after preprocessing", "Not enough features", "Other"), names_to = "Criterion") %>% 
  select(dataset_name, Criterion, value) %>% 
  drop_na(value) %>% 
  filter(value == 1) %>% 
  distinct(dataset_name, .keep_all = T) %>% 
  group_by(Criterion) %>% 
  summarise(count = n()) %>% 
  write_csv("automl_regression_decomposition.csv")





