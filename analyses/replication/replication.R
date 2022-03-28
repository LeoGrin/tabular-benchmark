library(tidyverse)

df_me <- read_tsv("analyses/replication/published - full_df_published.tsv") %>% 
  select(doi, title_CR, journal_pr, journal.x, journal.y, is_published_api, is_published_reported)

df_me_2 <- read_csv("analyses/replication/full_df_published.csv")

df_them <- read_csv("analyses/replication/resolutions_for_calculations.csv") %>% 
  mutate(doi = paper_doi) %>% 
  select(title_CR, doi, publication_date, published, journal) %>% 
  mutate(title_CR = str_to_upper(title_CR))


df_me_2 %>% left_join(df_them, by = c("doi")) %>% 
  #relocate(title_CR, doi.x, doi.y) %>% 
  filter(published==F) %>% 
  select(title_CR, journal_pr, is_published_api, is_published_reported, published)

df_me %>% 
  mutate(published = is_published_api | is_published_reported) %>% 
  filter(published==T)
