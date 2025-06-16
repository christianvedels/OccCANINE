# Test model performance by language
# Created:  2025-06-16
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Make pretty tables of test data performance for the paper

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)
source("Project_dissemination/Paper_replication_package/Model_eval_R/000_Functions.R")

# ==== Load data ====
mean0 = function(x) mean(as.numeric(x))
files = list.files("Project_dissemination/Paper_replication_package/Data/test_performance/lang", full.names = TRUE) %>% sample(replace = FALSE)
start_time = Sys.time()
foreach(i = seq_along(files), .combine = "bind_rows") %do% {
    
    f = files[i]
    
    eval_data = read_csv(f, show_col_types = FALSE, guess_max = 1000000, progress = FALSE) %>%
        mutate(file = f) %>%
        mutate(file = gsub("Project_dissemination/Paper_replication_package/Data/test_performance/lang/", "", file)) %>%
        ungroup()
    
    return(eval_data)
    
} -> test_performance


# ==== Clean file ====
plot_data = test_performance %>% 
  mutate(
    lang_info = !grepl("unk_digits", file)
  ) %>% 
  pivot_longer(c(accuracy, precision, recall, f1), names_to = "stat") %>% 
  mutate(
    stat = case_when(
      stat == "accuracy" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  ) %>% 
  mutate(
    digits = case_when(
      grepl("digits_1", file) ~ 1,
      grepl("digits_2", file) ~ 2,
      grepl("digits_3", file) ~ 3,
      grepl("digits_4", file) ~ 4,
      grepl("digits_5", file) ~ 5
    )
  ) %>% 
  mutate(
    label = paste0(scales::percent(value, 0.1))
  )

#  ==== Make plot function ====
make_plot = function(data, all_stats = NULL){
  
  lvls = data %>% 
    filter(stat == "F1 score") %>% 
    arrange(-value) %>% pull(lang)
  
  p1 = data %>% 
    mutate(lang = factor(lang, lvls)) %>% 
    ggplot(aes(lang, value)) + 
    geom_bar(stat = "identity", alpha = 0.8, fill = colours$red) +
    scale_y_continuous(
      labels = scales::percent,
      breaks = seq(0,100, by = 0.1)
    ) + 
    theme_bw() + 
    facet_wrap(~stat, scales = "free") + 
    geom_text(
      y = 0.4,
      aes(label = label, x = lang),
      col = "grey", 
      angle = 90
    ) + 
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5)
    ) + 
    labs(
      y = "Statistic",
      x = "Language"
    )
  
  if(!is.null(all_stats)){
    p1 = p1 + 
      geom_hline(aes(yintercept = value), data = all_stats, lty = 2) + 
      geom_text(
        data = all_stats,
        aes(label = scales::percent(value, 0.1), x = lang, y = value-0.075),
        inherit.aes = FALSE
      )
  }
  
  return(p1)
}

# ==== All stats ====
plot_data$stat %>% unique()

all_stats = data.frame(
  stat = c("Accuracy", "Precision", "Recall", "F1 score"),
  value = c(0.960, 0.960, 0.967, 0.963),
  lang = "nl"
)

# ==== Make plots ====
# Flat 5 digits
p1 = plot_data %>% 
  filter(lang_info) %>% 
  filter(prediction_type == "flat") %>% 
  filter(digits == 5) %>% 
  make_plot()

p1
ggsave(
  "Project_dissemination/Paper_replication_package/Figures/Performance_by_lang_flat.png",
  plot = p1,
  width = dims$width,
  height = dims$height
)

# Flat 5 digits
p1 = plot_data %>% 
  filter(lang_info) %>% 
  filter(prediction_type == "greedy") %>% 
  filter(digits == 5) %>% 
  make_plot(all_stats)

p1
ggsave(
  "Project_dissemination/Paper_replication_package/Figures/Performance_by_lang_greedy.png",
  plot = p1,
  width = dims$width,
  height = dims$height
)




