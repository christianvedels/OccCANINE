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
files = list.files("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/lang", full.names = TRUE) %>% sample(replace = FALSE)
start_time = Sys.time()
foreach(i = seq_along(files), .combine = "bind_rows") %do% {
    
    f = files[i]
    
    eval_data = read_csv(f, show_col_types = FALSE, guess_max = 1000000, progress = FALSE) %>%
        mutate(file = f) %>%
        mutate(file = gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/lang/", "", file)) %>%
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

make_plot_lang_info = function(data, all_stats = NULL){
  
  lvls = data %>% 
    filter(stat == "F1 score") %>% 
    arrange(-value) %>% 
    pull(lang) %>% 
    unique()

  
  p1 = data %>% 
    mutate(lang = factor(lang, lvls)) %>% 
    ggplot(aes(lang, value, fill = lang_info)) + 
    geom_bar(stat = "identity", alpha = 0.8, position = "dodge") +
    scale_y_continuous(
      labels = scales::percent,
      breaks = seq(0,100, by = 0.1)
    ) + 
    scale_fill_manual(
      values = c(colours$red, colours$green),
      name = "Language info provided"
    ) +
    theme_bw() + 
    facet_wrap(~stat, scales = "free") + 
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
all_stats = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance_greedy_wlang.csv")
all_stats = all_stats %>%
  rename(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    `F1 score` = f1
  ) %>%
  pivot_longer(
    cols = c(Accuracy, Precision, Recall, `F1 score`),
    names_to = "stat",
    values_to = "value"
  ) %>% 
  mutate(
    lang = "nl" # Choose what visually looks best
  ) %>% 
  select(stat, lang, value)


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

# Greedy 5 digits
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


# Greedy 5 digits - without language info
p1 = plot_data %>% 
  filter(prediction_type == "greedy") %>% 
  filter(digits == 5) %>% 
  make_plot_lang_info() + theme(legend.position = "bottom")

p1
ggsave(
  "Project_dissemination/Paper_replication_package/Figures/Performance_by_lang_greedy_lang_info.png",
  plot = p1,
  width = dims$width,
  height = dims$height
)

sink("Project_dissemination/Paper_replication_package/Tables/Performance_by_lang_(boost_from_knowing_lang).txt", append = FALSE)
cat("\nTest set performance by language and prediction type")
cat("\n ---> This is just used to check the maximum value of the boost from knowing the language")
cat("\n ---> (This is the reason it is not a latex table)\n")
plot_data %>% 
  filter(prediction_type == "greedy") %>% 
  filter(digits == 5) %>%
  select(-label, -file) %>%
  mutate(lang_info = ifelse(lang_info, "yes", "no")) %>%
  pivot_wider(
    names_from = "lang_info",
    values_from = "value",
    names_prefix = "lang_info_"
  ) %>%
  mutate(
    dif = abs(lang_info_yes - lang_info_no)
  ) %>%
  arrange(-dif) %>% select(-threshold, -digits) %>% print()
sink()




