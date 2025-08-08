# Descriptive statistics
# Created:  2025-06-15
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Plots and tables of threshold tuning results for the paper

# ==== Libraries ====
library(tidyverse)
library(foreach)
source("Project_dissemination/Paper_replication_package/Model_eval_R/000_Functions.R")

# ==== Load data ====
flat_overall = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_flat_langknown.csv")
flat_overall_unk = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_flat_langunk.csv")
full_overall = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_full_langknown.csv")
full_overall_unk = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/threshold_tuning_full_langunk.csv")

# Load data by language to data.frame
files = list.files("Project_dissemination/Paper_replication_package/Data/thr_tuning_by_lang", full.names = TRUE)
thr_tuning_by_lang = foreach(f = files, .combine = "bind_rows") %do% {
    read_csv(f, show_col_types = FALSE) %>%
        mutate(file = f) %>%
        mutate(file = gsub("Project_dissemination/Paper_replication_package/Data/thr_tuning_by_lang/", "", file)) %>%
        mutate(lang = gsub("threshold_tuning_full_", "", file)) %>%
        mutate(lang = gsub("threshold_tuning_flat_", "", lang)) %>%
        mutate(lang = gsub(".csv", "", lang)) %>%
        mutate(lang = gsub("langunk_", "", lang)) %>%
        mutate(lang = gsub("langknown_", "", lang)) %>%
        mutate(
            type = ifelse(grepl("langunk", file), "Without language info.", "With language info.")
        )
}

flat_by_lang = thr_tuning_by_lang %>%
    filter(grepl("flat", file))

full_by_lang = thr_tuning_by_lang %>%
    filter(grepl("full", file))

# ==== Data preparation ====
overall_flat = flat_overall %>% 
    mutate(type = "With language info.") %>%
    bind_rows(
        flat_overall_unk %>%
        mutate(type = "Without language info.")
    ) %>%
    pivot_longer(cols = c("accuracy", "precision", "recall", "f1"), 
               names_to = "statistic", 
               values_to = "value") %>%
    group_by(type, statistic) %>%
    mutate(best = value == max(value)) %>%
    ungroup()

overall_full = full_overall %>% 
    mutate(type = "With language info.") %>%
    bind_rows(
        full_overall_unk %>%
        mutate(type = "Without language info.")
    ) %>%
    pivot_longer(cols = c("accuracy", "precision", "recall", "f1"), 
               names_to = "statistic", 
               values_to = "value") %>%
    group_by(type, statistic) %>%
    mutate(best = value == max(value)) %>%
    ungroup()

bylang_flat = flat_by_lang %>%
    filter(grepl("flat", file)) %>%
    pivot_longer(cols = c("accuracy", "precision", "recall", "f1"), 
               names_to = "statistic", 
               values_to = "value") %>%
    group_by(lang, type, statistic) %>%
    mutate(best = value == max(value)) %>%
    ungroup()

bylang_full = full_by_lang %>%
    filter(grepl("full", file)) %>%
    pivot_longer(cols = c("accuracy", "precision", "recall", "f1"), 
               names_to = "statistic", 
               values_to = "value") %>%
    group_by(lang, type, statistic) %>%
    mutate(best = value == max(value)) %>%
    ungroup()

# ==== threshold_tuning_plot ====
options(scipen = 999) # Disable scientific notation
threshold_tuning_plot = function(data, observation) {
  p1 = data %>%
    ggplot(aes(x = threshold, y = value, color = type)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ statistic, scales = "free_y") +
    theme_bw() +
    theme(legend.title = element_blank())  + 
    theme(legend.position = "bottom") +
    labs(title = NULL, subtitle = NULL) + 
    scale_color_manual(values = c(colours$red, colours$green)) + 
    labs(
        x = "Threshold",
        col = "",
        shape = "",
        y = "Statistic"
    ) +
    labs(
        caption = paste0("N val. obs.: ", observation)
    )

  # Add max
  p1 = p1 +
    geom_point(data = subset(data, best), aes(threshold, value), shape = 4, size = 3, col = "black") +  # Highlight max points
    geom_vline(data = subset(data, best), aes(xintercept = threshold), lty = 2) +
    geom_hline(data = subset(data, best), aes(yintercept = value), lty = 2)
  
  return(p1)
}

# ==== Plot overall known language ====
n_obs_flat = overall_flat$n %>% unique()
n_obs_full = overall_full$n %>% unique()

p1 = threshold_tuning_plot(overall_flat, observation = n_obs_flat)
p2 = threshold_tuning_plot(overall_full, observation = n_obs_full)

ggsave(
    "Project_dissemination/Paper_replication_package/Figures/threshold_tuning_flat.png", 
    p1, 
    width = dims$width, 
    height = dims$height,
    dpi = 600
)
ggsave(
    "Project_dissemination/Paper_replication_package/Figures/threshold_tuning_flat.pdf", 
    p1, 
    width = dims$width, 
    height = dims$height,
    dpi = 600
)

ggsave(
    "Project_dissemination/Paper_replication_package/Figures/threshold_tuning_full.png", 
    p2, 
    width = dims$width, 
    height = dims$height
)
ggsave(
    "Project_dissemination/Paper_replication_package/Figures/threshold_tuning_full.pdf", 
    p2, 
    width = dims$width, 
    height = dims$height
)

# ==== Plot by language ====
foreach(l = unique(bylang_flat$lang)) %do% {
  data = bylang_flat %>% filter(lang == l)
  
  n_l = unique(data$n)

  if(length(n_l) != 1) {
    warning(paste0("Multiple n values for language ", l, ": ", paste(n_l, collapse = ", ")))
  }

  p = threshold_tuning_plot(data, observation = n_l)
  
  ggsave(
    paste0("Project_dissemination/Paper_replication_package/Figures/threshold_tuning_flat/threshold_tuning_flat_", l, ".png"), 
    p, 
    width = dims$width, 
    height = dims$height
  )
  ggsave(
    paste0("Project_dissemination/Paper_replication_package/Figures/threshold_tuning_flat/threshold_tuning_flat_", l, ".pdf"), 
    p, 
    width = dims$width, 
    height = dims$height
  )
}

foreach(l = unique(bylang_full$lang)) %do% {
  data = bylang_full %>% filter(lang == l)
  
  n_l = unique(data$n)

  if(length(n_l) != 1) {
    warning(paste0("Multiple n values for language ", l, ": ", paste(n_l, collapse = ", ")))
  }

  p = threshold_tuning_plot(data, observation = n_l)
  
  ggsave(
    paste0("Project_dissemination/Paper_replication_package/Figures/threshold_tuning_full/threshold_tuning_full_", l, ".png"), 
    p, 
    width = dims$width, 
    height = dims$height
  )
  ggsave(
    paste0("Project_dissemination/Paper_replication_package/Figures/threshold_tuning_full/threshold_tuning_full_", l, ".pdf"), 
    p, 
    width = dims$width, 
    height = dims$height
  )
}

# ==== Print tables of best thresholds overall ====
tmp1 = overall_full %>% 
    filter(best) %>%
    mutate(method = "full") %>%
    select(statistic, method, type, threshold, value, n) %>%
    mutate(type = ifelse(type == "With language info.", "Yes", "No")) %>%
    rename(`Lang. info.` = type) %>%
    arrange(statistic)

tmp2 = overall_flat %>% 
    filter(best) %>%
    mutate(method = "flat") %>%
    select(statistic, method, type, threshold, value, n) %>%
    mutate(type = ifelse(type == "With language info.", "Yes", "No")) %>%
    rename(`Lang. info.` = type) %>%
    arrange(statistic)

tmp = bind_rows(tmp1, tmp2) %>%
    select(method, statistic, `Lang. info.`, threshold, value) %>%
    arrange(statistic, method, `Lang. info.`) 

sink("Project_dissemination/Paper_replication_package/Tables/threshold_tuning_overall.txt", append = FALSE)
print(tmp)

cat("\nNumber of observations full:\n", tmp1$n %>% unique(), "\n")
cat("\nNumber of observations flat:\n", tmp2$n %>% unique(), "\n")

sink()

# ==== Print tables of best threshholds by lang ====
x1 = overall_flat %>% 
    filter(best) %>%
    # filter(type == "With language info.") %>%
    mutate(method = "flat") %>%
    mutate(data = "all_data")

x2 = overall_full %>% filter(best) %>% 
    filter(best) %>%
    # filter(type == "With language info.") %>%
    mutate(method = "full") %>%
    mutate(data = "all_data")

x3 = bylang_flat %>% filter(best) %>% 
    # filter(type == "With language info.") %>%
    mutate(method = "flat") %>%
    mutate(data = "by_lang")

x4 = bylang_full %>% filter(best) %>%
    # filter(type == "With language info.") %>%
    mutate(method = "full") %>%
    mutate(data = "by_lang")

res = x1 %>% bind_rows(x2, x3, x4) %>%
    select(method, data, type, lang, statistic, threshold, value, n) %>%
    drop_na(lang, n) %>%
    arrange(data, method, type, lang, statistic)

# If several, pick the first
res1 = res %>%
    group_by(method, data, type, lang, statistic) %>%
    slice(1) %>%
    ungroup() %>%
    filter(type == "With language info.") %>%
    select(-type)

# Print to Tables/threshold_tuning.txt
sink("Project_dissemination/Paper_replication_package/Tables/threshold_tuning.txt", append = FALSE)

res1 %>%
    knitr::kable(, 
      digits = 3, 
      format = "latex",
      booktabs = TRUE) %>% print()

sink()




