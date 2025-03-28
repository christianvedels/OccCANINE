# Threshold tuning
# Created:  2025-03-27
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Best threshold for the model

# ==== Libraries ====
library(tidyverse)
source("Project_dissemination/Paper_replication_package/Model_eval_R/000_Functions.R")
library(reticulate)
library(foreach)
use_condaenv("hisco_dev")

# Use reticulate to load numpy
np = reticulate::import("numpy")

# ==== Load data ====
key = read_csv("histocc/Data/Key.csv")[-1,]
val1_data = read_csv("Data/eval-results/val1_probs/val1_data.csv", guess_max = 100000)
probs_flat = np$load("Data/eval-results/val1_probs/probs_flat.npy")
# probs_mixer = np$load("Data/eval-results/val1_probs/probs_mixer.npy")

# Sample for development
set.seed(20)
rows = sample(1:NROW(val1_data), 1000)
probs_flat = probs_flat[rows,]
val1_data = val1_data[rows,]

# ==== prob to pred ====
prob_to_pred = function(probs, threshold){
  probs = probs > threshold
  probs = apply(probs, 1, 
                function(x){
                  res = which(x == TRUE)-1 # Zero indexing
                  if(length(res) == 0){
                    return(2) # Code for "-1"
                  } else {
                    return(res)
                  }
                })
  return(probs)
}
# ==== Convert true labels to list form ====
# Convert code1, code2, code3, code4, code5 into a list
true_data = val1_data %>% 
    select(code1, code2, code3, code4, code5) %>% 
    pmap(function(code1, code2, code3, code4, code5) {
        c(code1, code2, code3, code4, code5) %>% 
            na.omit() %>% 
            as.integer()
    })




get_stats = function(probs, true_data, threshold){
  y_pred = prob_to_pred(probs, threshold)
  
  res = foreach(i = 1:length(y_pred), .combine = "bind_rows") %do% {
    y_true_i = true_data[[i]]
    y_pred_i = y_pred[[i]]
    
    data.frame(
        acc = acc(y_true_i, y_pred_i),
        prec = prec(y_true_i, y_pred_i),
        recall = recall(y_true_i, y_pred_i),
        f1_score = f1_score(y_true_i, y_pred_i)
    )
  }
  return(res)
}

threshold_tuning = function(probs_flat, true_data, output_path = "Project_dissemination/Paper_replication_package/Figures/threshold_tuning.png") {
    # ===== Overall threshold tuning =====
    # Tune thresholds
    thresholds = seq(0.01, 0.99, 0.01)
    results = foreach(threshold = thresholds, .combine = "bind_rows") %do% {
        # cat(threshold, "\n")
        get_stats(probs_flat, true_data, threshold) %>% 
            mutate(threshold = threshold) %>% 
            summarise_all(mean)
    }

    max_res = results %>% 
            pivot_longer(-threshold, names_to = "metric", values_to = "value") %>%
            group_by(metric) %>% 
            filter(value == max(value)) %>% 
            ungroup()

    print(max_res)

    # ==== Make plot ====
    p1 = results %>% 
            pivot_longer(-threshold, names_to = "metric", values_to = "value") %>% 
            ggplot(aes(x = threshold, y = value)) +
            geom_point() +
            geom_line() +
            facet_wrap(~metric, scales = "free") +
            labs(
                    title = "Threshold tuning",
                    x = "Threshold",
                    y = "Value"
            ) +
            theme_bw()

    # Add max points
    p1 = p1 + 
            geom_point(data = results %>% 
                                         filter(f1_score == max(f1_score)) %>% 
                                         select(threshold, f1_score) %>% 
                                         mutate(metric = "f1_score"), aes(x = threshold, y = f1_score), color = "red", size = 3) +
            geom_point(data = results %>% 
                                         filter(prec == max(prec)) %>% 
                                         select(threshold, prec) %>% 
                                         mutate(metric = "prec"), aes(x = threshold, y = prec), color = "blue", size = 3) +
            geom_point(data = results %>% 
                                         filter(recall == max(recall)) %>% 
                                         select(threshold, recall) %>% 
                                         mutate(metric = "recall"), aes(x = threshold, y = recall), color = "green", size = 3) +
            geom_point(data = results %>%
                                             filter(acc == max(acc)) %>% 
                                             select(threshold, acc) %>% 
                                             mutate(metric = "acc"), aes(x = threshold, y = acc), color = "purple", size = 3)  

    ggsave(output_path, p1, width = 10, height = 8)
}

threshold_tuning_by_language = function(val1_data, true_data, probs_flat, output_dir = "Project_dissemination/Paper_replication_package/Figures/") {
    # Foreach language in val1_data
    langs = unique(val1_data$lang)
    val1_data$index = 1:NROW(val1_data)
    for(lang in langs){
        cat(lang, "\n")
        lang_data = val1_data %>% filter(lang == !!lang)
        lang_true_data = true_data[lang_data$index]
        lang_probs_flat = probs_flat[lang_data$index,]
        
        threshold_tuning(lang_probs_flat, lang_true_data, output_path = paste0(output_dir, "threshold_tuning_", lang, ".png"))
    }
}

# ==== Run threshold tuning ====
threshold_tuning(probs_flat, true_data, output_path = "Project_dissemination/Paper_replication_package/Figures/threshold_tuning.png")
# threshold_tuning(probs_mixer, true_data, output_path = "Project_dissemination/Paper_replication_package/Figures/threshold_tuning_mixer.png")

threshold_tuning_by_language(val1_data, true_data, probs_flat, output_dir = "Project_dissemination/Paper_replication_package/Figures/")
# threshold_tuning_by_language(val1_data, true_data, probs_mixer, output_dir = "Project_dissemination/Paper_replication_package/Figures/")
