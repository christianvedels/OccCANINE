# Functions
# Created:    2024-01-16
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Functions for model evaluation


# ==== Pred_to_pred ====
# Transforms the pred files outputted from n301_predict_eval.py to label predictions

Pred_to_pred = function(x, threshold = 0.5){
  
  # Extract relevant data
  probs = x %>% 
    select(RowID, starts_with("prob_"))
  
  hiscos = x %>% 
    select(RowID, starts_with("hisco_"))
  
  # Reshape data for vectorized operations
  hisco_long = hiscos %>% 
    pivot_longer(cols = starts_with("hisco_"), names_to = "hisco_name", values_to = "hisco_value") %>% 
    mutate(hisco_name = gsub("hisco", "", hisco_name))
  prob_long = probs %>% 
    pivot_longer(cols = starts_with("prob_"), names_to = "prob_name", values_to = "prob_value") %>% 
    mutate(prob_name = gsub("prob", "", prob_name))
  
  # Join info and cut at threshold
  res = hisco_long %>% 
    left_join(prob_long, by = c("RowID", "hisco_name" = "prob_name")) %>% 
    filter(
      prob_value > threshold
    ) %>% 
    # Make wider
    group_by(RowID) %>%
    mutate(tmp_id = row_number()) %>%
    filter(tmp_id <= 5) %>%  # Only yop 5 hiscos
    
    # Make sure that there are  no -1 together with other valid occupations:
    # Criteria, which together should trigger filtering:
    #   1. More than one code
    #   2. One of these codes are -1
    mutate(
      crit1 = max(tmp_id)>1
    ) %>% 
    mutate(
      crit2 = hisco_value == -1
    ) %>% 
    filter(
      !(crit1&crit2)
    ) %>% 
    select(
      -crit1, -crit2
    ) %>% 
    
    # Make wider
    pivot_wider(id_cols = RowID, names_from = tmp_id, values_from = hisco_value, names_prefix = "pred_hisco_")
  
  # Add any missing
  res = data.frame(RowID = x$RowID) %>% 
    left_join(res, by = "RowID") 
    
  # Test
  test = NROW(res %>% count(RowID) %>% filter(n>1))!=0
  if(test){
    stop("None unique ids in res")
  }
  
  # Replace NA in first column
  res = res %>% 
    mutate(
      pred_hisco_1 = ifelse(is.na(pred_hisco_1), -1, pred_hisco_1)
    )
  
  # Fix missing columns
  if(!"pred_hisco_1" %in% names(res)) res$pred_hisco_1 = NA
  if(!"pred_hisco_2" %in% names(res)) res$pred_hisco_2 = NA
  if(!"pred_hisco_3" %in% names(res)) res$pred_hisco_3 = NA
  if(!"pred_hisco_4" %in% names(res)) res$pred_hisco_4 = NA
  if(!"pred_hisco_5" %in% names(res)) res$pred_hisco_5 = NA
  
  return(res)
}


# ==== Run_tests ====
acc = function(x, y){
  # x: Truth; y: Prediction
  x = x[!is.na(x)]
  y = y[!is.na(y)]
  res = all(x %in% y) & all(y %in% x)
  return(res)
}

precision = function(x, y){
  # x: Truth; y: Prediction
  x = x[!is.na(x)]
  y = y[!is.na(y)]
  res = mean(y %in% x)
  return(res)
}

recall = function(x, y){
  # x: Truth; y: Prediction
  x = x[!is.na(x)]
  y = y[!is.na(y)]
  res = mean(x %in% y)
  return(res)
}

# Performs tests of acc, precision, recall, f1
Run_tests = function(pred, truth){
  x = truth %>%
    select(RowID, lang, Source, occ1, hisco_1:hisco_5) %>% 
    left_join(pred, by = "RowID")
  
  # Fix missing columns
  if(!"pred_hisco_1" %in% names(pred)) pred$pred_hisco_1 = NA
  if(!"pred_hisco_2" %in% names(pred)) pred$pred_hisco_2 = NA
  if(!"pred_hisco_3" %in% names(pred)) pred$pred_hisco_3 = NA
  if(!"pred_hisco_4" %in% names(pred)) pred$pred_hisco_4 = NA
  if(!"pred_hisco_5" %in% names(pred)) pred$pred_hisco_5 = NA
    
  # Tests
  x = x %>% 
    rowwise() %>% 
    mutate(
      acc = acc(
        c(hisco_1, hisco_2, hisco_3, hisco_4, hisco_5), 
        c(pred_hisco_1, pred_hisco_2, pred_hisco_3, pred_hisco_4, pred_hisco_5)
      ),
      precision = precision(
        c(hisco_1, hisco_2, hisco_3, hisco_4, hisco_5), 
        c(pred_hisco_1, pred_hisco_2, pred_hisco_3, pred_hisco_4, pred_hisco_5)
      ),
      recall = recall(
        c(hisco_1, hisco_2, hisco_3, hisco_4, hisco_5), 
        c(pred_hisco_1, pred_hisco_2, pred_hisco_3, pred_hisco_4, pred_hisco_5)
      ),
    )
  
  sum_all =  x %>% 
    ungroup() %>%  
    mutate(n = n()) %>% 
    summarise_at(c("acc", "precision", "recall", "n"), .funs = mean) %>% 
    mutate(
      f1 = 2*precision*recall/(precision+recall)
    ) %>% 
    mutate(
      summary = "All"
    )
  
  sum_lang = x %>% 
    group_by(lang) %>% 
    mutate(n = n()) %>% 
    summarise_at(c("acc", "precision", "recall", "n"), .funs = mean) %>% 
    mutate(
      f1 = 2*precision*recall/(precision+recall)
    ) %>% 
    mutate(
      summary = "Lang"
    )
  
  sum_source = x %>% 
    group_by(Source) %>% 
    mutate(n = n()) %>% 
    summarise_at(c("acc", "precision", "recall", "n"), .funs = mean) %>% 
    mutate(
      f1 = 2*precision*recall/(precision+recall)
    ) %>% 
    mutate(
      summary = "Source"
    )
  
  sum_hisco1 = x %>% 
    group_by(hisco_1) %>% 
    mutate(n = n()) %>% 
    summarise_at(c("acc", "precision", "recall", "n"), .funs = mean) %>% 
    mutate(
      f1 = 2*precision*recall/(precision+recall)
    ) %>% 
    mutate(
      summary = "hisco"
    )
  
  res = list(
    sum_all,
    sum_lang,
    sum_source,
    sum_hisco1
  ) %>% bind_rows()
  
  return(res)
}

# ==== floor0 ====
# floor0 is a modicaiton of floor

floor0 = function(x, n){
  floor(x*n)/n
}

# ==== plot of thresholds ====
plot_of_thresholds = function(x, name){
  
  plot_stats = x %>% 
    pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
    group_by(stat, lang_info) %>% 
    mutate(
      best = max(value) == value
    ) %>% 
    # filter(!lang_info) %>%  # Something weird happens when lang is included
    ungroup()
  
  p1 = plot_stats %>% 
    ggplot(aes(thr, value, col = lang_info)) + 
    geom_point() + 
    geom_line(lty = 2) +
    geom_point(data = subset(plot_stats, best), aes(thr, value), shape = 4, size = 3, col = "black") +  # Highlight max points
    geom_vline(data = subset(plot_stats, best), aes(xintercept = thr), lty = 2) +
    geom_hline(data = subset(plot_stats, best), aes(yintercept = value), lty = 2) +
    facet_wrap(~stat) +  # Changed from ~name to ~stat based on your data structure
    theme_bw() +
    scale_y_continuous(
      labels = scales::percent,
      breaks = seq(floor0(min(plot_stats$value),1), 1, by = 0.025),  # Labels at every 0.1
    ) + 
    scale_x_continuous(
      breaks = seq(0, 1, by = 0.1),  # Labels at every 0.1
      minor_breaks = seq(0, 0, by = 0.05) # Lines at every 0.05
    ) + 
    labs(
      x = "Threshold",
      y = "Statistic",
      title = name,
      subtitle = paste("N =", plot_stats$n[1])
    )
  
  path = paste0("Eval_plots/Optimal_threshold/", name, ".png")
  ggsave(path, width = 10, height = 8, plot = p1)
  
  return(p1)
}

# ==== Fix_HISCO ====
Fix_HISCO = function(x){
  case_when(
    nchar(x) == 5 ~ as.character(x),
    nchar(x) == 4 ~ paste0("0",x),
    TRUE ~  as.character(x)
  )
}

