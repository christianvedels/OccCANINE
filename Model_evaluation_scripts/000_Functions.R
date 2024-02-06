# Functions
# Created:    2024-01-16
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Functions for model evaluation

# ==== Colors ====
blue = "#273a8f"
green = "#2c5c34"
red = "#b33d3d"
orange = "#DE7500"

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
      )
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
    ggplot(aes(thr, value, col = lang_info, shape = lang_info)) + 
    geom_point() + 
    geom_line(aes(lty = lang_info)) +
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
  ggsave(path, width = 6, height = 3.5, plot = p1)
  
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

# ==== Get_wrong_cases ====
Get_wrong_cases = function(x0, pred_data, key){
  x = x0 %>% 
    Pred_to_pred() 
  
  cases = pred_data %>%
    select(RowID, lang, Source, occ1, hisco_1:hisco_5) %>% 
    left_join(x, by = "RowID") %>% 
    filter( # Only where one code is the TRUE code
      !is.na(hisco_1),
      is.na(hisco_2),
    ) %>% 
    rowwise() %>% 
    filter( # Only where one 
      !acc(hisco_1, pred_hisco_1)
    ) %>% 
    ungroup() %>% 
    select(RowID, hisco_1, occ1) %>% 
    left_join(key %>% select(hisco, en_hisco_text), by = c(hisco_1="hisco")) %>% 
    left_join(x0, by = "RowID", suffix = c("_correct", ""))
  
  return(cases)
}

# ==== Plot_wrong ====
# capN: How many plots?
Plot_wrong = function(x, capN = 100, n_guesses = 10, n_char = 20, name){
  x_long = x %>%
    pivot_longer(
      cols = -c(RowID, hisco_1_correct, en_hisco_text, occ1), 
      names_to = c(".value", "set"), 
      names_pattern = "(.*)_(\\d+)"
    ) %>% 
    mutate(
      hisco = Fix_HISCO(hisco),
      hisco_correct = Fix_HISCO(hisco_1_correct)
    )
  
  ids = unique(x_long$RowID)
  
  # Define custom colors for each category
  custom_colors = c("TRUE" = red,
                    "FALSE" = "grey")
  
  foreach(i = seq(capN)) %do% {
    id_i = ids[i]
    if(is.na(id_i)){
      return(0)
    }
    
    # Make plot for ID
    plot_data_i = x_long %>% 
      filter(RowID==id_i) %>% 
      filter(as.numeric(set) <= n_guesses) %>% 
      mutate(
        correct = hisco_correct == hisco
      ) %>% 
      mutate(
        label = paste(hisco, desc, sep = ": ")
      ) %>% 
      mutate(
        label = ifelse(
          nchar(label) > n_char,
          paste0(substr(label, 1, n_char), "..."),
          label
        )
      ) %>% 
      mutate(
        label = factor(label, levels = label)
      ) %>% 
      mutate(
        prob_label = ifelse(
          prob > 0.001,
          signif(prob, 2),
          "<0.001"
        )
      )
    
    title_i = paste(unique(plot_data_i$hisco_correct), unique(plot_data_i$en_hisco_text))
    subtitle_i = paste0('Descirption: "', unique(plot_data_i$occ1), '"')
    
    p_i = plot_data_i %>% 
      ggplot(aes(
        label, prob, fill = correct
      )) +
      geom_bar(stat = "identity") + 
      theme_bw() + 
      theme(
        axis.text.x = element_text(angle = 90),
        legend.position = "bottom"
      ) +
      labs(
        title = title_i,
        subtitle = subtitle_i,
        y = "HISCO probability",
        x = "",
        fill = "Correct label:"
      ) + 
      geom_text(aes(y = prob, label = prob_label)) + 
      ylim(c(0,1)) + 
      scale_fill_manual(values = custom_colors) # Use custom fill colors
    
    fname_i = paste0("Eval_plots/Wrong_labels/", name,"/RowID_", id_i, ".png")
    ggsave(fname_i, plot = p_i, width = 6, height = 4)
  }
}
