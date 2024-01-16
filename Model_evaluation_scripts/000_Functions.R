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



# === floor0 ====
# floor0 is a modicaiton of floor

floor0 = function(x, n){
  floor(x*n)/n
}



