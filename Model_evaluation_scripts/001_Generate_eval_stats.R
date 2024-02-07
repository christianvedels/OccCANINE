# Evaluate model
# Created:    2024-01-16
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Reads predicted labels and produces evaluation results
#
# Output:     paste0("Data/Summary_data/Model_performance", model_name, ".csv")
# 
# NOTE: This loads a function to use in another script. 

Generate_eval_stats = function(model_name = "CANINE", toyrun = FALSE, overwrite = FALSE){
  # The function automatically loads previous results
  #
  # model_name:   Name of model in 'Predictions'
  # toyrun:       Should this be a quick toyrun with only the first 100 rows?
  # overwrite:    Should previous results be overwritten?
  
  # ==== Libraries ====
  require(tidyverse)
  require(foreach)
  source("Model_evaluation_scripts/000_Functions.R")
  
  # Test if this should even run:
  test1 = file.exists(paste0("Data/Summary_data/Model_performance", model_name, ".csv"))
  test2 = !overwrite
  
  if(test1 & test2){
    cat("\nResults already exists for this. Loading file and return it.")
    x = read_csv2(paste0("Data/Summary_data/Model_performance", model_name, ".csv"))
    return(x)
  }
  
  # ==== Read data ====
  key = read_csv("Data/Key.csv")
  
  n_max = ifelse(toyrun, 100, Inf)
  
  pred_data = read_csv(paste0("Data/Predictions/Predictions", model_name, "/pred_data.csv"), n_max = n_max)[,-1]
  preds_w_lang = read_csv(paste0("Data/Predictions/Predictions", model_name, "/preds_w_lang.csv"), n_max = n_max)[,-1]
  preds_wo_lang = read_csv(paste0("Data/Predictions/Predictions", model_name, "/preds_wo_lang.csv"), n_max = n_max)[,-1]
  
  # ==== RowID ====
  preds_w_lang$RowID = pred_data$RowID
  preds_wo_lang$RowID = pred_data$RowID
  
  # ==== Few repeating IDs ====
  problem_ids = preds_w_lang %>%
    count(RowID) %>%
    filter(n>1) %>%
    select(RowID) %>%
    unlist()

  length(problem_ids) # 159

  pred_data = pred_data %>%
    filter(!RowID %in% problem_ids)

  preds_w_lang = preds_w_lang %>%
    filter(!RowID %in% problem_ids)

  preds_wo_lang = preds_wo_lang %>%
    filter(!RowID %in% problem_ids)
  
  # ==== Extract source from RowID ====
  pred_data = pred_data %>%
    mutate(Source = str_replace_all(RowID, "[0-9]", ""))
  
  # ==== Test accuracy at thresholds ====
  thresholds = seq(from = 0.01, to = 0.99, by = 0.01)
  
  sum_stats = foreach(thr = thresholds) %do% {
    cat("Threshold: ", thr, "         \r")
    list(
      summary_tables_w_lang = preds_w_lang %>% 
        Pred_to_pred(threshold = thr) %>% 
        Run_tests(pred_data) %>% 
        mutate(thr = thr) %>% 
        mutate(lang_info = TRUE),
      
      summary_tables_wo_lang = preds_wo_lang %>% 
        Pred_to_pred(threshold = thr) %>% 
        Run_tests(pred_data) %>% 
        mutate(thr = thr) %>% 
        mutate(lang_info = FALSE)
    ) %>% bind_rows()
  } %>% bind_rows()
  
  sum_stats %>% write_csv2(paste0("Data/Summary_data/Model_performance", model_name, ".csv"))
  
  return(sum_stats)
  
}





