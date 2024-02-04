# Mistakes
# Created:    2024-01-16
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Extracts mistakes 
#
# Output:     paste0("Data/Summary_data/Model_performance", model_name, ".csv")
# 
# NOTE: This loads a function to use in another script. 

Nature_of_mistakes = function(model_name = "CANINE", toyrun = FALSE){
  # The function automatically loads previous results
  #
  # model_name:   Name of model in 'Predictions'
  # toyrun:       Should this be a quick toyrun with only the first 100 rows?
  # overwrite:    Should previous results be overwritten?
  
  # ==== Libraries ====
  require(tidyverse)
  require(foreach)
  source("Model_evaluation_scripts/000_Functions.R")
  
  # ==== Read data ====
  key = read_csv("Data/Key.csv")
  
  n_max = ifelse(toyrun, 100, Inf)
  
  pred_data = read_csv(paste0("Data/Predictions/Predictions", model_name, "/pred_data.csv"), n_max = n_max)[,-1]
  preds_w_lang = read_csv(paste0("Data/Predictions/Predictions", model_name, "/preds_w_lang.csv"), n_max = n_max)[,-1]
  preds_wo_lang = read_csv(paste0("Data/Predictions/Predictions", model_name, "/preds_wo_lang.csv"), n_max = n_max)[,-1]
  
  # ==== RowID ====
  preds_w_lang$RowID = pred_data$RowID
  preds_wo_lang$RowID = pred_data$RowID
  
  # ==== Fix a few repeating RowID ====
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
  
  # ==== Find mistakes ====
  Get_wrong_cases(preds_w_lang, pred_data, key) %>% Plot_wrong(name = paste0(model_name, "_preds_w_lang"))
  Get_wrong_cases(preds_wo_lang, pred_data, key) %>% Plot_wrong(name = paste0(model_name, "_preds_wo_lang"))
  
}





