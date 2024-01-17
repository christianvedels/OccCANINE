# Embedding visualisation
# Created:    2024-01-17
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Reads predicted labels and produces evaluation results
#
# Output:     paste0("Data/Summary_data/Model_performance", model_name, ".csv")

# ==== Libraries ====
library(tidyverse)  
library(Rtsne)      
library(plotly)     
source("Model_evaluation_scripts/000_Functions.R")

# ==== Load Data ==== 
key = read_csv("Data/Key.csv")

n_max = 10000
embeddings_w_lang = read_csv('Data/Predictions/PredictionsCANINE/embeddings_w_lang.csv', n_max = n_max)[,-1]
embeddings_w_lang_base = read_csv('Data/Predictions/PredictionsCANINE/embeddings_w_lang_base.csv', n_max = n_max)[,-1]
embeddings_wo_lang = read_csv('Data/Predictions/PredictionsCANINE/embeddings_wo_lang.csv', n_max = n_max)[,-1]
embeddings_wo_lang_base = read_csv('Data/Predictions/PredictionsCANINE/embeddings_wo_lang_base.csv', n_max = n_max)[,-1]
pred_data = read_csv('Data/Predictions/PredictionsCANINE/pred_data.csv', n_max = n_max)

# ===== Prepare Data ====
pred_data = pred_data %>% 
  mutate( # Only one hisco code
    only1 = is.na(hisco_2)*is.na(hisco_3)*is.na(hisco_4)*is.na(hisco_5)
  ) %>% 
  left_join(key, by = c(hisco_1="hisco"))

prep_emb = function(x){
  x$occ1 = pred_data$occ1
  x$hisco_1 = pred_data$hisco_1
  x$only1 = pred_data$only1
  x$lang = pred_data$lang
  x$en_hisco_text = pred_data$en_hisco_text
  x = x %>% 
    mutate(
      hisco_1 = Fix_HISCO(hisco_1)
    ) %>% 
    filter(only1==1) %>% 
    mutate(
      first_digit = substr(hisco_1, 1, 1)
    ) %>% 
    mutate(
      first_digit = ifelse(first_digit=="-", "-1", first_digit)
    )
  
  x = x %>% # Only unique values
    group_by(occ1, lang) %>% 
    summarise_all(~.x[1]) %>% 
    ungroup()
  
  return(x)
}


# ==== Function Definitions ====
construct_label = function(x){
  paste0(
    "Input: '", x$occ1, "'", " (lang: ", x$lang, ")\n",
    "HISCO: ", x$hisco_1,", Description: ", x$en_hisco_text
  )
}

# visualize_embeddings: Function to perform t-SNE and generate plots
visualize_embeddings = function(embeddings, name) {
  # t-SNE Computation
  set.seed(20)  # for reproducibility
  tmp = embeddings %>% select(-c(occ1, hisco_1, only1, first_digit, lang, en_hisco_text)) 
  tsne_results = Rtsne(tmp, dims=2, perplexity=30, theta=0.0, check_duplicates=FALSE, max_iter = 10000)
  tsne_results3d = Rtsne(tmp, dims=3, perplexity=30, theta=0.0, check_duplicates=FALSE, max_iter = 10000)
  
  # Merge t-SNE results with labels
  tsne_data = as.data.frame(tsne_results$Y)
  tsne_data$first_digit = embeddings$first_digit
  tsne_data$label = construct_label(embeddings)
  
  tsne_data3d = as.data.frame(tsne_results3d$Y)
  tsne_data3d$first_digit = embeddings$first_digit
  tsne_data3d$label = construct_label(embeddings)
  
  
  
  # 2D Visualization using ggplot2
  p1 = tsne_data %>% 
    ggplot(aes(x=V1, y=V2, col = first_digit)) +
    geom_point(alpha=0.7) +
    labs(
      paste(name, '2D t-SNE Visualization'),
      col = "First HISCO\ndigit"
    ) +
    theme_bw()
  
  p1
  # Save 2D plot as PNG
  ggsave(paste0("Eval_plots/Embedding_tsne/",name, "_tsne_2d.png"), plot=p1, width=10, height=8)
  
  # Interactive Visualization using plotly
  p2d = plot_ly(tsne_data, x = ~V1, y = ~V2, text = ~label, mode = 'markers', color = ~first_digit, marker = list(opacity=0.7)) %>%
    layout(title = paste0("Embedding space (t-sne)<br><sup>",name,"</sup>"))
  fname = paste0("Eval_plots/Embedding_tsne/",name, "Interactive_tsne_2d.Rdata") 
  save(p2d, file = fname)
  
  
  # 3D Interactive Visualization using plotly
  p3d = plot_ly(tsne_data3d, x = ~V1, y = ~V2, z = ~V3, text = ~label, mode = 'markers', color = ~first_digit, marker = list(opacity=0.7)) %>%
    layout(title = paste0("Embedding space (t-sne)<br><sup>",name,"</sup>"))
  fname = paste0("Eval_plots/Embedding_tsne/",name, "Interactive_tsne_3d.Rdata") 
  save(p3d, file = fname)
}

# ==== Main Execution ====
embeddings_w_lang %>% 
  prep_emb() %>% 
  visualize_embeddings("CANINE finetuned (w. lang)")

embeddings_w_lang_base %>% 
  prep_emb() %>% 
  visualize_embeddings("CANINE baseline (w. lang)")

embeddings_wo_lang %>% 
  prep_emb() %>% 
  visualize_embeddings("CANINE finetuned (wo. lang)")

embeddings_wo_lang_base %>% 
  prep_emb() %>% 
  visualize_embeddings("CANINE baseline (wo. lang)")



# embeddings_w_lang_base$occ1 = pred_data$occ1
# embeddings_wo_lang$occ1 = pred_data$occ1
# embeddings_wo_lang_base$occ1 = pred_data$occ1



