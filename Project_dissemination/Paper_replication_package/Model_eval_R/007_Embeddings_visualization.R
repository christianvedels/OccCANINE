# Embedding visualisation
# Updated:    2025-07-13
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Reads predicted labels and produces evaluation results

# ==== Libraries ====
library(tidyverse)  
library(plotly)     
source("Data_cleaning_scripts/000_Functions.R")

# ==== Load Data ==== 
key = read_csv("Data/Key.csv")

tsne_data_2d = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/tsne_results.csv")
tsne_data_3d = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/tsne_results_3d.csv")

# ==== Clean data ====
clean_data = function(df){
    df = df %>%
        mutate(
            first_digit = substr(hisco_1, 1, 1),
        ) %>%
        mutate(
            first_digit = ifelse(first_digit == "-", -1, first_digit)
        ) %>%
        left_join(key, by = c("hisco_1" = "hisco")) %>%
        distinct()
    
    return(df)
}

tsne_data_2d = tsne_data_2d %>% clean_data()
tsne_data_3d = tsne_data_3d %>% clean_data()

# ==== Function Definitions ====
construct_label = function(x){
  paste0(
    "Input: '", x$occ1, "'", " (lang: ", x$lang, ")\n",
    "HISCO: ", x$hisco_1,", Description: ", x$en_hisco_text
  )
}


plot_emb = function(tsne_data_2d, tsne_data_3d, name){
    # Ensure the data has the correct structure
    tsne_data_2d$label = construct_label(tsne_data_2d)
    tsne_data_3d$label = construct_label(tsne_data_3d)

    # 2D Visualization using ggplot2
    p1 = tsne_data_2d %>% 
        ggplot(aes(x=V1, y=V2, col = first_digit)) +
        geom_point(alpha=0.7) +
        labs(
        paste(name, '2D t-SNE Visualization'),
        col = "First HISCO\ndigit",
        shape = "First HISCO\ndigit",
        x = "t-SNE Dimension 1",
        y = "t-SNE Dimension 2"
        ) +
        theme_bw() + 
        theme(legend.position = "bottom") 
  
    print(p1)

    # Save 2D plot as PNG
    ggsave(paste0("Project_dissemination/Paper_replication_package/Figures/tsne/",name, "_tsne_2d.pdf"), plot=p1, width=dims$width, height=dims$height, dpi = 600)
    ggsave(paste0("Project_dissemination/Paper_replication_package/Figures/tsne/",name, "_tsne_2d.png"), plot=p1, width=dims$width, height=dims$height, dpi = 600)
    
    # Interactive Visualization using plotly
    p2d = plot_ly(tsne_data_2d, x = ~V1, y = ~V2, text = ~label, mode = 'markers', color = ~first_digit, marker = list(opacity=0.7)) %>%
        layout(title = paste0("Embedding space (t-sne)<br><sup>",name,"</sup>"))
    fname = paste0("Project_dissemination/Paper_replication_package/Figures/tsne/",name, "Interactive_tsne_2d.Rdata") 
    save(p2d, file = fname)
    
    
    # 3D Interactive Visualization using plotly
    p3d = plot_ly(tsne_data_3d, x = ~V1, y = ~V2, z = ~V3, text = ~label, mode = 'markers', color = ~first_digit, marker = list(opacity=0.7)) %>%
        layout(title = paste0("Embedding space (t-sne)<br><sup>",name,"</sup>"))
    fname = paste0("Project_dissemination/Paper_replication_package/Figures/tsne/",name, "Interactive_tsne_3d.Rdata") 
    save(p3d, file = fname)
  
    return(0) 
}

# ==== Main Execution ====
plot_emb(tsne_data_2d, tsne_data_3d, "embeddings")
