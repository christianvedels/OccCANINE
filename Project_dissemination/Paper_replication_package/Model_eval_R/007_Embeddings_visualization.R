# Embedding visualisation
# Updated:    2025-07-13
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Reads predicted labels and produces evaluation results

# ==== Libraries ====
library(tidyverse)  
library(Rtsne)      
library(plotly)     
source("Data_cleaning_scripts/000_Functions.R")

# ==== Load Data ==== 
key = read_csv("Data/Key.csv")

embeddings = read_csv("Project_dissemination/Paper_replication_package/Data/big_files/embeddings_test.csv")
embeddings_null = read_csv("Project_dissemination/Paper_replication_package/Data/big_files/embeddings_test_null.csv")

# ==== Clean data ====
clean_data = function(df){
    df = df %>%
        mutate(
            first_digit = substr(hisco_1, 1, 1),
        ) %>%
        mutate(
            first_digit = ifelse(hisco_1 == "-1", -1, first_digit)
        ) %>%
        left_join(key, by = c("hisco_1" = "hisco")) %>%
        distinct()
    
    return(df)
}

embeddings = embeddings %>% clean_data()
embeddings_null = embeddings_null %>% clean_data()

# ==== Function Definitions ====
construct_label = function(x){
  paste0(
    "Input: '", x$occ1, "'", " (lang: ", x$lang, ")\n",
    "HISCO: ", x$hisco_1,", Description: ", x$en_hisco_text
  )
}

# visualize_embeddings: Function to perform t-SNE and generate plots
run_tsne = function(embeddings, d = 2){ 
    # t-SNE Computation
    set.seed(20)  # for reproducibility
    tmp = embeddings %>% select(-c(occ1, hisco_1, first_digit, lang, en_hisco_text)) 
    tsne_results = Rtsne(tmp, dims=d, perplexity=30, theta=0.0, check_duplicates=FALSE, max_iter = 10000)
    
    # Merge t-SNE results with labels
    tsne_data = as.data.frame(tsne_results$Y)
    tsne_data$first_digit = embeddings$first_digit
    tsne_data$label = construct_label(embeddings)
    
    return(
        tsne_data
    )
}

plot_emb = function(embeddings, name){
    # Construct fname and check if it exists
    fname = paste0("Project_dissemination/Paper_replication_package/Figures/tsne/", name, "_tsne_3d.Rdata")
    if (file.exists(fname)) {
        message(paste("File", fname, "already exists. Skipping t-SNE computation."))
        load(fname)
        return(0)
    }
    
    tsne_data_2d = embeddings %>%
        run_tsne()

    tsne_data_3d = embeddings %>%
        run_tsne(d = 3)

    # 2D Visualization using ggplot2
    p1 = tsne_data_2d %>% 
        ggplot(aes(x=V1, y=V2, col = first_digit, shape = first_digit)) +
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
plot_emb(embeddings, "embeddings")
plot_emb(embeddings_null, "embeddings_null")