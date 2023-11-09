# Data from HISCO website
# Created:    2023-11-09
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    This script scrapes and cleans occupational titles from historyofwork.iisg.nl
#
# Output:     Clean tmp version of the data

# ==== Libraries ====
library(tidyverse)
library(rvest)

# ==== Run ====
# Define the URL
url = "https://historyofwork.iisg.nl/detail_hiswi.php?know_id=7983&lang="

# Download and parse the webpage
webpage = read_html(url)

# Extract all the links and their text
links = webpage %>% html_nodes("a")
link_text = links %>% html_text()
link_urls = links %>% html_attr("href")

# Create a data frame to store the results
scraped_data = data.frame(Text = link_text, URL = link_urls)

# Print the first few rows of the data frame
head(scraped_data)
