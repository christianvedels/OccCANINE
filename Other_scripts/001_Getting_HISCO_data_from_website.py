# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:04:50 2023

@author: christian-vs
"""
#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries 
import requests
from bs4 import BeautifulSoup
import pandas as pd
from random import randint
from time import sleep
import pickle

# %% Keeping progress
# Create a set to store checked 'i' values
checked_i_values = set()

# Function to save checked_i_values to a file
def save_checked_values(checked_i_values):
    with open('checked_i_values.pkl', 'wb') as file:
        pickle.dump(checked_i_values, file)

# Function to load checked_i_values from a file
def load_checked_values():
    if os.path.exists('checked_i_values.pkl'):
        with open('checked_i_values.pkl', 'rb') as file:
            return pickle.load(file)
    return set()

# Load checked_i_values at the beginning
checked_i_values = load_checked_values()

# %% Get one

def get_one(i, sleep_max = 3, sleep_min = 1, save_interval=100):
    
    # ==== Setup ====    
    id0 = i
    url = f"https://historyofwork.iisg.nl/detail_hiswi.php?know_id={id0}&lang="
    
    # ==== Check if checked =====
    
    if os.path.isfile(f"../Data/Raw_data/HISCO_website/Many_files/{id0}.csv"):
        print(f"{i} has already been scraped.")
        return 0
    
    # Check if 'i' has already been checked
    if id0 in checked_i_values:
        print(f"{i} has already been checked to be empty.")
        return 0
    
    # ==== Get response ====
    print(f"Trying {i}: {url}")
        
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table in the HTML content
    table = soup.find('table')
    
    # Create an empty list to store the table data
    table_data = []
    
    # Extract the data from each row of the table
    for row in table.find_all('tr'):
        row_data = []
        for cell in row.find_all(['th', 'td']):
            row_data.append(cell.text.strip())
        table_data.append(row_data)
        
    table_data = table_data[-8:]
    
    # ==== Housekeeping ====
    sleep(randint(sleep_min, sleep_max))
    # Update progress saver
    # Add 'i' to the set of checked values
    checked_i_values.add(id0)
    
    # Check if it's time to save checked_i_values
    if len(checked_i_values) % save_interval == 0:
        save_checked_values(checked_i_values)
    
    if(len(table_data)<8):
        print(f'Skipped id {id0}')
        return 0
    
    # Delete 'column' element, which is empty:
    if len(table_data ) > 0 and len(table_data[0]) > 1:
        del table_data[0][1]
    
    # Create a pandas DataFrame from the table data
    df = pd.DataFrame(table_data, columns = ["Type", "Content"])
    
    sleep(randint(1,5))
    
    # Save file 
    fname = f"../Data/Raw_data/HISCO_website/Many_files/{id0}.csv"
    df.to_csv(fname)
    
    print(f"Found data in id {id0}")
    print(df['Content'][0])
    
    return 1

# %% Run it
for i in range(7000, 100000):
    get_one(i)
