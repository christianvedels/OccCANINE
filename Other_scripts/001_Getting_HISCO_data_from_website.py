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
import json

# %% Keeping progress
# Create a set to store checked 'i' values
# checked_i_values = set()

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

# %%  test_dim
# Tests the dimensions of the response

def test_dim(x):
    rows0 = len(x)
    cols0 = [len(i) for i in x]
    cols1 = cols0[0]
    cols0 = cols0[1:]
    
    if rows0 == 8 and all(c == 2 for c in cols0) and cols1 == 3:
        return "Standard"
    else:
        return "Non standard"

# %% Get one

def get_one(i, sleep_max = 3, sleep_min = 1, save_interval=100):
    # breakpoint()
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
    
    # Test dim
    table_type = test_dim(table_data)
    
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
    
    # ==== Process and save results =====
    if table_type == "Standard":
        # Delete 'column' element, which is empty:
        if len(table_data ) > 0 and len(table_data[0]) > 1:
            del table_data[0][1]
        
        # Create a pandas DataFrame from the table data
        df = pd.DataFrame(table_data, columns = ["Type", "Content"])
                
        # Save file 
        fname = f"../Data/Raw_data/HISCO_website/Many_files/{id0}.csv"
        df.to_csv(fname)
        
        print(f"Found data in id {id0}")
        print(df['Content'][0])
        
        return 1
    
    if table_type == "Non standard":
        # Try to find 'Occupational title'
        print(f"Trying to fix data in {id0}")
        
        # Create an empty list to store the table data
        table_data = []
        
        # Extract the data from each row of the table
        for row in table.find_all('tr'):
            row_data = []
            for cell in row.find_all(['th', 'td']):
                row_data.append(cell.text.strip())
            table_data.append(row_data)
            
        # Find 'Occupational title'
        index_with_occupational_title = [i for i, sublist in enumerate(table_data) if 'Occupational title' in sublist]
        # Check when tupple
        start_index = [i for _, i in enumerate(index_with_occupational_title) if len(table_data[i]) in [2, 3]]
        
        table_data = table_data[start_index[0]:]
        
        # Delete 'column' element, which is empty:
        if len(table_data ) > 0 and len(table_data[0]) > 1:
            del table_data[0][1]
        
        # Try to create a pandas DataFrame from the table data
        try:
            df = pd.DataFrame(table_data, columns = ["Type", "Content"])
                        
            # Save file 
            fname = f"../Data/Raw_data/HISCO_website/Many_files/{id0}.csv"
            df.to_csv(fname)
            
            print(f"Found data in id {id0}")
            print(df['Content'][0])
            return 1
    
        except:
            # Create an empty list to store the table data
            table_data = []
            
            # Extract the data from each row of the table
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['th', 'td']):
                    row_data.append(cell.text.strip())
                table_data.append(row_data)
            
            # Specify the file path where you want to save the JSON file
            fname = f"../Data/Raw_data/HISCO_website/Many_files/Non_standard{id0}.json"

            # Open the file in write mode and use json.dump() to save the list as JSON
            with open(fname, 'w') as json_file:
                json.dump(table_data, json_file)
            
            print(f"Found unfixable non-standard data in id {id0}")
            
            return 0
        
def get_one_again(i, sleep_max = 3, sleep_min = 1, save_interval=100):
    # Second run
    # breakpoint()
    # ==== Setup ====    
    id0 = i
    url = f"https://historyofwork.iisg.nl/detail_hiswi.php?know_id={id0}&lang="
    
    # ==== Check if checked =====
    
    if os.path.isfile(f"../Data/Raw_data/HISCO_website/Many_files/{id0}.csv"):
        print(f"{i} has already been scraped.")
        if id0 in to_rescrape['id'].tolist():
            print("-->But it is on rescrape list")
        else:
            return 0
    
    # Check if 'i' has already been checked
    if id0 in checked_i_values:
        print(f"{i} has already been checked to be empty.")
        if id0 in to_rescrape['id'].tolist():
            print("-->But it is on rescrape list")
        else:
            return 0
    
    # ==== Get response ====
    print(f"Trying {i}: {url}")
        
    response = requests.get(url)
    sleep(randint(sleep_min, sleep_max))
    
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
        
    # Find 'Occupational title'
    index_with_occupational_title = [i for i, sublist in enumerate(table_data) if 'Occupational title' in sublist]
    
    if len(index_with_occupational_title)==0:
        print(f'Found no occupational title in {i}: {url}')
        return 0
    
    # Check when tupple
    start_index = [i for _, i in enumerate(index_with_occupational_title) if len(table_data[i]) in [2, 3]]
    
    table_data = table_data[start_index[0]:]
    
    # Delete 'column' element, which is empty:
    if len(table_data ) > 0 and len(table_data[0]) > 1:
        del table_data[0][1]
    
    # ==== Housekeeping ====
    # Update progress saver
    # Add 'i' to the set of checked values
    checked_i_values.add(id0)
    
    # Check if it's time to save checked_i_values
    if len(checked_i_values) % save_interval == 0:
        save_checked_values(checked_i_values)
    
    if(len(table_data)<8):
        print(f'Skipped id {id0}')
        return 0
        
    # ==== Process and save results =====
        
    # Create a pandas DataFrame from the table data
    df = pd.DataFrame(table_data, columns = ["Type", "Content"])
            
    # Save file 
    fname = f"../Data/Raw_data/HISCO_website/Many_files/{id0}.csv"
    df.to_csv(fname)
    
    print(f"Found data in id {id0}")
    print(df['Content'][0])
    
    return 1

# %% Run it
# for i in range(7722, 100000):
#     get_one(i)


    
# %% Run it again 
to_rescrape = pd.read_csv("../Data/Raw_data/HISCO_website/To_rescrape.csv")
for i in range(7722, 100000):
    get_one_again(i)
