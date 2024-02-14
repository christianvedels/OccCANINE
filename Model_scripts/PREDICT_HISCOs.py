# -*- coding: utf-8 -*-
"""
Created on 2024-01-15

@author: christian-vs

Prediction for applications

SETUP:
    - See readme2.md
"""

import os
script_directory = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_directory)

# %% Import necessary modules
from n103_Prediction_assets import Finetuned_model

# %% Load model
model = Finetuned_model(
    name = "CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256"
    )

# %% Example 1
model.predict(
    ["han arbejder primært som borgmestersekretær"], 
    lang = "da", 
    get_dict = True, 
    threshold = 0.11 # Best F1 for Danish
    )

# %% Example 2
model.predict(
    ["denne bager bager den bedste borgmesterkringle"], 
    lang = "da", 
    get_dict = True, 
    threshold = 0.11 # Best F1 for Danish
    )

# %% Example 3
model.predict(
    ["tailor of the finest suits"], 
    lang = "en", 
    get_dict = True, 
    threshold = 0.22 # Best F1 for English
    )

# %% Example 4
model.predict(
    ["the train's fireman"], 
    lang = "en", 
    get_dict = True, 
    threshold = 0.22 # Best F1 for English
    )

# %% Example 5
model.predict(
    ["This guy sweeps the chimney"],
    lang = "en", 
    get_dict = True, 
    threshold = 0.22 # Best F1 for English
    )

# %% Example 6 - 73 random occs at once works instantaneously
x = model.predict(
    ["Works as a blacksmith, forging tools and horseshoes.",
    "Builds and repairs wooden structures as a carpenter.",
    "Creates and repairs wheels and wagons as a wheelwright.",
    "Lays bricks and stones, working as a mason.",
    "Sews and alters clothing, employed as a tailor.",
    "Crafts and mends footwear, known as a shoemaker.",
    "Weaves cloth and textiles, operating as a weaver.",
    "Cultivates crops and raises livestock, farming his land.",
    "Operates a mill for grinding grain, known as a miller.",
    "Bakes bread and pastries, working as a baker.",
    "Butchers animals for meat, known as a butcher.",
    "Catches fish for a living, working as a fisherman.",
    "Navigates and works on ships, employed as a sailor.",
    "Serves in the military, known as a soldier.",
    "Educates children and adults, employed as a teacher.",
    "Leads religious services, known as a clergyman.",
    "Practices medicine, working as a physician.",
    "Cares for the sick and injured, known as a nurse.",
    "Assists with childbirth, working as a midwife.",
    "Buys and sells goods, known as a merchant.",
    "Performs administrative duties, employed as a clerk.",
    "Keeps financial records, working as a bookkeeper.",
    "Operates a printing press, known as a printer.",
    "Publishes books and newspapers, working as a publisher.",
    "Writes for newspapers or magazines, known as a journalist.",
    "Practices law, employed as a lawyer.",
    "Oversees court proceedings, known as a judge.",
    "Engages in political life, working as a politician.",
    "Performs government duties, employed as a civil servant.",
    "Represents his country abroad, known as a diplomat.",
    "Designs machinery and structures, employed as an engineer.",
    "Measures land, working as a surveyor.",
    "Designs buildings, known as an architect.",
    "Creates artworks, employed as an artist.",
    "Shapes sculptures, known as a sculptor.",
    "Captures images with a camera, working as a photographer.",
    "Performs or composes music, known as a musician.",
    "Writes musical works, working as a composer.",
    "Acts in plays and films, known as an actor.",
    "Writes plays or scripts, employed as a playwright.",
    "Manages library collections, known as a librarian.",
    "Gives talks on various subjects, working as a lecturer.",
    "Conducts scientific research, known as a scientist.",
    "Studies chemicals and their reactions, employed as a chemist.",
    "Investigates the laws of nature, working as a physicist.",
    "Studies stars and planets, known as an astronomer.",
    "Studies living organisms, employed as a biologist.",
    "Studies the Earth's physical structure, known as a geologist.",
    "Writes about the past, working as a historian.",
    "Explores the fundamental nature of knowledge, known as a philosopher.",
    "Extracts minerals from the earth, employed as a miner.",
    "Cuts down trees, working as a lumberjack.",
    "Herds cattle, known as a cowboy.",
    "Tends to sheep, employed as a shepherd.",
    "Produces milk and dairy products, known as a dairyman.",
    "Cultivates gardens, working as a gardener.",
    "Grows and sells flowers, known as a florist.",
    "Makes and sells wine, employed as a vintner.",
    "Brews beer, working as a brewer.",
    "Distills spirits, known as a distiller.",
    "Processes leather, employed as a tanner.",
    "Dyes fabrics and materials, working as a dyer.",
    "Makes and sells hats, known as a hatter.",
    "Designs and sells jewelry, employed as a jeweler.",
    "Repairs watches and clocks, known as a watchmaker.",
    "Cuts hair and shaves beards, working as a barber.",
    "Styles hair and wigs, known as a hairdresser.",
    "Creates and sells perfumes, employed as a perfumer.",
    "Makes and sells soap, working as a soapmaker.",
    "Produces and sells candles, known as a candlemaker.",
    "Installs and repairs glass, employed as a glazier.",
    "Shapes clay into pottery, known as a potter.",
    "Blows glass to form objects, working as a glassblower."],
    lang = "en", 
    threshold = 0.22 # Best F1 for English
    )

print(x)

# %% Example 7 - 50k are still very fast
import pandas as pd
df = pd.read_csv("../Data/Application_data/Copenhagen Burial Records/transcribed_sources/CBP/CBP_20210309.csv")
df = df[["positions"]]
df = df[df["positions"].notnull()]
df = df.sample(50000)
print(f"Producing HISCO codes for {df.shape[0]} observations")
print(f"Estimated human ours saved: {df.shape[0]*10/60/60} hours")
model.verbose = True # Set updates to True
x = model.predict(
    df["positions"],
    lang = "da",
    threshold = 0.11,
    get_dict = True
    )