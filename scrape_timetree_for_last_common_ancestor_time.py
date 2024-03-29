from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

def get_time_to_ancestor(species_1 = "Homo sapiens", species_2 = "Canis lupus"):
    # Start a new Chrome browser instance
    service = Service()
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    #driver = webdriver.Chrome()
    # Navigate to the Timetree website
    driver.get("http://timetree.org/")
    # Find the input boxes by their respective ids
    taxon_a_input = driver.find_element("id", "taxon-a")
    taxon_b_input = driver.find_element("id", "taxon-b")
    # Enter your desired search terms
    taxon_a_input.send_keys(species_1)  
    taxon_b_input.send_keys(species_2)  
    # Find the search button and click it
    search_button = driver.find_element("id", "pairwise-search-button1")
    search_button.click()
    driver.implicitly_wait(30)
    time.sleep(5)
    
    # Get page source
    result = driver.page_source

    # Parse with BeautifulSoup
    soup = BeautifulSoup(result, 'html.parser') 

    # Get Median Time result
    child_soup = soup.find_all('text')
    time_to_ancestor = -1
    for idx, txt in enumerate(child_soup):
        if 'Median Time:' == txt.text :
            time_to_ancestor = str(child_soup[idx + 1].text)
            time_to_ancestor = float(time_to_ancestor.split(' ')[0])
    
    # Close the browser 
    driver.close()

    return time_to_ancestor

taxon_df = pd.read_csv("../human_bias_data/selected_uniprot_top_species.csv")

for idx, species_1 in enumerate(list(taxon_df.timetree_species.values)):
    print(species_1)
    times_to_ancestor = [None for x in range(idx+1)]
    for specie in tqdm(list(taxon_df.timetree_species.values)[idx+1:]):
        try:
            result = get_time_to_ancestor(species_1 = species_1, species_2 = specie)

        except:
            result = -1
        times_to_ancestor.append(result)
        time.sleep(5)
    taxon_df[f'time to ancestor {species_1}'] = times_to_ancestor
