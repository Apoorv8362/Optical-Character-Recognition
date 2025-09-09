#!/usr/bin/env python
# coding: utf-8

# # Question 1: OCR

# # Necessary installations for OCR

# In[68]:


#Libraries to install dependencies

#!pip install pytesseract pillow pandas numpy
#!pip install pillow


# In[69]:


#Check for required packages

#import pillow
def install_required_packages():
    required_packages = ['pytesseract', 'pandas', 'numpy']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            


# # Libraries used

# In[70]:


import os
import re
import sys
import subprocess
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
from datetime import datetime


# # Check for tesseract Installation

# In[71]:


def check_and_install_tesseract():
    
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text = True)
        
        if result.returncode == 0:
            print("Tesseract is already installed")
            return True
    except FileNotFoundError:
        pass

    print("Tesseract not found. Installing via Homebrew...")
    try:
        
        subprocess.run(['brew', '--version'], capture_output = True, check = True)
        
        print("Installing tesseract....")
        result = subprocess.run(['brew', 'install', 'tesseract'], capture_output = True, text = True)
        
        if result.returncode ==0:
            print("Tesseract installed succesfuuly!")
            return True
        else:
            print("Error Installing tesseract;", result.stderr)
            return False
    except(FileNotFoundError, subprocess.CalledProcessError):
        print("\nHomebrew not found. Please install tesseract manually:")
        print("1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("2. Install tesseract: brew install tesseract")
        print("3. Or download from: https://github.com/tesseract-ocr/tesseract")
        return False


# In[72]:


def setup_tesseract():
    possible_paths = [
        '/opt/homebrew/bin/tesseract',  
        '/usr/local/bin/tesseract',     
        '/usr/bin/tesseract',           # System installation
        'tesseract'                     # If it's in PATH
        '/Users/apoorvsharma/anaconda3/lib/python3.11/site-packages/pytesseract/pytesseract.py'
    ]

    for path in possible_paths:
        if os.path.exists(path) or path == "tesseract":
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"Using tesseract from: {path}")
            return True

    print("Tesseract not found. Please install or update the path.")
    return False


# # Function to extract text from Image 

# In[73]:


def extract_text_from_image(image_path):

  try:
    image = Image.open(image_path)

    #if image.mode != 'RBG':
    #    image.convert('RBG')
        
    image = image.convert('L')
    
    width, height = image.size
    image = image.resize((width * 2, height * 2), Image.LANCZOS)

    image = image.point(lambda x: 0 if x<128 else 255, '1')

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    text = pytesseract.image_to_string(image, config = custom_config)

    return text

  except Exception as e:
    print(f"Error processing {image_path}: {e}")
    return ""


# 
# # Function to parse characters for the bond index 
# 
# Pattern for the image is pre encoded to ensure correct recognition 
# 
# tuple is used to handle null values and replace 0 in it's place to avoid OCR from inputting incorrect entries and missing values
# 
# 

# In[74]:


def parse_behltreu_data(text):
    
    pattern = re.compile(
        r'(\d{2}/\d{2}/\d{2})\s+'
        r'(\d{2}/\d{2}/\d{2})?\s*'
        r'(\d+\.\d+)\s+'
        r'(\d+\.\d+)\s*'
        r'(\d{2}/\d{2}/\d{2})?\s*'
        r'(\d+\.\d+)?\s*'
        r'(\d+\.\d+)?'
    )
    
    data = []

    for match in pattern.finditer(text):
        groups = match.groups()
        
        groups = tuple(g if g is not None else None for g in groups)
        
        # First block
        if groups[0] and groups[2] and groups[3]:
            data.append({
                'date': groups[0],
                'bond_price': groups[2],
                'pct_change': groups[3]
            })
        
        # Second block 
        if groups[1]!= '0' and groups[5]!= '0' and groups[6]!= '0':
            data.append({
                'date': groups[1],
                'bond_price': groups[5],
                'pct_change': groups[6]
            })
        
        # Third block 
        if groups[4]!= '0' and groups[5]!= '0' and groups[6]!= '0':
            data.append({
                'date': groups[4],
                'bond_price': groups[5],
                'pct_change': groups[6]
            })

    return data


# 
# # Function to parse characters for the future price 
# 
# Pattern for the image is pre encoded to ensure correct recognition 
# 
# tuple is used to handle null values and replace 0 in it's place to avoid OCR from inputting incorrect entries and missing values
# 
# 

# In[75]:


def parse_ahwm5_data(text):
    
    # Pattern: date followed by one or two prices
    pattern = re.compile(r'(\d{2}/\d{2}/\d{2})\s+(\d+\.\d+)(?:\s+(\d+\.\d+))?')
    data = []
    
    for match in pattern.finditer(text):
        date = match.group(1)
        price1 = match.group(2)
        price2 = match.group(3)
        
        data.append({'date': date, 'future_price': price1})
        if price2:
            # capture the second price for the same date with a separate record if needed
            data.append({'date': date, 'bid_price': price2})
            
    return data


# # Functions to clean the data after OCR Implementation

# In[76]:


# Adding missing data 
def manual_data_addition():
    
    
    manual_data =[
        {'date': '2025-06-20', 'bond_price': '301.1065'},
        {'date': '2025-06-19', 'bond_price': '300.9845'},
        {'date': '2025-06-18', 'bond_price': '301.3431'},
        {'date': '2025-06-17', 'bond_price': '301.4794'},
        {'date': '2025-04-21', 'future_price': np.nan},
        {'date': '2025-05-01', 'future_price': np.nan}
    ]
        
    
    manual_data = pd.DataFrame(manual_data)
    
    manual_data['date'] = pd.to_datetime(manual_data['date'])
    
    
    return manual_data


# In[77]:


#Correcting recognised data

def manual_data_correction():
    
    correct_data =[
         {'date': '2025-07-11', 'bond_price': np.nan},
        {'date': '2025-07-10', 'bond_price': np.nan},
        {'date': '2025-07-09', 'bond_price': np.nan},
        {'date': '2025-07-08', 'bond_price': np.nan},
    ]
    
    df_correct_data = pd.DataFrame(correct_data)
    
    df_correct_data['date'] = pd.to_datetime(df_correct_data['date'])
    
    
    return df_correct_data


# # Function to perform Optical Character Recognition for Bond and Future prices

# In[78]:


def ocr():
    
    #Check for Tesseract packages for OCR Implementation
    print("1. Checking required packages...")
    install_required_packages()
    
    
    # Error for Tesseract Installation
    if not setup_tesseract():
        print("Please install Tesseract OCR first.")
        return

    #Please change the location for the Images to be recognised below
    behltreu_image = '/Users/apoorvsharma/Downloads/Entry Technical Test/behltreu.png'
    ahwm5_image = '/Users/apoorvsharma/Downloads/Entry Technical Test/ahwm5.png'


    behltreu_data = []
    ahwm5_data = []
    
    print(behltreu_data)

    print("\n 1. Attempting OCR Extraction...")
    
    
    if os.path.exists(behltreu_image):
        print(f"Processing {behltreu_image}")
        text = extract_text_from_image(behltreu_image)
        behltreu_data = parse_behltreu_data(text)
        print(f"OCR text for BEHLTREU:\n{text[:1000]}")

        print(f"Extracted {len(behltreu_data)} BEHLTREU records")

    if os.path.exists(ahwm5_image):
        print(f"Processing {ahwm5_image}")
        text = extract_text_from_image(ahwm5_image)
        ahwm5_data = parse_ahwm5_data(text)
        print(f"OCR text for AHWM5:\n{text[:1000]}")

        
        print(f"Extracted {len(ahwm5_data)} AHWM5 records")
        
    print(f"Final data: {len(behltreu_data)} BEHLTREU records, {len(ahwm5_data)} AHWM5 records")
    
    df_behltreu_ocr = pd.DataFrame(behltreu_data)
    df_ahwm5_ocr = pd.DataFrame(ahwm5_data)

    print(df_behltreu_ocr)
    print(df_ahwm5_ocr)
    
    #Merging the ocr datasets 
    
    print("\n 2. Merging data...")
    
    merged_df = pd.merge(df_behltreu_ocr, df_ahwm5_ocr, on = 'date', how='outer')
    
    merged_df = merged_df.groupby('date', as_index = False).agg({'bond_price':'first', 'future_price':'first'})

    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values(by='date')
    merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')

    merged_df = merged_df[['date', 'bond_price', 'future_price']]

    #Manual Data correction for the remaining dataset
    print("\n 3. Manual Data Correction for the remaining data...")
    
    df_manual_data = manual_data_addition()
    
    
    #df_behltreu = pd.concat([df_behltreu_ocr, df_behltreu_manual], ignore_index=True)
    df_merged_combined = pd.concat([merged_df, df_manual_data])
    df_merged_combined['date'] = pd.to_datetime(df_merged_combined['date'])

    df_merged = (
    df_merged_combined
    .sort_values(by=['date'], ascending=[True])
    .groupby('date', as_index=False)
    .agg({
        'bond_price': lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan,
        'future_price': lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan
    })
    )
    
    print("\n Data Summary:")
    print(merged_df.describe())
    
    
    return df_merged


# # OCR extraction
# 

# In[79]:


merged_df = ocr()


# # Data Validation

# In[80]:


merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df = merged_df.sort_values(by='date').reset_index(drop=True)

#Handling Incorrect data interpretation during OCR:
merged_corrected_data = manual_data_correction()

# Ensure 'date' is datetime in both
merged_corrected_data['date'] = pd.to_datetime(merged_corrected_data['date'])

# Set 'bond_price' to NaN for these dates
merged_df.loc[merged_df['date'].isin(merged_corrected_data['date']), 'bond_price'] = np.nan

merged_df['bond_price'] = pd.to_numeric(merged_df['bond_price'], errors='coerce')
merged_df['future_price'] = pd.to_numeric(merged_df['future_price'], errors='coerce')


# In[81]:


print(merged_df)


# # Generating the CSV file for final output

# In[82]:


output_file = "behltreu_ahwm5_data.csv"
merged_df.to_csv(output_file, index=False)

print(f"\n 4. Data saved to {output_file}")
print(f"Total records: {len(merged_df)}")

