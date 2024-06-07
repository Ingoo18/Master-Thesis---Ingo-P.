import os
import shutil
import pandas as pd


file_path= r'C:\Users\ingop\Desktop\ICA_classification\Classification (Labled Data JSON File)\project-2-at-2023-11-24-02-32-0c7745bf.json'
df = pd.read_json(file_path)

#4720 Total

pd.set_option('display.max_colwidth', None)

labels = []

for annotations in df['annotations']:
    label_found = False
    for annotation in annotations:
        if annotation['result'] and 'value' in annotation['result'][0] and 'choices' in annotation['result'][0]['value']:
            choice = annotation['result'][0]['value']['choices'][0]
            labels.append(choice)
            label_found = True
            break  
       
    if not label_found:
        labels.append(None) 

df['label'] = labels
df.label.value_counts()

#Create new Dataframe
df1 = pd.DataFrame()
df1['label'] = df['label']
#Convert data to str., exclude everything before study, remove last two characters
df1['data'] = df['data'].astype(str).str.split('study', n=1).str[-1].apply(lambda x: 'study' + x[:-2])

df1.columns = df1.columns.astype(str)

df1_cleaned = df1.dropna()

#Check for Difference / New df contains 21 less
#print(df1.isna().sum())



###Sort Pictures into Folder###

# Base directory where the pictures are stored
base_dir = r'C:\Users\ingop\Desktop\ICA_classification\_all_images'  # Update with the path to your pictures folder

# Directory where sorted pictures will be placed
output_base_dir = r'C:\Users\ingop\Desktop\ICA_classification\Classification'  # Update with the path to your output folder

# Ensure the output base directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Function to move pictures to the corresponding folders
def move_pictures(row):
    label = row['label']
    data = row['data']
    src_path = os.path.join(base_dir, data)
    dest_dir = os.path.join(output_base_dir, label)
    
    # Create attribute directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Move the picture to the corresponding directory
    dest_path = os.path.join(dest_dir, data)
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"File {src_path} does not exist")

#Apply the function to each row in the dataframe
df1_cleaned.apply(move_pictures, axis=1)
