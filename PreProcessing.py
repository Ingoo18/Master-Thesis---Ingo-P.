

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


df1 = pd.DataFrame()
df1['label'] = df['label']
df1['data'] = df['data']

import re

# Define a function to clean the data column using regular expressions
def clean_image_data(entry):
    if isinstance(entry, str):
        entry = re.sub(r"\{'image': '", "", entry)
        entry = re.sub(r"'}", "", entry)
    return entry

# Apply the custom function to the 'data' column
df1['data'] = df1['data'].apply(clean_image_data)


#if df1['data'].dtype != 'object':
    #df1['data'] = df1['data'].astype(str)

#df['data'] = df['data'].str.replace(r"{'image': ", "", regex=True)
#df['data'] = df['data'].str.replace(r"}'", "", regex=True)                              
print(df1)
