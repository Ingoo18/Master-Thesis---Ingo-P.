import os
import shutil
import pandas as pd

# Specify the directory containing your json file with the Lables/Image Names
file_path= r'directory'
df = pd.read_json(file_path)


pd.set_option('display.max_colwidth', None)

#Label Extraction

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

#New Data Frame
df1 = pd.DataFrame()
df1['label'] = df['label']
#Convert data to str., exclude everything before study, remove last two characters
df1['data'] = df['data'].astype(str).str.split('study', n=1).str[-1].apply(lambda x: 'study' + x[:-2])

df1.columns = df1.columns.astype(str)

df1_cleaned = df1.dropna()

#Check for Difference, NA actually dropped
#print(df1.isna().sum())


###Sort Pictures into Folder###

# Base directory where the pictures are curretnly stored
base_dir = r'directory'  #Update

# Directory where sorted pictures will be stored
output_base_dir = r'directory'  # Update

# Ensure the output base directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Function to move the pictures into folders
def move_pictures(row):
    label = row['label']
    data = row['data']
    src_path = os.path.join(base_dir, data)
    dest_dir = os.path.join(output_base_dir, label)
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    dest_path = os.path.join(dest_dir, data)
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"File {src_path} does not exist")

#Applying the function to each row in the dataframe
df1_cleaned.apply(move_pictures, axis=1)
