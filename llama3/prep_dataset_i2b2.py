import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle
import os

# Read the XML file
file_path = '../datasets/i2b2/PHI_Processed_data/deid_surrogate_train_all_version2.xml'
tree = ET.parse(file_path)
root = tree.getroot()

# Define the output directory
output_dir = '../datasets/i2b2/train_jsons'

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionary to hold the JSON data
data = {}

# Find all RECORD elements
records = root.findall('.//RECORD')
all_train_jsons = {}

# Extract the record ID
for record in tqdm(records, desc="Processing records"):
    record_id = record.get('ID')
    data[record_id] = []

    # Find all PHI tags and append to the list
    phi_tags = record.findall('.//PHI')
    for phi in phi_tags:
        phi_entry = {
            'TYPE': phi.get('TYPE'),
            'TEXT': phi.text
        }
        data[record_id].append(phi_entry)

# import code; code.interact(local=locals())


output_file_path = os.path.join(output_dir, "all_records_train.pkl")
with open(output_file_path, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)

print(f"All records data saved to {output_file_path}")