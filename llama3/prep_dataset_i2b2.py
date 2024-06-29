import xml.etree.ElementTree as ET
import json
import os

# Read the XML file
file_path = '../datasets/i2b2/PHI_Processed_data/deid_surrogate_train_all_version2.xml'
tree = ET.parse(file_path)
root = tree.getroot()

# Initialize dictionary to hold the JSON data
data = {}

# Extract the record ID
record_id = root.find('.//RECORD').get('ID')
data[record_id] = []

# Find all PHI tags and append to the list
phi_tags = root.findall('.//PHI')
for phi in phi_tags:
    phi_entry = {
        'TYPE': phi.get('TYPE'),
        'TEXT': phi.text
    }
    data[record_id].append(phi_entry)

# Convert to JSON
json_data = json.dumps(data, indent=4)


# Define the output directory
output_dir = '../datasets/i2b2/train_jsons'

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

output_file_path = os.path.join(output_dir, f"{record_id}.json")
with open(output_file_path, 'w') as json_file:
    json_file.write(json_data)

print(f"JSON data saved to {output_file_path}")