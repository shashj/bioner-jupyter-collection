import os
import pickle
from tqdm import tqdm
from lxml import etree

# Read the XML file
file_path = '../datasets/i2b2/PHI_Processed_data/deid_surrogate_train_all_version2.xml'
with open(file_path, 'r') as file:
    xml_content = file.read()

# Define the output directory
output_dir = '../datasets/i2b2/train_jsons'

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionary to hold all records
all_records = {}

# Parse the XML content
parser = etree.XMLParser(recover=True)
root = etree.fromstring(xml_content, parser)

# Find all RECORD elements
records = root.findall('.//RECORD')

# Function to get the start and end offset of an element
# Function to get the start and end offset of an element within its parent TEXT element

def find_all_occurrences(full_text, element_text):
    offsets = []
    start = 0
    while True:
        start = full_text.find(element_text, start)
        if start == -1:  # substring not found
            return offsets
        offsets.append((start, start + len(element_text)))
        start += 1  # move past the last match

def get_offsets(element, text_content):
    full_text = etree.tostring(element.getparent(), encoding='unicode', method='text')
    element_text = element.text
    #import code; code.interact(local=locals())
    return find_all_occurrences(full_text, element_text)

# Process each RECORD element with a progress bar
for record in tqdm(records, desc="Processing records"):
    record_id = record.get('ID')
    all_records[record_id] = []
    
    # Get the TEXT element of the current RECORD
    text_element = record.find('TEXT')
    if text_element is not None:
        text_content = etree.tostring(text_element, encoding='unicode', method='text')
        
        # Find all PHI tags within the current RECORD
        phi_tags = text_element.findall('.//PHI')
        for phi in phi_tags:
            offsets = get_offsets(phi, text_content)
            for start_offset, end_offset in offsets:
                phi_entry = {
                    'TYPE': phi.get('TYPE'),
                    'TEXT': phi.text,
                    'START_OFFSET': start_offset,
                    'END_OFFSET': end_offset
                }
                #import code; code.interact(local=locals())
                all_records[record_id].append(phi_entry)

# Save all records dictionary to a pickle file
pickle_file_path = os.path.join(output_dir, 'all_records_train.pkl')
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(all_records, pickle_file)

print(f"All records data saved to {pickle_file_path}")