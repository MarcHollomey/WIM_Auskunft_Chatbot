import csv
from bs4 import BeautifulSoup
import os

path = os.getcwd()
print("Current Directory", path)

abs = os.path.abspath(os.path.join(path, os.pardir))

dataset = os.path.join(abs, 'input/datasets/oewb/OEWB_statische_Auszuege_07-2024.xml')

# Load the XML document
with open(dataset, 'r') as f:
    data = f.read()

# Parse the XML document with BeautifulSoup
soup = BeautifulSoup(data, 'xml')

# Find all the entries in the document
entries = soup.find_all('entry')

# Create a list to store the data
data_list = []

for entry in entries:
    oesterreichisch = entry.find('form', {'type': 'lemma'}).find('orth').text

    sense = entry.find('sense')
    if sense is not None:
        def_elem = sense.find('def')
        if def_elem is not None:
            deutsch = def_elem.text
        else:
            deutsch = ''
    else:
        xr_elem = entry.find('xr', {'type': 'equivalence'})
        if xr_elem is not None:
            ref_elem = xr_elem.find('ref')
            if ref_elem is not None:
                deutsch = ref_elem.text
            else:
                deutsch = ''
        else:
            deutsch = ''

    if deutsch:
        data_list.append({'oesterreichisch': oesterreichisch, 'deutsch': deutsch})

output = os.path.join(abs, 'input/datasets/train/oewb_for_sft.csv')
# Write the data to a CSV file
with open(output, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['oesterreichisch', 'deutsch'])
    writer.writeheader()
    writer.writerows(data_list)
