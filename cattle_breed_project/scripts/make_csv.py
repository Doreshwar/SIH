import os
import pandas as pd


# Base directory = parent of scripts folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
RAW_DATASET = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "annotations")
CSV_FILE = os.path.join(OUTPUT_DIR, "breed_labels.csv")

# Make sure annotations folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

# Walk through subdirectories (CATTLE/BUFFALO → actual breed folders)
for species_folder in os.listdir(RAW_DATASET):
    species_path = os.path.join(RAW_DATASET, species_folder)
    if not os.path.isdir(species_path):
        continue

    for folder in os.listdir(species_path):
        folder_path = os.path.join(species_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # Example: INDIA_CATTLE_0800_AMRITMAHAL_03001
        parts = folder.split("_")

        if len(parts) >= 4:
            species = parts[1].lower()        # cattle / buffalo
            code = parts[2]                   # numeric code
            breed_name = parts[3].lower()     # breed name
        else:
            species, code, breed_name = "unknown", "0000", folder.lower()

        rows.append({
            "folder_name": folder,
            "breed_name": breed_name,
            "accession_number": code,
            "species": species
        })

# Save as CSV
df = pd.DataFrame(rows)
df.to_csv(CSV_FILE, index=False)

print(f"✅ Saved {CSV_FILE} with {len(rows)} entries.")
