import os
import csv

root = r"G:\Projects - Final\cattle_breed_project\data\raw"
out_csv = r"G:\Projects - Final\cattle_breed_project\annotations\labels.csv"

rows=[]
for animal_type in os.listdir(root):
  type_path=os.path.join(root,animal_type)
  if not os.path.isdir(type_path):
    continue
  for breed in os.listdir(type_path):
    breed_path=os.path.join(type_path,breed)
    if not os.path.isdir(breed_path):
      continue
    for fname in os.listdir(breed_path):
      if fname.lower().endswith(('.jpg','.jpeg','.png')):
        filepath=os.path.join('data/raw',animal_type,breed,fname)
        rows.append([filepath,breed])

with open(out_csv,'w',newline='') as f:
  writer=csv.writer(f)
  writer.writerow(['filepath','label'])
  writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out_csv}")