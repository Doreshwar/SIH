from PIL import Image
import os

root='data/raw'
bad=[]
for dirpath, _, files in os.walk(root):
  for f in files:
    if f.lower().endswith(('.jpg','.jpeg','.png')):
      p=os.path.join(dirpath,f)
      try:
        Image.open(p).verify()
      except Exception:
        bad.append(p)
print("Broken images:",len(bad))
for b in bad[:20]:
  print(b)