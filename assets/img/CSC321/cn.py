import os
import re

img_path=os.getcwd()
pattern_=re.compile("Lec \d+\.")
files= os.listdir(img_path)
images=[file for file in files if file.endswith(".png")]
for i,img in enumerate(images):
    prefix=pattern_.search(img).group()
    os.rename(os.path.join(img_path,img),os.path.join(img_path,prefix+str(i+1)+".png"))
    
