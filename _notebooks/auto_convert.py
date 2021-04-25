import os
import tkinter
from tkinter import filedialog
import subprocess
import shutil
from datetime import datetime
import re

cur_path=os.getcwd()
root = tkinter.Tk()
root.withdraw()
file =  filedialog.askopenfilenames(parent=root,initialdir =cur_path,title = "Choose notebook to convert")
file_name=file[0].split("/")[-1]
file_path=file[0].split(file_name)[0]

extend=".ipynb"
post_path="../_posts/"
img_path="../assets/notebooks/"

if not file_name.endswith(extend):
    print("Not notebook file")
    exit(0)
else:
    file_name=file_name.split(extend)[0]
    print(file_name)

os.rename(file[0],file_path+"tmp"+extend)

response= subprocess.check_call(f"jupyter nbconvert --to markdown tmp.ipynb --config jekyll.py",cwd=file_path)

os.rename(file_path+"tmp"+extend,file[0])

now = datetime.now()
formatted = f"{now.year}-{str(now.month).zfill(2)}-{str(now.day).zfill(2)}-"+file_name.replace(" ","-")+".md"
os.rename(file_path+"tmp.md",file_path+formatted)
shutil.move(file_path+formatted,"../_posts/"+formatted)

if os.path.isdir("/tmp_files"):
    images = os.listdir("tmp_files")
    for i,img in enumerate(images):
        shutil.move(file_path+"tmp_files/"+img,"../assets/notebooks/"+img)
    os.rmdir(r"./tmp_files")

pattern=re.compile(r'''[-]{3}\n[a-zA-Z0-9_: \n"\-]*[---][ ]*[\n]''')
front_matter=f'''---\ntitle: {file_name}\nauthor: INEED COFFEE\ndate: {str(now.year)}-{str(now.month).zfill(2)}-{str(now.day).zfill(2)} 14:00:00 +0800\ncategories: [Pytorch101]\ntags: [colab]\ntoc: true\ncomments: true\ntypora-root-url: ../\n---\n'''
with open("../_posts/"+formatted,"r",encoding="utf-8") as f:
    txt=f.read()

subbed = pattern.sub(front_matter,txt)

with open("../_posts/"+formatted,"w",encoding="utf-8") as f:
    f.write(subbed)
