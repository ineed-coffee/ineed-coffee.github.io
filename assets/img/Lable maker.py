import os
import tkinter
from tkinter import filedialog

cur_path=os.getcwd()
root = tkinter.Tk()
root.withdraw()
files =  filedialog.askopenfilenames(parent=root,initialdir =cur_path,title = "Choose files to be renamed")
print(f'{len(files)}files selected')
print()
outpath =  filedialog.askdirectory(parent=root,initialdir=cur_path,title='Please select a directory')
prefix= input("Enter prefix : ")

for i,file in enumerate(files):
    os.rename(file,os.path.join(outpath,prefix+str(i+1)+".png"))

print(f'{len(files)}files renamed')
print()
