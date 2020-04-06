# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:16:18 2020

@author: gutia
"""
import os
import time
import shutil

#read file function
def readFile(filename):
    filehandle = open(filename)
    print (filehandle.read())
    filehandle.close()

#prepare to create a file directory management function
fileDir = os.path.dirname(os.path.realpath('__file__'))
print (fileDir)

#File diretory management function
def file_name(filename):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    fileName = os.path.join(fileDir, 'Anaconda3\\',filename)
    return fileName

#main program
def main(): 
    shutil.copy2(file_name('\\dst\\dir\\newname.txt')) # complete target filename given
    shutil.copy2(file_name('\\dst\\dir\\file.ext')) # target filename is /dst/dir/file.ext

#timing control
starttime=time.time()
timeout = time.time() + (60*60*12)  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(15*60 - ((time.time() - starttime) % 15.0*60)) # orignial 300=5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()