#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 18:16:47 2017

@author: fmercermoss
"""

import os
import sys
from os import rename
import requests
import shutil
from pdb import set_trace as bp
__author__ = 'Felix Mercer Moss'



def main(argv):
    fileList = []
    fileSize = 0
    folderCount = 0
    rootdir = "/Users/fmercermoss/ProjectsMAC/BBC_test/lfw"
    maleFolder = "/Users/fmercermoss/ProjectsMAC/BBC_test/faces/male"
    femaleFolder = "/Users/fmercermoss/ProjectsMAC/BBC_test/faces/female"
    count = 0
    tmp = ""
    print("Running file...")
    for root, subFolders, files in os.walk(rootdir):
        folderCount += len(subFolders)
        for file in files:
            f = os.path.join(root,file)
            fileSize = fileSize + os.path.getsize(f)
            fileSplit = file.split("_")
            fileList.append(f)
            count += 1
   
            malePath = "%s/%s" % (maleFolder,file)
            femalePath = "%s/%s" % (femaleFolder,file)
            print("Analysing "+ file)    
            if os.path.exists(malePath) or os.path.exists(femalePath):
                print("Skipping...")
                continue
                
            
            if count == 1:
                result = requests.get("https://api.genderize.io/?name=%s" % fileSplit[0])
                result = result.json()
                tmp = fileSplit[0]
            elif tmp != fileSplit[0]:
                result = requests.get("https://api.genderize.io/?name=%s" % fileSplit[0])
                result = result.json()
                tmp = fileSplit[0]
            else:
                tmp = fileSplit[0]

            try:
                
                
                    
                if float(result['probability']) > 0.9:
                    if result['gender'] == 'male':
                        shutil.copyfile(f,"%s/%s" % (maleFolder,file))
                    elif result['gender'] == 'female':
                        shutil.copyfile(f,"%s/%s" % (femaleFolder,file))
            except Exception as e:
                print(result['name'])

            print(count)



if __name__ == "__main__":
    main(sys.argv)
