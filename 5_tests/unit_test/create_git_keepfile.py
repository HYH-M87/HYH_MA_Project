'''
create  file for keeping the dir structure during git push
'''

import os

def CreateFile(directory, filename):
    
    file_path = os.path.join(directory, filename)
    
    if not os.path.isfile(file_path):
        with open(file_path, 'w+') as file:
            file.write('This is a file for keeping the dir structure during git push\n')

def CreateFiles2Dir(start_path, filename):
    for root, dirs, files in os.walk(start_path):
        for dir in dirs:
            CreateFile(os.path.join(root, dir), filename)

def DeleteFiles2Dir(start_path, filename):
    for root, dirs, files in os.walk(start_path):
        if filename in files:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"{e}")

if __name__ == "__main__":
    start_path = '3_configurations/code_configuration/configs'  
    filename = '.keepfile'
    DeleteFiles2Dir(start_path, filename)