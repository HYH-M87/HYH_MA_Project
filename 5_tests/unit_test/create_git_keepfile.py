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


if __name__ == "__main__":
    start_path = '.'  
    filename = '.keepfile'
    CreateFiles2Dir(start_path, filename)