'''
create  file for keeping the dir structure during git push
'''
import argparse
import os

def CreateFile(directory, filename):
    
    file_path = os.path.join(directory, filename)
    
    if not os.path.isfile(file_path):
        with open(file_path, 'w+') as file:
            file.write('This is a file for keeping the dir structure during git push\n')

def CreateFiles2Dir(start_path, filename):
    for root, dirs, files in os.walk(start_path):
        if len(dirs)==0:
            CreateFile(root,filename)
            break
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

def parse_args():
    parser = argparse.ArgumentParser(description='Create keepfile for dir')
    parser.add_argument(
        '--mode',
        default="c",
        help='create ort delete keepfile',
        choices=['d', 'c'])
    parser.add_argument('--start', help='the dir path to start')

    args = parser.parse_args()

    return args

def main():
    filename = '.keepfile'
    args = parse_args()    
    if args.mode == "d":
        DeleteFiles2Dir(args.start,filename)
    else:
        CreateFiles2Dir(args.start,filename)
if __name__ == "__main__":
    main()