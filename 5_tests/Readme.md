# Testing
Perform some code testing, or carry out some minor operations; if a test or operation proves useful, then place its execution script in the 6_scripts/utils directory.

# The Useful test
## 1.create_git_keepfile.py
To maintain consistency in the project structure when creating a new directory hierarchy that contains several empty directories, a .keep file needs to be created within the empty directories to allow them to be committed via Git.

```
START_PATH: the top-level directory to create keep file
MODE: Choose to create or delete the .keep file； "c" for create and "d" for delete
```

```
usage: bash 6_scripts/utils/keepfile.sh 9_logs c
```