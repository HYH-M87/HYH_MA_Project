for i in range(10):
    with open("kk.txt","a+") as f:
        f.writelines([f"{i}"])