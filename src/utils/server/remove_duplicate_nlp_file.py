import os
import sys
<<<<<<< HEAD
rootdir = '/srv/local/work/swang141/Sheng_repo/data/Pubmed/word_sentences/new_words/'
=======
rootdir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/Pubmed/word_sentences/new_words/'
>>>>>>> 73a411770798eff37b48abc4aced06e15cd28c8f
for iter in range(10):
    ct = 1
    if iter > 1 and fdel == 0:
        break
    fdel = 0
    for subdir, dirs, files in os.walk(rootdir):
<<<<<<< HEAD
        #print files
=======
>>>>>>> 73a411770798eff37b48abc4aced06e15cd28c8f
        if len(files)==0:
            continue
        min_file = 1000000
        file_name = ''
        for file in files:
            #print file
            if len(file) < min_file:
                min_file = len(file)
                file_name = file
        print file_name,min_file
        for file in files:
            if file!=file_name and len(file) > len(file_name):
                file_path = os.path.join(subdir, file)
                os.remove(file_path)
                fdel += 1
