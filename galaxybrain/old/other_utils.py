def _load_data(files = None, show_dir = False):
    """ 
    EXPERIMENTAL
    Load data from your data folder
    If show_dir is True, displays a list of files in your directory.  Omits the extension.
    If show_dir is False, assumes input is a string of format 'file1, file2 ...'
    Only really works for .csv for now
    """
    l = os.listdir('../data')
    l.remove('.DS_Store')
    if show_dir == True:
        for i, x in enumerate(l):
            x = x[:x.index('.')]
            l[i] = x
        print('Files in ../data: ', l)
        files = input("Enter file name(s). Separate with commas for multiple files.  Enter 'all' (w/o quotes)  to load all. \n")
        print('>Uploading...')
        if files == 'all':
            for z in l:
                globals()[z] = pd.read_csv("../data/"+ z +".csv", index_col = 0)
            print('>Done.')

        else:
            files = files.split(',')
            files = [i.translate({ord(i):None for i in ' '}) for i in files] # takes out spaces

            for z in files:
                globals()[z] = pd.read_csv("../data/"+ z +".csv", index_col = 0)
            print('>Done.')

    if show_dir == False:
        print('>Uploading...')
        if files == 'all':
            for z in l:
                globals()[z[:z.index('.')]] = pd.read_csv("../data/"+ z, index_col = 0)
            print('>Done.')
        else:
            files = files.split(',')
            files = [i.translate({ord(i):None for i in ' '}) for i in files] # takes out spaces
            for z in files:
                globals()[z] = pd.read_csv("../data/"+ z +".csv", index_col = 0)
            print('>Done.')