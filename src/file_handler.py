import glob 

def get_filenames_in_folder(path, format='wav'):
    if format == 'wav':
        filenames = glob.glob(path + '*.wav')
    elif format == 'mp3':
        filenames = glob.glob(path + '*.mp3')

    return filenames

