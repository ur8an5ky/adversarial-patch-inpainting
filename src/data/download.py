import urllib.request, tarfile, os

os.makedirs('../data', exist_ok=True)
print('Pobieranie Imagenette2-320...')
urllib.request.urlretrieve('https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
                           '../data/imagenette2-320.tgz')
print('Wypakowywanie...')
tar = tarfile.open('../data/imagenette2-320.tgz', 'r:gz')
tar.extractall('../data')
tar.close()
print('Gotowe!')
