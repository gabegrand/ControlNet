import gdown

url = "https://drive.google.com/u/0/uc?id=1_AIxKnZXQms5Ezb-cEeVIDIoVG-eliHc&export=download"
gdown.download(url, "image-downloader.zip", quiet=False)
