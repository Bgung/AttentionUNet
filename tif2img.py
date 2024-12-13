import PIL

from PIL import Image

ann = Image.open('data/High-Resolution Fundus (HRF) Image Database/manual1/01_dr.tif')
ann.save('01_dr.png')