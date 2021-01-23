import os
from PIL import Image
import numpy as np
input_folder = "./data"
output_folder = "./data2"

WIDTH = 64
HEIGHT = 64
SIZE = (HEIGHT, WIDTH)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
index = 0
for dirname, dirnames, filenames in os.walk(os.path.join(os.getcwd(), input_folder)):
    for filename in filenames:
        if(not dirname == output_folder and (".png" in filename or ".jpg" in filename)):
            # img = Image.open(os.path.join(dirname, filename)
            #                  ).convert("RGBA").resize(SIZE)
            img = Image.open(os.path.join(dirname, filename)
                             ).convert("RGBA")
            data = img.getdata()
            new_data = []
            for i, item in enumerate(data):
                if (item[3] < 100):
                    new_data.append((255, 255, 255, 255))
                else:
                    new_data.append(item)
            img = img.convert("RGB")
            # img = img.resize(SIZE)
            img.putdata(new_data)
            img.save(output_folder+"/"+str(index) + ".png")
            index += 1
print("Done!")
