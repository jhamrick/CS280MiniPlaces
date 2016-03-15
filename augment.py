import os
import re
import skimage.io
import skimage.color
import numpy as np


def change_saturation(image, saturation):
    image_hsv = skimage.color.rgb2hsv(image)
    if (image_hsv[:, :, 1] == 0).all():
        return image
    hsv = image_hsv[:, :, 1].copy()
    hsv *= saturation
    hsv[hsv > 1] = 1
    image_hsv[:, :, 1] = hsv
    return skimage.color.hsv2rgb(image_hsv)


def augment(N=9):
    images = []
    root = 'images/train'
    for dirname, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if re.match("[0-9]+.jpg", filename):
                images.append(os.path.join(dirname, filename))

    categories = np.loadtxt("development_kit/data/categories.txt", delimiter=" ", dtype=str)
    fh = open("development_kit/data/train.txt", "w")
    for image_path in sorted(images):
        print(image_path)
        path, name = os.path.split(image_path)
        cat = os.path.relpath(path, root)
        idx = int(categories[categories[:, 0] == "/" + cat, 1][0])
        fh.write("{} {}\n".format(os.path.join("train", cat, name), idx))

        image = skimage.io.imread(image_path)
        for saturation in np.linspace(0, 2, N):
            new_image_name = "{}-{}.jpg".format(os.path.splitext(name)[0], saturation)
            new_image_path = os.path.join(path, new_image_name)
            fh.write("{} {}\n".format(os.path.join("train", cat, new_image_name), idx))
            if os.path.exists(new_image_path):
                continue

            new_image = change_saturation(image, saturation)
            skimage.io.imsave(new_image_path, new_image)
            

if __name__ == "__main__":
    augment()
