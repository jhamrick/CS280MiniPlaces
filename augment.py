import os
import re
import skimage.io
import skimage.color
import numpy as np
import argparse


def change_saturation(image, saturation):
    image_hsv = skimage.color.rgb2hsv(image)
    if (image_hsv[:, :, 1] == 0).all():
        return image
    hsv = image_hsv[:, :, 1].copy()
    hsv *= saturation
    hsv[hsv > 1] = 1
    image_hsv[:, :, 1] = hsv
    return skimage.color.hsv2rgb(image_hsv)


def augment(N, image_root):
    images = []
    root = os.path.join(image_root, 'train')
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
    parser = argparse.ArgumentParser(
        description='Train and evaluate a net on the MIT mini-places dataset.')
    parser.add_argument('--image_root', default='./images/',
        help='Directory where images are stored')
    parser.add_argument('--N', default=9, type=int,
        help='Number of new images to generate')
    args = parser.parse_args()
    
    augment(args.N, args.image_root)
