import argparse
from pathlib import Path
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
from matplotlib import cm

def get_overlap_image(i, key, ax,class_id, input_folder):
    final_image = load_images_from_folder(key[0])
    final_image = (final_image/key[1])*255
    final_image = final_image.astype(np.uint8)
    im = Image.fromarray(np.uint8(cm.seismic(final_image)*255))
    output_folder = "{}/heatmaps/{}/".format(input_folder, class_id)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    im.save(os.path.join(output_folder,"{}.png".format(key[1])))
    ax.imshow(final_image, cmap="seismic")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(key[1])

def load_images_from_folder(folder):
    final_image = None
    for filename in os.listdir(folder):
        img = np.asarray(Image.open(os.path.join(folder,filename)))

        img[img < 125] = 0
        img[img >= 125] = 1
        if final_image is not None:
            final_image = final_image + img
        else:
            final_image = img
    return final_image
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--input_folder', default="../dictionary/mnist/test/", type=str, help='Dictionary images folder')
    args = parser.parse_args()
    return args

def main(args):
    for class_id in range(10):
        class_id = str(class_id)
        output_path = os.path.join(args.input_folder)
        path = os.path.join(args.input_folder, "originals", class_id)
        min_count = 0
        list_of_keys=[x[0] for x in os.walk(path)]
        list_of_keys.remove(path)
        dict = {}
        for key in list_of_keys:
            count = len([name for name in os.listdir(key) if os.path.isfile(os.path.join(key, name))])
            if count > min_count:
                dict[key] = count

        sort_dict = sorted(dict.items(), key=lambda x: x[1])

        f, axes = plt.subplots(1, len(sort_dict))

        for i, key in enumerate(sort_dict):
            print(key)
            get_overlap_image(i, key, axes[i], class_id, output_path)

        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.show()
        cv2.destroyAllWindows() 


if __name__ == "__main__":
    args = parse_arguments()
    main(args)



