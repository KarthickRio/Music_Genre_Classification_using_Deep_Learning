import json
import os
import cv2  # OpenCV for image processing
import numpy as np

DATASET_PATH = "images_original4"  # Path to the image dataset
JSON_PATH = "data_visual.json"
IMG_SIZE = (224, 224)  # Resize all images to a consistent size, e.g., 224x224 (commonly used for CNNs)


def save_image_features(dataset_path, json_path, img_size=IMG_SIZE, num_segments=5):
    """Extracts image features (e.g., pixel data) from spectrogram dataset and saves them to a JSON file along with labels.

    :param dataset_path (str): Path to dataset containing spectrogram images
    :param json_path (str): Path to JSON file to save features
    :param img_size (tuple): Size to which images should be resized (width, height)
    :param num_segments (int): Number of segments per image (if applicable)
    """

    # dictionary to store mapping, labels, and features
    data = {
        "mapping": [],
        "labels": [],
        "image_data": []
    }

    # loop through all genre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder
        if dirpath != dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all image files in genre sub-dir
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load image file
                img = cv2.imread(file_path)

                if img is not None:
                    # resize the image to a fixed size (e.g., 224x224)
                    img = cv2.resize(img, img_size)

                    # normalize pixel values to [0, 1] (optional, but common practice)
                    img = img / 255.0

                    # store image data and label
                    data["image_data"].append(img.tolist())  # Convert image to list to store in JSON
                    data["labels"].append(i - 1)  # Store the label (genre index)
                    print("Processed image: {}".format(file_path))

    # save image features to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_image_features(DATASET_PATH, JSON_PATH)
