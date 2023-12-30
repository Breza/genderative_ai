import cv2
import os
import polars as pl

print(pl.read_csv("unlabeled_path_test.csv").to_series().to_list())
def label_images(image_folder, output_file):
    labels = []
    #image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_files = pl.read_csv("unlabeled_path_test.csv").to_series().to_list()

    for image_path in image_files:
        #image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        cv2.imshow("Image", image)

        key = cv2.waitKey(0)  # Wait for a key press
        if key == ord('0'):
            labels.append((image_path, 0))
        elif key == ord('1'):
            labels.append((image_path, 1))

        cv2.destroyAllWindows()

    # Save labels to a file
    with open(output_file, 'w') as f:
        for item in labels:
            f.write(f"{item[0]},{item[1]}\n")

# Usage
label_images("path/to/image/folder", "labels.csv")
