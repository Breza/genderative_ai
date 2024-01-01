import os
import cv2
import polars as pl


def label_images(csv_file, output_file):
    labels = []

    # Read image paths from the CSV file
    # image_files = pl.read_csv(csv_file).to_series().to_list()

    # TODO: Remove this section before using in production
    image_files = []
    for file in os.listdir(".."):
        if file.endswith(".jpeg"):
            print(os.path.join("..", file))
            image_files.append(os.path.join("..", file))

    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is not None:
            cv2.putText(image, '0=Female, 1=Male, 2=Discard', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Image", image)

            key = cv2.waitKey(0)  # Wait for a key press
            if key == ord('0'):
                labels.append((image_path, "Female"))
            elif key == ord('1'):
                labels.append((image_path, "Male"))
            elif key == ord('2'):
                labels.append((image_path, "Discard"))
            else:
                raise ValueError("Only allowed values are 0, 1, and 2")
                pass
            print(f"Key pressed: {chr(key)}")

            cv2.destroyAllWindows()
        else:
            print(f"Could not read image: {image_path}")

    with open(output_file, 'w') as f:
        for item in labels:
            f.write(f"{item[0]},{item[1]}\n")


label_images("unlabeled_path_test.csv", "labels.csv")
