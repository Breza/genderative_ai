import datetime
import hashlib
import os
import shutil
from functools import cache
from typing import Dict, List, NamedTuple, Union

import PIL
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import polars as pl
import seaborn as sns
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from IPython.display import display
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

# Helpers


# Functions
def current_time_only(file_safe: bool = False) -> str:
    """
    Returns the current time at second precision without date.

    Parameters:
    file_safe (bool): If True, returns time formatted for file naming (replaces ':' with '_').

    Returns:
    str: Current time formatted as 'HH:MM:SS' or 'DD_HH_MM_SS' if file_safe is True.
    """
    if file_safe:
        return datetime.datetime.now().strftime("%d:%H:%M:%S").replace(":", "_")
    else:
        return datetime.datetime.now().strftime("%H:%M:%S")


def count_files_in_directory(path: Union[str, "LiteralString"]) -> int:
    """
    Counts the number of files in a given directory.

    Parameters:
    path (Union[str, 'LiteralString']): The file path to the directory whose contents are to be counted.

    Raises:
    ValueError: If path is not a directory or does not exist.

    Returns:
    int: The number of files in the specified directory.
    """
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory.")
    return len(os.listdir(path))


def get_dataset_structure_hash() -> str:
    """
    Get list of all training and holdout files. Uses data_dir to define base directory.

    Returns:
    str: SHA512 hash of the file names of all training and holdout files.
    """
    file_path_list = []

    for outer_dir in ["Labeled", "Holdout"]:
        for inner_dir in ["Female", "Male"]:
            dir_path = os.path.join(data_dir, outer_dir, inner_dir)

            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory does not exist: {dir_path}")

            with os.scandir(dir_path) as entries:
                for entry in entries:
                    file_path_list.append(entry.path)
    file_path_list.sort()
    file_path_list = "".join(file_path_list)
    file_path_list_hash = hashlib.sha512(file_path_list.encode()).hexdigest()
    return file_path_list_hash


def plot_training_progress(
    train_acc: list,
    train_loss: list,
    val_acc: list,
    validation_loss: list,
    title="Model results",
):
    """
    Plot training-vs-testing accuracy and loss for each epoch.

    Parameters:
        train_acc (list): List of training accuracy values from each epoch, must be same length as val_acc
        train_loss (list): List of training loss values from each epoch, must be same length as val_loss
        val_acc (list): List of validation accuracy values from each epoch, must be same length as train_acc
        validation_loss (list): List of validation loss values from each epoch, must be same length as train_loss
        title (str): Plot title

    Raises:
        ValueError: If the lengths of accuracy or loss lists don't match each other
    """
    if (
        not (len(train_acc) == len(val_acc) and len(train_loss) == len(validation_loss))
        and len(train_acc) > 0
        and len(train_loss) > 0
    ):
        raise ValueError("Lengths of training and validation lists must match")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(train_acc, label="Training accuracy", color=train_color, linewidth=3)
    axs[0].plot(val_acc, label="Validation accuracy", color=val_color, linewidth=3)
    axs[0].set_title("Accuracy")
    axs[0].legend()

    axs[1].plot(train_loss, label="Training loss", color=train_color, linewidth=3)
    axs[1].plot(validation_loss, label="Validation loss", color=val_color, linewidth=3)
    axs[1].set_title("Loss")
    axs[1].set(ylim=(0, None))
    axs[1].legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


def add_occupation(image_data_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Adds an 'occupation' column to the DataFrame by first neutralizing gender-specific terms
    in the 'file_path' column and then extracting the occupation.

    The function first removes 'female_' or 'male_' from the 'file_path', then assumes that
    the occupation is embedded in the modified file path string, following a specific pattern
    (fourth underscore-separated value after removal of gender terms).

    Parameters:
    image_data_frame (pl.DataFrame): A DataFrame with a column named 'file_path'.

    Returns:
    pl.DataFrame: The original DataFrame with an added 'occupation' column,
                  and 'neutered_file_path' if intermediate results are to be kept.

    Raises:
    ValueError: If the 'file_path' column does not exist in the DataFrame.
    """
    if "file_path" not in image_data_frame.columns:
        raise ValueError("The DataFrame must contain a 'file_path' column.")

    return image_data_frame.with_columns(
        pl.col("file_path")
        .str.replace(pattern="(female_|male_)", value="")
        .str.extract("(?:[^_]*_){4}([^_]*)_")
        .str.replace("dietitian", "nutritionist")
        .alias("occupation")
    )


def label_images(csv_file: str, output_file: str):
    """
    Reads images from a specified CSV file and allows the user to label them interactively.

    This function opens each image specified in the CSV file and displays it to the user.
    The user can then label the image as 'Female', 'Male', or 'Discard' by pressing the corresponding
    keys ('f', 'm', 'd' or '0', '1', '2'). The function logs the user's decision along with the image
    path and writes this data to an output file.

    Parameters:
    csv_file (str): Path to the CSV file containing the paths of the images to be labeled.
    output_file (str): Path to the output CSV file where the image paths and labels will be saved.

    Raises:
    ValueError: If a key pressed is not among the specified keys ('f', 'm', 'd', '0', '1', '2').

    Notes:
    The function adds a text overlay to each image before displaying it, which indicates the keys
    for labeling. If an image cannot be read, it logs the path with the label "ERROR".
    """
    image_labels = []
    image_files = pl.read_csv(csv_file).to_series().to_list()
    image_files_length = len(image_files)

    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is not None:
            cv2.putText(
                image,
                "'F'emale, 'M'ale, 'D'iscard",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Image", image)

            key = cv2.waitKey(0)  # Wait for a key press
            if key == ord("0") or key == ord("f"):
                image_labels.append((image_path, "Female"))
            elif key == ord("1") or key == ord("m"):
                image_labels.append((image_path, "Male"))
            elif key == ord("2") or key == ord("d"):
                image_labels.append((image_path, "Discard"))
            else:
                raise ValueError("Only allowed values are 0, 1, 2, f, m, and d")
            image_index = image_files.index(image_path) + 1
            print(f"Image {image_index}/{image_files_length} classified as {chr(key)}")

            cv2.destroyAllWindows()
        else:
            print(f"Could not read image: {image_path}")
            image_labels.append((image_path, "ERROR"))

    with open(output_file, "w") as f:
        for item in image_labels:
            f.write(f"{item[0]},{item[1]}\n")


@cache
def calculate_normalization(
    custom_normalization: bool, hash_value: str
) -> Dict[str, List[float]]:
    """
    Calculates the mean and standard deviation of the dataset for normalization.

    Parameters:
    custom_normalization (bool): Determines if custom normalization values are to be calculated.
    hash_value (str): The hash value of the dataset structure. Used for cache invalidation.

    Returns:
    Dict[str, List[float]]: A dictionary containing 'mean' and 'std', each a list of three floats.

    Notes:
    Function uses global variables batch_size and data_dir.
    """
    # No-op to satisfy IDE checks
    _ = len(hash_value)

    if custom_normalization:
        # Create a dataset without normalization
        unnormalized_dataset = ImageFolder(
            os.path.join(data_dir, "Labeled"), transform=tensor_transform
        )
        unnormalized_loader = DataLoader(
            unnormalized_dataset, batch_size=batch_size, shuffle=True
        )

        def calculate_mean_std(loader: DataLoader) -> (torch.Tensor, torch.Tensor):
            """
            Calculate the mean and standard deviation of images in a DataLoader.

            Parameters:
            loader (DataLoader): The DataLoader containing the dataset.

            Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation tensors.
            """
            mean_accumulator = 0.0
            variance_accumulator = 0.0
            for images, _ in loader:
                batch_samples = images.size(0)
                images = images.view(batch_samples, images.size(1), -1)
                mean_accumulator += images.mean(2).sum(0)
                variance_accumulator += images.var(2).sum(0)

            mean_accumulator /= len(loader.dataset)
            std_deviation = torch.sqrt(variance_accumulator / len(loader.dataset))
            return mean_accumulator, std_deviation

        dataset_mean, dataset_std = calculate_mean_std(unnormalized_loader)
        normalization_dict = {
            "mean": dataset_mean.tolist(),
            "std": dataset_std.tolist(),
        }
        del unnormalized_dataset, unnormalized_loader
        print(f"Custom normalization values: {normalization_dict}")
    else:
        normalization_dict = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        print("Standard normalization values used")
    return normalization_dict


def display_images_from_dataloader(dataloader: DataLoader, num_images: int = 8):
    """
    Fetches a batch of images from the given DataLoader and displays them to user.

    Parameters:
    dataloader (DataLoader): A PyTorch DataLoader object from which to fetch the images.
    num_images (int): The number of images to display from the batch.
    """

    def imshow(img: torch.Tensor):
        """
        Display an image by denormalizing and clipping its values.

        This function takes a PyTorch tensor representing a grid of images, which have been normalized
        previously, and performs denormalization to convert them back to their original color
        space. It then clips the image values to be within the range [0, 1] to ensure
        proper display. The image is displayed using matplotlib.

        Parameters:
        img (torch.Tensor): A PyTorch tensor representing a grid of images.
        """
        img = img.numpy().transpose((1, 2, 0))
        img = (
            normalization_values["std"] * img + normalization_values["mean"]
        )  # Denormalize
        img = np.clip(img, 0, 1)  # Clip values to be between 0 and 1
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    num_images = min(batch_size, num_images)
    dataset_images, dataset_labels = next(iter(dataloader))
    images_subset = dataset_images[:num_images]
    imshow(torchvision.utils.make_grid(images_subset))
    print(" ".join(f"{classes[dataset_labels[j]]:5s}" for j in range(num_images)))


def predict_model_resnet(my_dataloader):
    """
    Performs predictions using a ResNet model on a given dataloader.

    This function runs the ResNet model in evaluation mode and generates predictions for the input data.
    It requires the dataloader to have a batch size of exactly 1. The function outputs a dataframe
    with file paths, actual labels, predicted labels, probabilities of being female, and a flag indicating
    correct predictions.

    Note: The sorting logic in the function is based on the condition of correct predictions and their confidence scores.

    Parameters:
    my_dataloader (Dataloader): A PyTorch Dataloader object containing the data to be predicted.
                                 The dataloader must have a batch size of 1.

    Raises:
    ValueError: If the dataloader's batch size is not 1.

    Returns:
    Polars.DataFrame: A dataframe containing columns for file paths, labels, predictions,
                      probabilities, and correct predictions.
    """
    if my_dataloader.batch_size != 1:
        raise ValueError("Predictions dataloader requires batch size of exactly 1")
    model_resnet.eval()
    prediction_results = pl.DataFrame(
        schema={"file_path": str, "label": int, "prediction": int, "prob_female": float}
    )
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(my_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_resnet(inputs)
            _, predicted = torch.max(outputs.data, 1)
            prob_female = torch.nn.functional.softmax(outputs, dim=1)[:, 0].tolist()
            predicted = predicted.tolist()
            labels = labels.tolist()
            file_paths = my_dataloader.dataset.samples[i][0]
            batch_results = pl.DataFrame(
                {
                    "file_path": file_paths,
                    "label": labels,
                    "prediction": predicted,
                    "prob_female": prob_female,
                }
            )
            prediction_results = pl.concat(
                [prediction_results, batch_results], how="vertical_relaxed"
            )
        prediction_results = prediction_results.with_columns(
            (1 * (pl.col("label") == pl.col("prediction"))).alias("correct_prediction")
        )
        # I couldn't figure out the sorting logic so I asked SO
        # https://stackoverflow.com/questions/77700489/how-to-perform-a-conditional-sort-in-polars/77700711
        prediction_results = prediction_results.with_columns(
            abs(pl.col("prob_female") - 0.5).alias("confidence")
        ).sort(
            [
                (good_prediction := pl.col("label").eq(pl.col("prediction"))),
                (good_prediction - 1) * pl.col("confidence"),
                pl.col("confidence"),
            ]
        )
        return prediction_results


def move_image(file_path, new_folder) -> None:
    """
    Move image located at file_path to new_folder.

    :param file_path: Location of file to be moved.
    :param new_folder: Folder where you want the file to be moved.
    :return: None
    """
    new_path = os.path.join(data_dir, new_folder, os.path.basename(file_path))
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.move(file_path, new_path)
    print(f"File {file_path} successfully moved to {new_folder}")
    with open("file_move_history.txt", "a") as f:
        f.write(f"{file_path}\n")


def move_labeled_images(image_label_dataframe_path: str = "labels.csv"):
    image_label_dataframe = pl.read_csv(
        image_label_dataframe_path,
        has_header=False,
        new_columns=["file_path", "gender"],
        schema={"file_path": pl.Utf8, "gender": pl.Utf8},
    ).with_columns(
        # Only allow the three valid image labels
        pl.col("gender").cast(pl.Enum(categories=["Female", "Male", "Discard"]))
    )

    for row in image_label_dataframe.iter_rows():
        file_path, label = row
        sub_folder = (
            file_path.replace(data_dir, "", 1).strip(os.sep).split(os.sep)[0].lstrip()
        )
        sub_folder = data_dir + sub_folder
        destination_folder = os.path.join(sub_folder, label)
        current_folder = os.path.dirname(file_path)
        # Check if the current folder is the same as the destination folder
        if current_folder != destination_folder:
            if os.path.isfile(file_path):
                move_image(file_path, destination_folder)
            else:
                print(f"Image not found: {file_path}")
        else:
            print(f"File is already in the correct folder: {file_path}")
            with open("file_move_history.txt", "a") as f:
                f.write(f"{file_path}\n")


# Classes
class ImbalancedClassesException(ValueError):
    """Exception raised when classes must be perfectly balanced and are not."""


class Colors(NamedTuple):
    train: str
    val: str
    unlabeled: str
