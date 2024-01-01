import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv


# Function to update the display with the next image
def load_next_image():
    global current_index, image_label
    if current_index < len(image_files):
        img = Image.open(image_files[current_index])
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        progress_bar['value'] = current_index + 1
        current_index += 1
    else:
        save_to_csv()


# Function to handle the classification and load the next image
def classify_image(label):
    if current_index <= len(image_files):
        classifications.append((image_files[current_index - 1], label))
        load_next_image()


# Function to save classifications to a CSV file
def save_to_csv():
    with open('tkinter_classifications.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(classifications)
    print("Classifications saved to classifications.csv")
    root.quit()


# List of image files (replace with your actual file paths)
image_files = ["../A_photograph_of_a_veterinarian_S1633674812_St40_G7.5.jpeg",
               "../A_photograph_of_a_male_veterinarian_S1633674812_St40_G7.5.jpeg",
               "../A_portrait_of_a_veterinarian_S1345394950_St25_G7.5.525.jpeg",
               "../A_photograph_of_a_female_veterinarian_S1633674812_St40_G7.5.jpeg"]
classifications = []
current_index = 0

# Create the main window
root = tk.Tk()
root.title("Image Classification Tool")

# Create a label for image display
image_label = tk.Label(root)
image_label.pack()

# Add buttons for classification
tk.Button(root, text="Female", command=lambda: classify_image("Female")).pack(side=tk.LEFT)
tk.Button(root, text="Male", command=lambda: classify_image("Male")).pack(side=tk.LEFT)
tk.Button(root, text="Discard", command=lambda: classify_image("Discard")).pack(side=tk.LEFT)

# Add a progress bar
progress_bar = ttk.Progressbar(root, length=100, mode='determinate', maximum=len(image_files))
progress_bar.pack()

# Bind keyboard events
root.bind('f', lambda event: classify_image("Female"))
root.bind('m', lambda event: classify_image("Male"))
root.bind('d', lambda event: classify_image("Discard"))

# Load the first image
load_next_image()

# Start the GUI event loop
root.mainloop()
