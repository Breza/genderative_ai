import tkinter as tk
from PIL import Image, ImageTk


def classify_image(label):
    # Function to handle the classification logic
    print(f"Image classified as {label}")


# Create the main window
root = tk.Tk()
root.title("Image Classification Tool")

# Create a canvas for image display
canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

# Load and display an image (example)
image = Image.open("../A_photograph_of_a_female_veterinarian_S1633674812_St40_G7.5.jpeg")
image = ImageTk.PhotoImage(image.resize((300, 300)))
canvas.create_image(0, 0, anchor=tk.NW, image=image)

# Add buttons
tk.Button(root, text="Female", command=lambda: classify_image("Female")).pack(side=tk.LEFT)
tk.Button(root, text="Male", command=lambda: classify_image("Male")).pack(side=tk.LEFT)
tk.Button(root, text="Discard", command=lambda: classify_image("Discard")).pack(side=tk.LEFT)

# Start the GUI event loop
root.mainloop()
