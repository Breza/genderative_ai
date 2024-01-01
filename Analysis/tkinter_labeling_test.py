import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import csv


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def show_tip(self, text):
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


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


def classify_image(label):
    if current_index <= len(image_files):
        classifications.append((image_files[current_index - 1], label))
        log_window.insert(tk.END, f"Image {current_index} classified as {label}\n")
        log_window.see(tk.END)
        load_next_image()


def save_to_csv():
    with open('classifications.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(classifications)
    print("Classifications saved to classifications.csv")
    root.quit()


# List of image files
image_files = ["../A_photograph_of_a_veterinarian_S1633674812_St40_G7.5.jpeg",
               "../A_photograph_of_a_male_veterinarian_S1633674812_St40_G7.5.jpeg",
               "../A_portrait_of_a_veterinarian_S1345394950_St25_G7.5.525.jpeg",
               "../A_photograph_of_a_female_veterinarian_S1633674812_St40_G7.5.jpeg"]
classifications = []
current_index = 0

# Use a themed Tk instance
root = ThemedTk(theme="equilux")
root.title("Image Classification Tool")

# Create a label for image display
image_label = tk.Label(root)
image_label.pack()

# Add buttons with tool tips
female_button = tk.Button(root, text="Female", command=lambda: classify_image("Female"))
female_button.pack(side=tk.LEFT)
ToolTip(female_button).show_tip("Press 'f' for Female")

male_button = tk.Button(root, text="Male", command=lambda: classify_image("Male"))
male_button.pack(side=tk.LEFT)
ToolTip(male_button).show_tip("Press 'm' for Male")

discard_button = tk.Button(root, text="Discard", command=lambda: classify_image("Discard"))
discard_button.pack(side=tk.LEFT)
ToolTip(discard_button).show_tip("Press 'd' to Discard")

# Add a progress bar
progress_bar = ttk.Progressbar(root, length=100, mode='determinate', maximum=len(image_files))
progress_bar.pack()

# Bind keyboard events
root.bind('f', lambda event: classify_image("Female"))
root.bind('m', lambda event: classify_image("Male"))
root.bind('d', lambda event: classify_image("Discard"))

# Logging Window
log_window = tk.Text(root, height=4, width=50)
log_window.pack()

# Load the first image
load_next_image()

# Start the GUI event loop
root.mainloop()
