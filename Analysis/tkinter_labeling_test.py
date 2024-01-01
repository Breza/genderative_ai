import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import csv

# Constants
IMAGE_SIZE = (512, 512)
# TODO: Replace IMAGE_FILES with images for review
IMAGE_FILES = [
    "../A_photograph_of_a_veterinarian_S1633674812_St40_G7.5.jpeg",
    "../A_photograph_of_a_male_veterinarian_S1633674812_St40_G7.5.jpeg",
    "../A_portrait_of_a_veterinarian_S1345394950_St25_G7.5.525.jpeg",
    "../A_photograph_of_a_female_veterinarian_S1633674812_St40_G7.5.jpeg"
]


class ToolTip:
    def __init__(self, widget: tk.Widget):
        self.widget = widget
        self.tipwindow = None
        self.x = self.y = 0

    def show_tip(self, text: str):
        if self.tipwindow or not text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.current_index = 0
        self.classifications = []
        self.preloaded_images = self._preload_images()
        self.setup_ui()

    def _preload_images(self):
        return [Image.open(file) for file in IMAGE_FILES]

    def setup_ui(self):
        self.image_label = ttk.Label(self.root)
        self.image_label.pack()

        self.create_buttons()
        self.create_progress_bar()
        self.create_log_window()

        self.load_next_image()

    def create_buttons(self):
        female_button = ttk.Button(self.root, text="Female", command=lambda: self.classify_image("Female"))
        female_button.pack(side=tk.LEFT)
        ToolTip(female_button).show_tip("Press 'f' for Female")

        male_button = ttk.Button(self.root, text="Male", command=lambda: self.classify_image("Male"))
        male_button.pack(side=tk.LEFT)
        ToolTip(male_button).show_tip("Press 'm' for Male")

        discard_button = ttk.Button(self.root, text="Discard", command=lambda: self.classify_image("Discard"))
        discard_button.pack(side=tk.LEFT)
        ToolTip(discard_button).show_tip("Press 'd' to Discard")

        self.root.bind('f', lambda event: self.classify_image("Female"))
        self.root.bind('m', lambda event: self.classify_image("Male"))
        self.root.bind('d', lambda event: self.classify_image("Discard"))

    def create_progress_bar(self):
        self.progress_bar = ttk.Progressbar(self.root, length=100, mode='determinate', maximum=len(IMAGE_FILES))
        self.progress_bar.pack()

    def create_log_window(self):
        self.log_window = tk.Text(self.root, height=4, width=50)
        self.log_window.pack()

    def load_next_image(self):
        if self.current_index < len(self.preloaded_images):
            img = ImageTk.PhotoImage(self.preloaded_images[self.current_index].resize(IMAGE_SIZE))
            self.image_label.configure(image=img)
            self.image_label.image = img  # Keep a reference
            self.progress_bar['value'] = self.current_index + 1
            self.current_index += 1
        else:
            self.save_to_csv()

    def classify_image(self, label: str):
        if self.current_index <= len(IMAGE_FILES):
            self.classifications.append((IMAGE_FILES[self.current_index - 1], label))
            self.log_window.insert(tk.END, f"Image {self.current_index} classified as {label}\n")
            self.log_window.see(tk.END)
            self.load_next_image()

    def save_to_csv(self):
        with open('classifications.csv', 'w', newline='') as file:
            writer d= csv.writer(file)
            writer.writerows(self.classifications)
        print("Classifications saved to classifications.csv")
        self.root.quit()


def run_image_labeler():
    root = ThemedTk(theme="arc")
    root.title("Genderative AI Labeler")
    app = ImageClassifierApp(root)
    root.mainloop()


run_image_labeler()
