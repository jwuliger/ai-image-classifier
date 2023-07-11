import os
import shutil
import numpy as np
import nltk
import tkinter as tk
from tkinter import ttk, filedialog
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as kimage


class ImageOrganizer:
    def __init__(self):
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.selected_directory = None  # Initialize selected directory

        # Setup GUI
        self.root = tk.Tk("AI Image Organizer")
        self.root.geometry('800x600')  # Set window size

        # Setup theme
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Set theme

        # Create title label
        self.title = ttk.Label(
            self.root, text="AI Image Organizer", font=("Arial", 30))
        # Set title background
        self.title.configure(background=self.root.cget('bg'))
        self.title.pack(pady=20)  # Add title to window with vertical padding

        # Create select directory and organize images buttons
        self.select_button = ttk.Button(
            self.root, text="Select Directory", command=self.select_directory)
        # Initially disable organize button
        self.organize_button = ttk.Button(
            self.root, text="Organize Images", command=self.organize_images, state='disabled')
        # Add select button to window with padding
        self.select_button.pack(padx=10, pady=10)
        # Add organize button to window with padding
        self.organize_button.pack(padx=10, pady=10)

        # Create progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.root, length=300, variable=self.progress_var, mode='determinate')
        # Add progress bar to window with padding
        self.progress_bar.pack(padx=10, pady=10)

        self.root.mainloop()  # Start event loop

    def classify_image(self, model, image_path):
        # Load image and preprocess for model
        img = kimage.load_img(image_path, target_size=(600, 600))
        x = kimage.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict class of image
        preds = model.predict(x)

        # Return lemmatized class
        return self.lemmatizer.lemmatize(decode_predictions(preds, top=1)[0][0][1])

    def remove_empty_folders(self, path):
        # Remove empty directories in path
        for dirpath, dirnames, files in os.walk(path, topdown=False):
            if not dirnames and not files:
                os.rmdir(dirpath)

    def sort_images_into_folders(self, root_dir, model):
        # Get all image files in root directory
        image_files = [os.path.join(root, file) for root, dirs, files in os.walk(
            root_dir) for file in files if file.endswith((".jpg", ".png", ".jpeg"))]
        total_files = len(image_files)

        # Classify each image and move to corresponding folder
        for i, image_path in enumerate(image_files, start=1):
            image_class = self.classify_image(model, image_path)
            new_folder = os.path.join(root_dir, image_class)
            os.makedirs(new_folder, exist_ok=True)
            new_file_path = os.path.join(
                new_folder, os.path.basename(image_path))
            shutil.move(image_path, new_file_path)

            # Update progress bar
            self.progress_var.set(i / total_files * 100)

        # Remove any empty folders
        self.remove_empty_folders(root_dir)

    def organize_images(self):
        # If directory is selected, organize images
        if self.selected_directory:
            # Download wordnet for lemmatizer
            nltk.download('wordnet', quiet=True)
            model = EfficientNetB7(weights='imagenet')  # Load model
            self.sort_images_into_folders(
                self.selected_directory, model)  # Sort images

    def select_directory(self):
        # Open directory selection dialog and store selected directory
        self.selected_directory = filedialog.askdirectory()

        # If directory is selected, enable organize button
        if self.selected_directory:
            self.organize_button.config(state='normal')


if __name__ == "__main__":
    # Instantiate and run ImageOrganizer if script is run directly
    app = ImageOrganizer()
