import os
import shutil
import numpy as np
import nltk
import tkinter as tk
from tkinter import ttk, filedialog
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as kimage

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def classify_image(model, image_path):
    """Classify image by model"""
    img = kimage.load_img(image_path, target_size=(
        600, 600))  # EfficientNetB7 expects 600x600 images
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    # Lemmatize (singularize) the predicted class
    return lemmatizer.lemmatize(decode_predictions(preds, top=1)[0][0][1])


def remove_empty_folders(path):
    """Function to remove empty folders"""
    for dirpath, dirnames, files in os.walk(path, topdown=False):
        if not dirnames and not files:
            os.rmdir(dirpath)


def sort_images_into_folders(root_dir, model, progress_var):
    """Sort images into folders by class"""
    image_files = [os.path.join(root, file) for root, dirs, files in os.walk(
        root_dir) for file in files if file.endswith((".jpg", ".png", ".jpeg"))]
    total_files = len(image_files)
    for i, image_path in enumerate(image_files, start=1):
        image_class = classify_image(model, image_path)
        new_folder = os.path.join(root_dir, image_class)
        os.makedirs(new_folder, exist_ok=True)
        new_file_path = os.path.join(new_folder, os.path.basename(image_path))
        shutil.move(image_path, new_file_path)
        progress_var.set(i / total_files * 100)
    remove_empty_folders(root_dir)


def organize_images():
    """Sort the images in the selected directory"""
    if selected_directory:
        nltk.download('wordnet', quiet=True)
        model = EfficientNetB7(weights='imagenet')
        sort_images_into_folders(selected_directory, model, progress_var)


def select_directory():
    """Open a directory selection dialog and store the selected directory"""
    global selected_directory
    if selected_directory := filedialog.askdirectory():
        organize_button.config(state='normal')


# Create a simple GUI with two buttons
root = tk.Tk("AI Image Organizer")
root.geometry('800x600')  # Set the window size to 800x600

# Add a theme
style = ttk.Style()
style.theme_use('clam')  # You can replace 'clam' with your preferred theme

title = ttk.Label(root, text="AI Image Organizer", font=("Arial", 30))
# Match the label's background to the window's background
title.configure(background=root.cget('bg'))
title.pack(pady=20)  # Add the title to the window with some vertical padding

select_button = ttk.Button(
    root, text="Select Directory", command=select_directory)
organize_button = ttk.Button(root, text="Organize Images", command=organize_images,
                             state='disabled')  # Disable the button initially
select_button.pack(padx=10, pady=10)
organize_button.pack(padx=10, pady=10)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(
    root, length=300, variable=progress_var, mode='determinate')
progress_bar.pack(padx=10, pady=10)

root.mainloop()
