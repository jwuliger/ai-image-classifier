# AI Image Organizer

AI Image Organizer is a Python application that uses the power of machine learning for automatic image classification and organization. It leverages a pre-trained EfficientNetB7 model from TensorFlow's Keras API to classify images based on their content, and then sorts them into corresponding folders.

The application is built with a user-friendly interface using the tkinter library, enabling users to easily select a directory and organize the images within it.

## Key Features

- **Image Classification**: Uses EfficientNetB7, a state-of-the-art deep learning model, to classify images.
- **Automatic Organization**: Automatically sorts images into folders based on their classifications.
- **User-friendly Interface**: Provides a simple and intuitive GUI for users to select a directory and initiate image organization.
- **Progress Indicator**: Displays a progress bar during the image organization process to keep users informed of progress.

## Usage

Simply run the script and use the GUI to select a directory. The application will organize the images in the selected directory into folders based on their content.

**Please note**: The script requires several dependencies including numpy, nltk, tensorflow, and tkinter. Make sure to install these dependencies before running the script.

## Example

Let's say you have a directory with hundreds of unsorted images of cats, dogs, and cars. After running AI Image Organizer, you'll have three separate folders in the directory: one for cats, one for dogs, and one for cars.

## Future Work

In future versions, we plan to support more image formats and add more advanced features like multi-level image classification and custom classification labels.

**Disclaimer**: The accuracy of image classification depends on the pre-trained model used. While EfficientNetB7 is a powerful model, it may not perfectly classify all types of images.
