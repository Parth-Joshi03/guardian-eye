import os
import tkinter as tk
from tkinter import filedialog
import cv2


def take_photo():
    # Open the camera
    camera = cv2.VideoCapture(0)
    _, image = camera.read()
    
    # Close the camera
    camera.release()
    
    # Save the photo with the given name
    if name_entry.get():
        cv2.imwrite(f"faces/{name_entry.get()}.jpg", image)
        print("Photo saved successfully!")
    else:
        print("Error: No name given!")


def upload_photo():
    # Open the file dialog to choose a photo file
    filetypes = [("JPEG Files", "*.jpeg"), ("JPG Files", "*.jpg"), ("PNG Files", "*.png")]
    filename = filedialog.askopenfilename(initialdir="/", title="Select a file", filetypes=filetypes)
    
    # Check if a file is selected
    if filename:
        # Read the selected photo file
        image = cv2.imread(filename)
        
        # Save the photo with the given name
        if name_entry.get():
            cv2.imwrite(f"faces/{name_entry.get()}.jpg", image)
            print("Photo uploaded successfully!")
        else:
            print("Error: No name given!")


# Create the main window
window = tk.Tk()
window.title("Registry")
window.geometry("600x300")
window.resizable(False, False)  # Disable window resizing

# Create the name label and entry
name_label = tk.Label(window, text="Name:")
name_label.pack()

name_entry = tk.Entry(window)
name_entry.pack()


# Create the photo options
options_label = tk.Label(window, text="Photo Options:")
options_label.pack()

take_photo_button = tk.Button(window, text="Take from Camera", command=take_photo)
take_photo_button.pack()

upload_photo_button = tk.Button(window, text="Upload from Gallery", command=upload_photo)
upload_photo_button.pack()

# Create the faces directory if it doesn't exist
os.makedirs("faces", exist_ok=True)


# Function to handle window close event
def on_closing():
    window.destroy()


# Bind the window close event to the on_closing function
window.protocol("WM_DELETE_WINDOW", on_closing)

# Run the main window loop
window.mainloop()
