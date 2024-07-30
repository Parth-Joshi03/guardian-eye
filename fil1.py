import os

def create_faces_directory():
    base_directory = os.getcwd()  # Get the current working directory
    directory_name = "faces"
    directory_path = os.path.join(base_directory, directory_name)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_name}' created.")
    else:
        print(f"Directory '{directory_name}' already exists.")

    return directory_path

faces_directory = create_faces_directory()
