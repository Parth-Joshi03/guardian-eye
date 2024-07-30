from cx_Freeze import setup, Executable

# List of Python scripts to include
scripts = [
    "Application.py",
    "register.py",
    "fil1.py"
]

# List of directories to include
directories = [
    "faces"
]

# Define the list of files to include as resources
resources = ['/home/parthjoshi/Desktop/webside/face_detection/Face_recognition/shape_predictor_68_face_landmarks.dat']

setup(
    name="Your Script",
    version="1.0",
    description="Description of your script",
    executables=[Executable(script) for script in scripts],
    options={
        'build_exe': {
            'include_files': resources,
            'include_msvcr': True,
            'include_files': [(directory, directory) for directory in directories]
        }
    }
)

