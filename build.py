import PyInstaller.__main__
import os
import shutil
import sys

# Configuration
APP_NAME = "Webcam Pulse Detector"
MAIN_SCRIPT = "main.py"
ICON_FILE = "icon.ico"
ADD_DATA = ["assets/", "config.ini"]


# PyInstaller options
options = [
    '--onefile',
    '--windowed',
    f'--name={APP_NAME}',
    '--clean'
]

if os.path.exists(ICON_FILE):
    options.append(f'--icon={ICON_FILE}')

# Add additional data files
for data in ADD_DATA:
    if os.path.exists(data):
        dest_folder = os.path.dirname(data) if os.path.isfile(data) else data.rstrip("/\\")
        options.extend(['--add-data', f'{data}{os.pathsep}{dest_folder}'])

# Build command
options.append(MAIN_SCRIPT)

# Run PyInstaller
print(f"Running PyInstaller with options:\n{options}\n")
PyInstaller.__main__.run(options)

# Notify
print(f"\nBuild complete! Check the 'dist' folder for {APP_NAME}.exe")
