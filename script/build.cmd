mkdir "../pyinstall"
ERASE "../pyinstall" /S/Q
PUSHD "../pyinstall"

python build-version.py
pyinstaller "../ai_crop_images/main.py" --clean --name ai_crop_images --onefile --version-file "../versionfile.txt" 
POPD