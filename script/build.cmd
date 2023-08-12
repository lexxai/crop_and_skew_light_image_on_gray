python build-version.py

mkdir "../pyinstall"
ERASE "../pyinstall" /S/Q
PUSHD "../pyinstall"

pyinstaller "../ai_crop_images/main.py" --clean --name ai_crop_images --onefile --version-file "../versionfile.txt" 
POPD