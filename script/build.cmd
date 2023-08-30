python build-version.py
mkdir "../pyinstall"
rem ERASE "../pyinstall" /S/Q
PUSHD "../pyinstall"
pyinstaller "../ai_crop_images/main.py" --clean --name ai_crop_images --onefile --version-file "../versionfile.txt"
POPD
python build-version.py ../pyinstall/dist/ai_crop_images.exe
