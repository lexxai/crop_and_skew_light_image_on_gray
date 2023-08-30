import pyinstaller_versionfile
from configparser import ConfigParser
import sys
from pathlib import Path


parser = ConfigParser()
# read config file
config_setup=Path("..\setup.cfg")
if config_setup.is_file():
    parser.read(config_setup)
    ver = parser["metadata"].get("version", "0.0.1")
else:
    config_setup=Path("..\pyproject.toml")
    if config_setup.is_file():
        parser.read(config_setup)
        ver = parser["tool.poetry"].get("version", "0.0.1")
        ver = ver.strip('"')

if len(sys.argv) > 1:
    filename = Path(sys.argv[1])
    if filename.is_file():
        new_fn = filename.with_stem(f"{filename.stem}_{ver}")
        filename.rename(new_fn)
else:
    pyinstaller_versionfile.create_versionfile(
        output_file="..\\versionfile.txt",
        version=f"{ver}.0",
        company_name="Xuno",
        file_description="Using Python and OpenCV to split images by segments and try analyze and OCR it.",
        internal_name="ai_crop_images",
        legal_copyright="https://github.com/xuno/",
        original_filename=f"ai_crop_images_{ver}.exe",
        product_name="ai_crop_images",
    )
    print(f"Done: versionfile.txt in parent folder. version='{ver}.0'")
