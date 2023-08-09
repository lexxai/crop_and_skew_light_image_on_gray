import pyinstaller_versionfile

pyinstaller_versionfile.create_versionfile(
    output_file="versionfile.txt",
    version="0.1.0.0",
    company_name="lexxai",
    file_description="Using Python and OpenCV to detect the border of a white image on a gray background, crops and corrects its geometry.",
    internal_name="ai_crop_images",
    legal_copyright="https://github.com/lexxai/crop_and_skew_white_image_on_gray",
    original_filename="ai_crop_images.exe",
    product_name="ai_crop_images",
)

print("Done: versionfile.txt in current folder")
