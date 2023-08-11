import sys

try:
    sys.path.append("./")
    from ai_crop_images.main import cli
except ImportError:
    sys.path.append("../")
    from ai_crop_images.main import cli

if __name__ == "__main__":
    cli()
