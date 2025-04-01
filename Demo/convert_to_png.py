import os
from PIL import Image

# Specify the relative folder path
folder_path = "./images/mnist"

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".avif")):  # Added AVIF support
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        
        # Open the image
        try:
            with Image.open(file_path) as img:
                # Remove original extension and add ".png"
                png_filename = os.path.splitext(filename)[0] + ".png"
                png_path = os.path.join(folder_path, png_filename)
                
                # Save as PNG
                img.save(png_path, "PNG")
                print(f"Converted {filename} to {png_filename}")
            
            # Delete the original file
            os.remove(file_path)
            print(f"Deleted original file: {filename}")

        except Exception as e:
            print(f"Could not process {filename}. Error: {e}")

print("Conversion and cleanup complete!")
