Background Remover and Object Processor
A Python tool for automatically removing backgrounds from images, centering objects, and applying professional light effects.
Features

Remove backgrounds from images using deep learning
Automatically detect object boundaries in transparent images
Center and resize objects with customizable parameters
Apply professional daylight effects similar to Photoshop
Support for various image formats including RAW formats

Installation
bash# Clone the repository
git clone https://github.com/yourusername/background-remover.git
cd background-remover

# Install dependencies
pip install -r requirements.txt

# Download the model
# The model file (isnet_dis.onnx) should be placed in the models/ directory
Usage
bash# Basic usage
python main.py --input path/to/image.jpg --output path/to/output.png

# Advanced usage with parameters
python main.py --input path/to/image.jpg --output path/to/output.png --size 1600 1600 --factor 0.94 --brightness 1.05
Requirements

Python 3.6+
OpenCV
PIL/Pillow
NumPy
Matplotlib
dis_bg_remover (Background removal package)

Project Structure

main.py: Entry point script
src/: Source code modules

image_processor.py: Core image processing functions
effects.py: Visual enhancement functions
utils.py: Utility functions


models/: Directory for model files

License
MIT
