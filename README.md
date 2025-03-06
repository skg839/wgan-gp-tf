Setup and Installation:

1) Create a New Virtual Environment
Using venv (Python 3.x):
```
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```
Or using conda:
```
conda create --name custom_wgan_gp python=3.8
conda activate custom_wgan_gp
```

2) Install Dependencies
After activating your virtual environment, install the required packages:
```
pip install -r requirements.txt
```
Usage:
1) Prepare your dataset by placing your JPG or PNG images in a directory
2) Run the training script by specifying the path to your dataset
```
python main.py --data_dir path/to/your/dataset
```
3) Use python main.py --help to view additional options like number of epochs, batch size, and more.

Project Structure:
- main.py: Main training script
- requirements.txt: List of project dependencies
- checkpoints/: Directory for saving model checkpoints
- samples/: Directory for saving generated image samples
