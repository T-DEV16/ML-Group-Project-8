# ML-Group-Project-8: Pothole Detection on the Jetson Nano

## Data Preprocessing

* Our dataset consists of 2236 pairs of images. Each image is either 630 by 1024 or 640 by 1024.
In order to standardize, we scale each image down to 600 by 600. This also makes the training easier by decreasing the dimmenisions we input into our model.
* Once the image is loaded into our python environment as a PIL object, we convert to grayscale. This is actually only needed for the orginal images as they are in color and the pothole masks are already black and white.
* We then normalize the image by turning it into a numpy array and dividing by 255. 
* Our result is 2236 image pairs all of which are (600, 600) numpy arrays with values ranging from 0 to 1.

