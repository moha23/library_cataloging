## Problem Statement: Automated Library Cataloging. 

Catalog information about all books in the library from the images of the books.

## Dataset: 

We have SSD 512 trained on MS COCO dataset in which one of the 80 categories is book. For testing we use pictures collected from IIT Mandi Central Library.

## Procedure:

1. We have a trained SSD 512 on MS COCO dataset. When an image is given, it gives the detections. We take the bounding box coordinates and get only the patches which have books and save them.

2. Next we feed this sub images to our OCR model which extracts the text in the book (name/author etc).

3. The generated text files are used to do an automated google search,since detections don't give exact names, but google search auto completes and gives pretty accurate results.

4. Next from the amazon page of the target book we get required book info into saved data structures.

[“Area”, “ISBN Number”, “Author Name”, “Review”, “Cost”].

## Architecture:

![Architecture](https://i.ibb.co/s2jYDCW/reallyoutofnames.jpg)


## Some early attempts and descriptions:

https://docs.google.com/document/d/e/2PACX-1vQCQ6BUoEp1tHXESG443eILrcmkWYlr9mrw0ekMwg37D25eFJebwuI_zK7tO2kzE6OYpF7mf5jUtfLn/pub

## Running the code:

Main file to be run is library_catalog_main.py to which we need to provide the path of the image we want to work on. Rest of the pipeline is automated.

## Prerequisites

- Python 3.6
- Tensorflow 1.13.1
- Matplotlib: 3.0.2
- Scikit-image
- Numpy 1.16.1
- Pandas 0.24.1
- math, os, glob
- openCV
- imageio
- pytesseract
- tesseract

###### Note
This was a submission for a Hackathon organised as part of CS-671 Deep Learning and its Applications offered at IIT Mandi, along with my group members Preethi Srinivasan and Rajneesh Upadhyay.
