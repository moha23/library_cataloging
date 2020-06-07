from text_detection import process_image_and_make_text_files
import sys
import glob
import os
from detect_books_ssd512 import detect_books
imgpath = 'path/to/image/of/interest'



path_to_sub_images = detect_books(imgpath)

for pngfile in glob.iglob(os.path.join(path_to_sub_images, "*.png")):
	#use OCR on each of these pngs
    img_path = pngfile
    ret_val = process_image_and_make_text_files(pngfile)
    print('the retval we get ', ret_val)
    
