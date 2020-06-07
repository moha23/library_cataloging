# USAGE
# opencv-text-detection --image images/lebron_james.jpg

# import the necessary packages
import argparse
import os
import time
import imutils
import cv2
from nms import nms
import numpy as np
import pytesseract

import utils
from decode import decode
from draw import drawPolygons, drawBoxes
# from opencv_text_detection import utils
# from opencv_text_detection.decode import decode
# from opencv_text_detection.draw import drawPolygons, drawBoxes


def text_detection(image, east, min_confidence, width, height):
    # load the input image and grab the image dimensions
    image = cv2.imread(image)
    orig = image.copy()
    (origHeight, origWidth) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    ratioWidth = origWidth / float(newW)
    ratioHeight = origHeight / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (imageHeight, imageWidth) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    # print("[INFO] text detection took {:.6f} seconds".format(end - start))


    # NMS on the the unrotated rects
    confidenceThreshold = min_confidence
    nmsThreshold = 0.4

    # decode the blob info
    (rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)
    # print(len(rects))
    offsets = []
    thetas = []
    for b in baggage:
        offsets.append(b['offset'])
        thetas.append(b['angle'])

    ##########################################################

    # functions = [nms.felzenszwalb.nms, nms.fast.nms, nms.malisiewicz.nms]
    functions = [nms.felzenszwalb.nms]
    # print("[INFO] Running nms.boxes . . .")
    boxes = []
    for i, function in enumerate(functions):

        start = time.time()
        indicies = nms.boxes(rects, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                                 nsm_threshold=nmsThreshold)
        end = time.time()
        # print(indicies)
        indicies = np.array(indicies).reshape(-1)
        # print(indicies)
        drawrects = np.array(rects)[indicies]
        # print(drawrects)
        name = function.__module__.split('.')[-1].title()
        # print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(name, end - start, len(drawrects)))

        drawOn = orig.copy()
        drawBoxes(drawOn, drawrects, ratioWidth, ratioHeight, (0, 255, 0), 2)

        # title = "nms.boxes {}".format(name)
        # cv2.imshow(title,drawOn)
        # cv2.moveWindow(title, 150+i*300, 150)

    # cv2.waitKey(0)


    # convert rects to polys
    polygons = utils.rects2polys(rects, thetas, offsets, ratioWidth, ratioHeight)
    # print(len(polygons[0][0]))

    # print("[INFO] Running nms.polygons . . .")

    for i, function in enumerate(functions):

        start = time.time()
        indicies = nms.polygons(polygons, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                                 nsm_threshold=nmsThreshold)
        end = time.time()

        indicies = np.array(indicies).reshape(-1)
        # print(indicies)
        drawpolys = np.array(polygons)[indicies]
        # print(drawpolys)
        name = function.__module__.split('.')[-1].title()

        # print("[INFO] {} NMS took {:.6f} seconds and found {} boxes".format(name, end - start, len(drawpolys)))

        drawOn = orig.copy()
        drawPolygons(drawOn, drawpolys, ratioWidth, ratioHeight, (0, 255, 0), 2)

        # title = "nms.polygons {}".format(name)
        # cv2.imshow(title,drawOn)
        # cv2.moveWindow(title, 150+i*300, 150)

    # cv2.waitKey(0)
    return drawpolys

orig_path = os.getcwd()
def process_image_and_make_text_files(img_path, path_to_east_detector='/Users/momo/Downloads/opencv_text_detection/frozen_east_text_detection.pb', min_confidence = 0.45,width=320, height=320, padding=0):
	# print(img_path)
	boxes = text_detection(img_path, path_to_east_detector, min_confidence, width, height)
	image = cv2.imread(img_path)
	orig = image.copy()
	(origH, origW) = image.shape[:2]
	(newW, newH) = (width, height)
	rW = origW / float(newW)
	rH = origH / float(newH)
	results = []
	img_name, y = img_path.rsplit('/', 1)
	# print('folder_name_of_sent_image')
	# print(img_name)

	# path=os.path.join(my_path, str(y))
	# print(path)
	# return 
	path = os.path.join(orig_path, 'worked')
	ticks = time.time()
	y = y + str(ticks)
	path = os.path.join(path, y)
	try:
		os.makedirs(path)
	except FileExistsError:
		pass
	ret_path = path
	os.chdir(path)
	# print('The new folder path: ', os.getcwd(), sep = ' ')
	# return
	dir_name, img_name = img_path.rsplit('/', 1)
	file_name = "ocr_text_" + img_name + "_" + ".txt"
	with open(file_name, 'w+') as f:
		for (startX, startY, endX, endY) in boxes:
			x_list = []
			y_list = []

			x_list.append(startX[0])
			x_list.append(startY[0])
			x_list.append(endX[0])
			x_list.append(endY[0])

			y_list.append(startX[1])
			y_list.append(startY[1])
			y_list.append(endX[1])
			y_list.append(endY[1])

			startX = int(min(x_list))
			endX = int(max(x_list))
			startY = int(min(y_list))
			endY = int(max(y_list))

			dX = int((endX - startX) * padding)
			dY = int((endY - startY) * padding)

			startX = max(0, startX - dX)
			startY = max(0, startY - dY)
			endX = min(origW, endX + (dX * 2))
			endY = min(origH, endY + (dY * 2))

			roi = orig[startY:endY, startX:endX]

			config = ("-l eng --oem 1 --psm 6")
			text = pytesseract.image_to_string(roi, config=config)
			text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
			f.write(text)
			f.write(' ')

	return os.path.join(ret_path, file_name)





# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str,
#     help="path to input image")
# ap.add_argument("-east", "--east", type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'frozen_east_text_detection.pb'),
#     help="path to input EAST text detector")
# ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
#     help="minimum probability required to inspect a region")
# ap.add_argument("-w", "--width", type=int, default=320,
#     help="resized image width (should be multiple of 32)")
# ap.add_argument("-e", "--height", type=int, default=320,
#     help="resized image height (should be multiple of 32)")
# ap.add_argument("-p", "--padding", type=float, default=0.0,
# 	help="amount of padding to add to each border of ROI")
# ap.add_argument("-q", "--poly", type=int, default=0,
# 	help="amount of padding to add to each border of ROI")
# args = vars(ap.parse_args())
# path = '/home/rajanish/hackathon/tomhoag-opencv-text-detection-dbe095275253/'
# image = os.path.join(path, args["image"])
# boxes = text_detection(image, east=args["east"], min_confidence=args['min_confidence'], width=args["width"], height=args["height"], )
# print(len(boxes))
# path1 = '/home/rajanish/hackathon/tomhoag-opencv-text-detection-dbe095275253/'
# image = os.path.join(path1, args["image"])
# image = cv2.imread(image)
# # rotated = imutils.rotate_bound(image, 90)
# # cv2.imshow('stupid', rotated)
# orig = image.copy()
# (origH, origW) = image.shape[:2]
# # if(origW < origH):
# #     print('activated')
# #     rotated = imutils.rotate_bound(image, 90)
# # 	# cv2.imshow("Rotated (Problematic)", rotated)
# # 	# cv2.waitKey(0)
# #     image = rotated
# #     orig = rotated

# # for (x, y, z, w) in boxes:
# # 	print(x, y, z, w, sep=' ')
# # set the new width and height and then determine the ratio in change
# # for both the width and height
# (newW, newH) = (args["width"], args["height"])
# rW = origW / float(newW)
# rH = origH / float(newH)

# # boxes = text_detection_command()
# results = []
# def my_crop_fun(img, pts):
# 	rect = cv2.boundingRect(pts)
# 	x,y,w,h = rect
# 	croped = img[y:y+h, x:x+w].copy()

# ## (2) make mask
# 	pts = pts - pts.min(axis=0)

# 	mask = np.zeros(croped.shape[:2], np.uint8)
# 	cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

# ## (3) do bit-op
# 	dst = cv2.bitwise_and(croped, croped, mask=mask)

# ## (4) add the white background
# 	bg = np.ones_like(croped, np.uint8)*255
# 	cv2.bitwise_not(bg,bg, mask=mask)
# 	dst2 = bg+ dst
# 	return dst
# for (startX, startY, endX, endY) in boxes:
# 	# scale the bounding box coordinates based on the respective
# 	# ratios
# 	# print(startX, startY, endX, endY, sep=' ')


# 	# Perfectly fine code; doing something stupid
	
# 	# print(roi)
# 	if(args["poly"] == 1):
# 		pts = []
# 		pts.append([int(startX[0]), int(startX[1])])
# 		pts.append([int(startY[0]), int(startY[1])])
# 		pts.append([int(endX[0]), int(endX[1])])
# 		pts.append([int(endY[0]), int(endY[1])])
# 		pts = np.array(pts)
# 		print(pts)
# 		roi = my_crop_fun(orig, pts)
# 	else:
# 		x_list = []
# 		y_list = []
# 		x_list.append(startX[0])
# 		x_list.append(startY[0])
# 		x_list.append(endX[0])
# 		x_list.append(endY[0])

# 		y_list.append(startX[1])
# 		y_list.append(startY[1])
# 		y_list.append(endX[1])
# 		y_list.append(endY[1])
# 		# # print('here we go')
# 		# # print(x_list)
# 		# # print('here xe go')
# 		# # print(y_list)
# 		startX = int(min(x_list))
# 		endX = int(max(x_list))
# 		startY = int(min(y_list))
# 		endY = int(max(y_list))
# 		# print('wow')
# 		# print(startX, endX, startY, endY, sep=' ')
# 		# startX = int(startX * rW)
# 		# startY = int(startY * rH)
# 		# endX = int(endX * rW)
# 		# endY = int(endY * rH)
# 		# print(startX, endX, startY, endY, sep=' ')
# 		# # in order to obtain a better OCR of the text we can potentially
# 		# # apply a bit of padding surrounding the bounding box -- here we
# 		# # are computing the deltas in both the x and y directions
# 		dX = int((endX - startX) * args["padding"])
# 		dY = int((endY - startY) * args["padding"])

# 		# # apply padding to each side of the bounding box, respectively
# 		startX = max(0, startX - dX)
# 		startY = max(0, startY - dY)
# 		endX = min(origW, endX + (dX * 2))
# 		endY = min(origH, endY + (dY * 2))
# 		# # print(orig.shape)
# 		# # extract the actual padded ROI
# 		if(startX > endX) :
# 			startX, endX = endX, startX
# 		if(startY > endY):
# 			startY, endY = endY, startY
# 		# print(startX, endX, startY, endY, sep=' ')
# 		roi = orig[startY:endY, startX:endX]


	
# 	# in order to apply Tesseract v4 to OCR text we must supply
# 	# (1) a language, (2) an OEM flag of 4, indicating that the we
# 	# wish to use the LSTM neural net model for OCR, and finally
# 	# (3) an OEM value, in this case, 7 which implies that we are
# 	# treating the ROI as a single line of text
# 	# config = ("-l eng --oem 1 --psm 7")
# 	config = ("-l eng --oem 1 --psm 6")
# 	text = pytesseract.image_to_string(roi, config=config)
# 	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

# 	dir_name, img_name = args["image"].rsplit('/', 1)
# 	with open("res_" + img_name + "_" + str(args["poly"]) + '.txt', 'a+') as f:
# 		f.write(text)
# 		f.write(' ')
# 	# print(text)
# 	# add the bounding box coordinates and OCR'd text to the list
# 	# of results
# 	results.append(((startX, startY, endX, endY), text))

# # sort the results bounding box coordinates from top to bottom
# # results = sorted(results, key=lambda r:r[0][1])
# # Rresults = results[::-1]
# # loop over the results
# if(args["poly"] != 1):
# 	for ((startX, startY, endX, endY), text) in results:
# 		# display the text OCR'd by Tesseract
# 		print("OCR TEXT")
# 		print("========")
# 		print("{}\n".format(text))

# 		# strip out non-ASCII text so we can draw the text on the image
# 		# using OpenCV, then draw the text and a bounding box surrounding
# 		# the text region of the input image
# 		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
# 		output = orig.copy()
# 		cv2.rectangle(output, (startX, startY), (endX, endY),
# 			(0, 0, 255), 2)
# 		cv2.putText(output, text, (startX, startY - 20),
# 			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# 		# show the output image
# 		cv2.imshow("Text Detection", output)
# 		cv2.waitKey(0)