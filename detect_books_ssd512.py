from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import os

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

def detect_books(img_path):
    # Set the image size.
    img_height = 512
    img_width = 512

    K.clear_session() # Clear previous models from memory.

    model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=80,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales= [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0/3.0], [1.0, 2.0, 0.5, 3.0, 1.0/3.0], [1.0, 2.0, 0.5, 3.0, 1.0/3.0],[1.0, 2.0, 0.5, 3.0, 1.0/3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 128, 256, 512],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    # 2: Load the trained weights into the model.

    # TODO: Set the path of the trained weights.
    weights_path = '/path/to/weights/of/SSD/VGG_coco_SSD_512x512_iter_360000.h5'

    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    orig_images.append(imread(img_path))
    workon_image=cv2.imread(img_path)
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    #Make predictions
    y_pred = model.predict(input_images)


    confidence_threshold = 0.3

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[-------'--------0])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
    classes = ['background','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

    plt.figure(figsize=(20,12))
    plt.imshow(orig_images[0])

    current_axis = plt.gca()
    
    j=0
    for box in y_pred_thresh[0]:
        j+=1
        # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
        xmin = box[-4] * orig_images[0].shape[1] / img_width
        ymin = box[-3] * orig_images[0].shape[0] / img_height
        xmax = box[-2] * orig_images[0].shape[1] / img_width
        ymax = box[-1] * orig_images[0].shape[0] / img_height

        if (int(box[0])==74):
            x_min=math.floor(int(xmin))
            y_min=math.floor(int(ymin))
            y_max=math.floor(int(ymax))
            x_max=math.floor(int(xmax))
            my_path = os.getcwd()
            img_name, y = img_path.rsplit('/', 1)
            path=os.path.join(my_path, str(y))
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
            book = workon_image[y_min:y_max,x_min:x_max]
            file_name = y + '_bounded_img_' + str(j) + '.png'
            cv2.imwrite(os.path.join(path,file_name),book)
            
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    plt.show()
    return path
