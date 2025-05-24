# import cv2
# import numpy as np

# def create_binary_mask(image_path, detections, output_path):
#     """
#     Creates a binary mask for detected objects in an image and saves it.

#     Args:
#         image_path (str): Path to the input image.
#         detections (list): List of object detection results. 
#                            Each detection should be a dictionary containing 'mask' (segmentation mask).
#         output_path (str): Path to save the binary mask image.
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#          raise FileNotFoundError(f"Image not found at {image_path}")
    
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)

#     for detection in detections:
#         segmentation_mask = detection['mask']
#         contour = np.array(segmentation_mask, dtype=np.int32)
#         cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED) # Fill the contour with white (255)

#     cv2.imwrite(output_path, mask)
#     print(f"Binary mask saved to {output_path}")

# # Example usage (replace with your actual data)
# image_path = 'B:/Parking_Dec_Final/Screenshotcarp.png'
# detections = [
#     {'mask': [[975, 195],[975, 195],
# [975, 195],[774, 206],
# [250, 291],[412, 359],
# [250, 359],[413, 430],
# [250, 429],[413, 503],
# [248, 217],[411, 282],
# [250, 134],[410, 212],
# [61, 125],[250, 208],
# [65, 214],[248, 287],
# [63, 287],[249, 359],
# [67, 358],[248, 428],
# [66, 429],[251, 502],
# [69, 505],[250, 575],
# [246, 508],[415, 570],
# [69, 578],[250, 643],
# [250, 574],[412, 635],
# [74, 642],[255, 714],
# [249, 639],[454, 715],
# [255, 713],[426, 786],
# [258, 719],[73, 797],
# [597, 118],[766, 206],
# [771, 133],[952, 201],
# [769, 204],[935, 278],
# [767, 211],[610, 281],
# [605, 288],[769, 354],
# [773, 284],[939, 353],
# [936, 353],[776, 424],
# [768, 360],[606, 428],
# [606, 428],[776, 494],
# [774, 495],[934, 422],
# [932, 499],[773, 567],
# [770, 559],[605, 502],
# [610, 571],[770, 639],
# [770, 639],[935, 566],
# [934, 632],[774, 707],
# [774, 707],[611, 642],
# [1132, 129],[1305, 187],
# [1305, 187],[1126, 277],
# [1126, 277],[1302, 351],
# [1302, 351],[1129, 420],
# [1129, 420],[1302, 494],
# [1302, 494],[1124, 565],
# [1124, 565],[1301, 637],
# [1301, 637],[1138, 707],
# [1138, 707],[1295, 776],
# [1353, 275],[1506, 351],
# [1506, 351],[1363, 420],
# [1363, 420],[1514, 492],
# [1514, 492],[1356, 564],
# [1356, 564],[1508, 631],
# [1508, 631],[1357, 711],
# [1357, 711],[1510, 783],
# [1358, 209],[1510, 282]]}, # Example mask coordinates
#     # {'mask': [[300, 350], [400, 350], [400, 450], [300, 450]]}  # Another example mask
# ]
# output_path = 'Image_mask.png'

# try:
#     create_binary_mask(image_path, detections, output_path)
# except FileNotFoundError as e:
#     print(e)
# except Exception as e:
#     print(f"An error occurred: {e}")

import cv2
import numpy as np

def create_binary_mask(image_path, detections, output_path):
    """
    Creates a binary mask for detected objects in an image and saves it.

    Args:
        image_path (str): Path to the input image.
        detections (list): List of object detection results.
                            Each detection should be a dictionary containing 'mask' (segmentation mask).
        output_path (str): Path to save the binary mask image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Create an empty black mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for detection in detections:
        segmentation_mask = detection['mask']
        
        # Convert the list of points to a NumPy array
        # Ensure points are integers
        contour_points = np.array(segmentation_mask, dtype=np.int32)
        
        # Reshape the contour points to (N, 1, 2) as expected by cv2.drawContours
        # This explicitly tells OpenCV that each point is a single entry in the contour
        if contour_points.ndim == 2 and contour_points.shape[1] == 2:
            contour = contour_points.reshape((-1, 1, 2))
        else:
            print(f"Warning: Skipping malformed contour data: {segmentation_mask}")
            continue # Skip this detection if the mask format is incorrect

        # Draw the contour on the mask, filling it with white (255)
        # Using [contour] because drawContours expects a list of contours
        try:
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        except cv2.error as e:
            print(f"OpenCV error drawing contour: {e}")
            print(f"Problematic contour points: {contour}")
            # You might want to log this or handle it more gracefully
            continue

    cv2.imwrite(output_path, mask)
    print(f"Binary mask saved to {output_path}")

# Example usage (replace with your actual image path)
image_path = 'B:/Parking_Dec_Final/Screenshotcarp.png'
detections = [
    {'mask': [[975, 195],[975, 195],[975, 195],[774, 206],[250, 291],[412, 359],[250, 359],[413, 430],[250, 429],[413, 503],[248, 217],[411, 282],[250, 134],[410, 212],[61, 125],[250, 208],[65, 214],[248, 287],[63, 287],[249, 359],[67, 358],[248, 428],[66, 429],[251, 502],[69, 505],[250, 575],[246, 508],[415, 570],[69, 578],[250, 643],[250, 574],[412, 635],[74, 642],[255, 714],[249, 639],[454, 715],[255, 713],[426, 786],[258, 719],[73, 797],[597, 118],[766, 206],[771, 133],[952, 201],[769, 204],[935, 278],[767, 211],[610, 281],[605, 288],[769, 354],[773, 284],[939, 353],[936, 353],[776, 424],[768, 360],[606, 428],[606, 428],[776, 494],[774, 495],[934, 422],[932, 499],[773, 567],[770, 559],[605, 502],[610, 571],[770, 639],[770, 639],[935, 566],[934, 632],[774, 707],[774, 707],[611, 642],[1132, 129],[1305, 187],[1305, 187],[1126, 277],[1126, 277],[1302, 351],[1302, 351],[1129, 420],[1129, 420],[1302, 494],[1302, 494],[1124, 565],[1124, 565],[1301, 637],[1301, 637],[1138, 707],[1138, 707],[1295, 776],[1353, 275],[1506, 351],[1506, 351],[1363, 420],[1363, 420],[1514, 492],[1514, 492],[1356, 564],[1356, 564],[1508, 631],[1508, 631],[1357, 711],[1357, 711],[1510, 783],[1358, 209],[1510, 282]]}
]
output_path = 'Image_mask.png'

try:
    create_binary_mask(image_path, detections, output_path)
except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")