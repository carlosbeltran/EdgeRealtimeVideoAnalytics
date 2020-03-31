# A Redis gear for orchestrating realtime video analytics
import io
import cv2
import redisAI
import numpy as np
from time import time
from PIL import Image

from redisgears import executeCommand as execute

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

# Globals for downsampling
_mspf = 1000 / 10.0      # Msecs per frame (initialized with 10.0 FPS)
_next_ts = 0             # Next timestamp to sample a frame

def process_image(img, height):
    ''' Utility to resize a rectangular image to a padded square (letterbox) '''
    color = (127.5, 127.5, 127.5)
    shape = img.shape[:2]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (height - new_shape[0]) / 2    # Width padding
    dh = (height - new_shape[1]) / 2    # Height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img = np.asarray(img, dtype=np.float32)
    img /= 255.0                        # Normalize 0..255 to 0..1.00
    return img

#def storeResults(x):
def storeYoloResults(ref_id,people,boxes):
    ''' Stores the results in Redis Stream and TimeSeries data structures '''
    global _mspf
    #ref_id, people, boxes= x[0], int(x[1]), x[2]
    ref_msec = int(str(ref_id).split('-')[0])

    # Store the output in its own stream
    res_id = execute('XADD', 'camera:0:yolo', 'MAXLEN', '~', 1000, '*', 'ref', ref_id, 'boxes', boxes, 'people', people)

    # Add a sample to the output people and fps timeseries
    res_msec = int(str(res_id).split('-')[0])
    execute('TS.ADD', 'camera:0:people', ref_msec, people)
    execute('TS.INCRBY', 'camera:0:out_fps', 1, 'RESET', 1)

    # Adjust mspf to the moving average duration
    total_duration = res_msec - ref_msec
    avg_duration = 40 # forced trying to get rid fo the profiler
    _mspf = avg_duration * 1.05  # A little extra leg room

    # Make an arithmophilial homage to Count von Count for storage in the execution log
    if people == 0:
        return 'Now there are none.'
    elif people == 1:
        return 'There is one person in the frame!'
    elif people == 2:
        return 'And now there are are two!'
    else:
        return 'I counted {} people in the frame! Ah ah ah!'.format(people)

def runYolo(x):
    ''' Runs the model on an input image from the stream '''
    IMG_SIZE = 416     # Model's input image size

    # log('read')

    # Read the image from the stream's message
    buf = io.BytesIO(x['image'])
    pil_image = Image.open(buf)
    numpy_img = np.array(pil_image)

    # log('resize')
    # Resize, normalize and tensorize the image for the model (number of images, width, height, channels)
    image = process_image(numpy_img, IMG_SIZE)
    # log('tensor')
    img_ba = bytearray(image.tobytes())
    image_tensor = redisAI.createTensorFromBlob('FLOAT', [1, IMG_SIZE, IMG_SIZE, 3], img_ba)

    # log('model')
    # Create the RedisAI model runner and run it
    modelRunner = redisAI.createModelRunner('yolo:model')
    redisAI.modelRunnerAddInput(modelRunner, 'input', image_tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'output')
    model_replies = redisAI.modelRunnerRun(modelRunner)
    model_output = model_replies[0]

    # log('script')
    # The model's output is processed with a PyTorch script for non maxima suppression
    scriptRunner = redisAI.createScriptRunner('yolo:script', 'boxes_from_tf')
    redisAI.scriptRunnerAddInput(scriptRunner, model_output)
    redisAI.scriptRunnerAddOutput(scriptRunner)
    script_reply = redisAI.scriptRunnerRun(scriptRunner)

    # log('boxes')
    # The script outputs bounding boxes
    shape = redisAI.tensorGetDims(script_reply)
    buf = redisAI.tensorGetDataAsBlob(script_reply)
    boxes = np.frombuffer(buf, dtype=np.float32).reshape(shape)

    # Iterate boxes to extract the people
    ratio = float(IMG_SIZE) / max(pil_image.width, pil_image.height)  # ratio = old / new
    pad_x = (IMG_SIZE - pil_image.width * ratio) / 2                  # Width padding
    pad_y = (IMG_SIZE - pil_image.height * ratio) / 2                 # Height padding
    boxes_out = []
    people_count = 0
    for box in boxes[0]:
        if box[4] == 0.0:  # Remove zero-confidence detections
            continue
        if box[-1] != 14:  # Ignore detections that aren't people
            continue
        people_count += 1

        # Descale bounding box coordinates back to original image size
        x1 = (IMG_SIZE * (box[0] - 0.5 * box[2]) - pad_x) / ratio
        y1 = (IMG_SIZE * (box[1] - 0.5 * box[3]) - pad_y) / ratio
        x2 = (IMG_SIZE * (box[0] + 0.5 * box[2]) - pad_x) / ratio
        y2 = (IMG_SIZE * (box[1] + 0.5 * box[3]) - pad_y) / ratio

        # Store boxes as a flat list
        boxes_out += [x1,y1,x2,y2]

    #return x['streamId'], people_count, boxes_out
    storeYoloResults(x['streamId'], int(people_count), boxes_out)
    return x

def runFaceDetection(x):

    # Read the image from the stream's message
    buf = io.BytesIO(x['image'])
    pil_image = Image.open(buf)
    numpy_img = np.array(pil_image)

    #img=cv2.imread('face.jpg')
    gray = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    boxes_out = []
    people_count = 0
    for (x,y,w,h) in faces:
        boxes_out += [x,y,x+w,y+h]
        people_count += 1
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray  = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        #eyes=eye_cascade.detectMultiScale(roi_gray)
        #for(ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh),(0,255,0),2)
        #        print(ex,ey)
        #cv2.imshow('img',img)
    ref_id = x['streamId']
    boxes= boxes_out
    people = int(people_count)
    # Store the output in its own stream
    res_id = execute('XADD', 'camera:0:facedect', 'MAXLEN', '~', 1000, '*', 'ref', ref_id, 'boxes', boxes, 'people', people)

# Create and register a gear that for each message in the stream
gb = GearsBuilder('StreamReader')
#gb.filter(downsampleStream)  # Filter out high frame rate
gb.map(runYolo)              # Run the model
#gb.map(storeResults)         # Store the results
gb.map(runFaceDetection)     # Run the face detection
gb.register('camera:0')
