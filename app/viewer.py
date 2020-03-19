# RedisEdge realtime video analytics web server
import argparse
import cv2
import io
import numpy as np
import redis
from urllib.parse import urlparse
from PIL import Image, ImageDraw
#from flask import Flask, render_template, Response

#from tkinter import *

class RedisImageStream(object):
    def __init__(self, conn, args):
        self.conn = conn
        self.camera = args.camera
        self.boxes = args.boxes
        self.field = args.field.encode('utf-8') 

    def get_last(self):
        ''' Gets latest from camera and model '''
        p = self.conn.pipeline()
        p.xrevrange(self.camera, count=1)  # Latest frame
        p.xrevrange(self.boxes, count=1)   # Latest boxes
        cmsg, bmsg = p.execute()
        if cmsg:
            last_id = cmsg[0][0].decode('utf-8')
            label = f'{self.camera}:{last_id}'
            data = io.BytesIO(cmsg[0][1][self.field])
            img = Image.open(data)
            if bmsg:
                boxes = np.fromstring(bmsg[0][1]['boxes'.encode('utf-8')][1:-1], sep=',')
                label += ' people: {}'.format(bmsg[0][1]['people'.encode('utf-8')].decode('utf-8'))
                for box in range(int(bmsg[0][1]['people'.encode('utf-8')])):  # Draw boxes
                    x1 = boxes[box*4]
                    y1 = boxes[box*4+1]
                    x2 = boxes[box*4+2]
                    y2 = boxes[box*4+3]
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(((x1, y1), (x2, y2)), width=5, outline='red')
            arr = np.array(img)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            cv2.putText(arr, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
            #ret, img = cv2.imencode('.jpg', arr)
            #return img.tobytes()
            return arr
        else:
            # TODO: put an 'we're experiencing technical difficlties' image
            pass

def videoDet(stream):

  #continuously processes video feed and displays in window until Q is pressed
  while True:
    
      #the read function gives two outputs. The check is a boolean function that returns if the video is being read
      #check, frame = video.read()
      frame = stream.get_last()
    
      cv2.imshow("Python Viewer", frame)

      #picks up the key press Q and exits when pressed
      key=cv2.waitKey(1)
      if key==ord('q'):
        break
  
  #Closes video window
  cv2.destroyAllWindows()

conn = None
args = None
#create an empty GUI window
#window=Tk()	

#app = Flask(__name__)

#@app.route('/video')
#def video_feed():
#    return Response(gen(RedisImageStream(conn, args)),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('camera', help='Input camera stream key', nargs='?', type=str, default='camera:0')
    parser.add_argument('boxes', help='Input model stream key', nargs='?', type=str, default='camera:0:yolo')
    parser.add_argument('--field', help='Image field name', type=str, default='image')
    parser.add_argument('--fmt', help='Frame storage format', type=str, default='.jpg')
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    args = parser.parse_args()

    # Set up Redis connection
    url = urlparse(args.url)

    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    #global stream
    stream = RedisImageStream(conn, args)

    videoDet(stream)
    
    #GUI tkinter button for starting the video capture
    #b1=Button(window, text="Start", command=videoDet)
    #b1.grid(row=0, column=0)

    #GUI widget label
    #l1=Label(window, text="Press Q to Stop Capturing")
    #l1.grid(row=0, column=1)

    #close GUI Window
    #window.mainloop()
    #app.run(host='0.0.0.0')
    '''
    while True:
    
        #the read function gives two outputs. The check is a boolean function that returns if the video is being read
        #check, frame = video.read()
        frame = stream.get_last()
        cv2.imshow("Capturing", frame)
        #picks up the key press Q and exits when pressed
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
  
    #Closes video window
    cv2.destroyAllWindows()
    '''
