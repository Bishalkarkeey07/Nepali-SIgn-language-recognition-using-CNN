import cv2
from cv2 import CamShift
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model



# Load the sign language recognition model
model = load_model(r"C:\Users\97798\Model\model123.h5")

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

# Define the labels for the sign language classes
# word_dict = {0:'ka',1:'ksha',2:'kha',3:'ga',4:'gha',5:'nga',6:'cha',7:'chha',8:'ja',9:'gya',10:'jha',11:'yan',12:'ta',13:'tha',14:'da',15:'dha',16:'naa',17:'Tah',18:'tra',19:'thao',20:'daa',21:'dhaa',22:'na',23:'pa',24:'pha',25:'ba',26:'bha',27:'ma',28:'ya',29:'ra',30:'la',31:'wa',32:'sha',33:'shha',34:'sa', 35: 'ha'}

word_dict = {0:'क',1:'क्ष',2:'ख',3:'ग',4:'घ',5:'ङ',6:'च',7:'छ',8:'ज',9:'ज्ञ',10:'झ',11:'ञ',12:'ट',13:'ठ',14:'ड',15:'ढ',16:'ण',17:'त',18:'त्र',19:'थ',20:'द',21:'ध',22:'न',23:'प',24:'फ',25:'ब',26:'भ',27:'म',28:'य',29:'र',30:'ल',31:'व',32:'श',33:'ष',34:'स',35:'ह'}
def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)
def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
     #Fetching contours in the frame (These contours can be of hand or any other object in foreground) …

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and thethresholded image of hand...
        return (thresholded, hand_segment_max_cont)
root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("1000x1000")

# Create a canvas to display the video feed
canvas = tk.Canvas(root, width=600, height=450)
canvas.pack(side=tk.LEFT)
title_label = tk.Label(root, text="Predicted Sign Language", font=("Arial", 20))
title_label.pack(side=tk.TOP, pady=(250, 0))

# Create a label to display the predicted label
label = tk.Label(root, font=("Arial", 170))
label.pack(side=tk.RIGHT, padx= 20, pady=(0,250))

    

# Define the function to recognize the sign language
def recognize_sign_language():
    cam = cv2.VideoCapture(0)
    
    num_frames =0
    while True:
        ret, frame = cam.read()

        # flipping the frame to prevent inverted image of captured frame...
        
        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)


        if num_frames < 70:
            
            cal_accum_avg(gray_frame, accumulated_weight)
            
            cv2.putText(frame_copy, "FETCHING BACKGROUND...WAIT",
            (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else: 
            
            # segmenting the hand region
            hand = segment_hand(gray_frame)
            
            # Checking if we are able to detect the hand...
            
            if hand is not None:
                    
                    
                    thresholded, hand_segment = hand
                    # Drawing contours around hand segment
                    cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
                    ROI_top)], -1, (255, 0, 0),1)
                
                    
                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))
                    pred = model.predict(thresholded)
                    label_index = np.argmax(pred)
                    pred_label = word_dict[label_index]
                    
                    

                    


                # Display the predicted label on the label window
    
                    label.config(text=pred_label)
               
            




        # Draw ROI on frame_copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
        ROI_bottom), (255,128,0), 3)

        # incrementing the number of frames for tracking
        num_frames += 1
        # frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (600, 450))
        # # Display the frame with segmented hand
        image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        # img = Image.fromarray(frame_copy)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        root.update()


        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

# Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()
recognize_sign_language()

# Run the tkinter event loop
root.mainloop()

