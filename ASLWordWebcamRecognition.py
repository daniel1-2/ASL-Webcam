import cv2
import time
import os
import shutil
import random
import mediapipe as mp
import tensorflow as tf
import model_communication

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

"""
SUPER IMP: 

MODEL WAS TRAINED ON RIGHT HANDED IMAGES
"""


# Setting countdown timer

TIMER = int(3)

# SET THE COUNTDOWN TIMER
# for simplicity we set it to 3
# We can also take this as input
TIMER = int(3)

font = cv2.FONT_HERSHEY_SIMPLEX
# Open the camera
cap = cv2.VideoCapture(0)

#loading in the ML model
model = tf.keras.models.load_model('model_SIBI.h5')

# Hard Encode for the Prediction

classes = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'K',
    10: 'L',
    11: 'M',
    12: 'N',
    13:'O',
    14:'P',
    15:'Q',
    16:'R',
    17:'S',
    18:'T',
    19:'U',
    20:'V',
    21:'W',
    22:'X',
    23:'Y'

}


def countdown(righthanded, prev, TIMER, font, position, path, STOP):
    #initiates the count down
    count = 0
    while ((TIMER > 0) & (STOP == False)):
        keyInput = cv2.waitKey(10)
        if keyInput == 27:
            #this will stop the countdown, terminating the program
            STOP = True

        ret, img = cap.read()
        imgFlipped = cv2.flip(img, 1)

        #MediaPipe annotating the image as countdown is occurring
        imgFlipped.flags.writeable = False
        imgFlipped = cv2.cvtColor(imgFlipped, cv2.COLOR_BGR2RGB)
        results = hands.process(imgFlipped)

        imgFlipped.flags.writeable = True
        imgFlipped = cv2.cvtColor(imgFlipped, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    imgFlipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        #"position" is actually being used as a font color
        cv2.putText(imgFlipped, str(TIMER),
                    (520,260), font,
                    7, position,
                    4, cv2.LINE_AA)
        #displaying user's image back to them
        cv2.imshow('Webcam Input', imgFlipped)
        keyInput = cv2.waitKey(10)

        # current time
        cur = time.time()

        # Update and keep track of Countdown
        # if time elapsed is one second
        # than decrease the counter
        if cur-prev >= 1:
            prev = cur
            TIMER = TIMER-1


    cv2.destroyWindow('Webcam Input')
    #makes a directory to store all pictures
    os.mkdir(path)
    os.chdir(path)

    #this is the actual part where user will do signs
    while cap.isOpened() & STOP == False:
        #if user hits esc, then the loop is broken


        ret, img = cap.read()
        imgFlipped = cv2.flip(img,1)

        #MediaPipe's annotation of the flipped image
        imgFlipped.flags.writeable = False
        imgFlipped = cv2.cvtColor(imgFlipped, cv2.COLOR_BGR2RGB)
        results = hands.process(imgFlipped)

        imgFlipped.flags.writeable = True
        imgFlipped = cv2.cvtColor(imgFlipped, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    imgFlipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style() )


        cv2.putText(imgFlipped, str(int(count/30)),
                (50, 50 ), font,
                2, position,
                2, cv2.LINE_AA)

        #every 3 pictures are being sent to classification:
        if count % 90 == 0:

            if righthanded:
                # saving the image of them since the model works off of an image path only:
                cv2.imwrite('frame{:d}.jpg'.format(count), img)
                #sending it to the model for prediction:
                classified_letter = model_communication.model("frame{:d}.jpg".format(count), model)
                #removing the picture after model has used it
                os.remove('frame{:d}.jpg'.format(count))
            else:
                #if they are left handed, then I want to flip the image and send it into model
                cv2.imwrite('frame{:d}.jpg'.format(count), imgFlipped)
                # sending it to the model for prediction:
                classified_letter = model_communication.model("frame{:d}.jpg".format(count), model)
                # removing the picture after model has used it
                os.remove('frame{:d}.jpg'.format(count))

        #writing the prediction on webcam display
        cv2.putText(imgFlipped, str(classified_letter),
                    (0, 700), font,
                    4, position,
                    4, cv2.LINE_AA)

        #showing them the image
        cv2.imshow("Snapshot", imgFlipped)

        count += 30
        cap.set(1, count)
        #removing pictures after every 5 are taken
        # if (count >= ) and (count % 90 == 0) :
        #     os.remove('frame{:d}.jpg'.format(count/3))
        # time for which image displayed
        keyInput = cv2.waitKey(20)
        if keyInput == 27:
            STOP = True

    #cleaning up the folder with the pictures
    os.chdir('../')
    shutil.rmtree(path)
    cv2.destroyWindow('Snapshot')




#setting up MP Hands

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    STOP = False

    if os.path.exists('Webcam_Pictures/'):
        #if the user already has a file path called Webcam_Pictures, we want to
        #create a unique new file path that they do not have
        path = "Webcam_Pictures_uniquename" + str(random.random()) + str(random.random())
    else:
        path = "Webcam_Pictures"
    while STOP == False:
        
        # Read and display each frame
        ret, img = cap.read()
        
        imgFlipped = cv2.flip(img, 1)


        #putting mediapipe onto hands:
        imgFlipped.flags.writeable = False
        imgFlipped = cv2.cvtColor(imgFlipped, cv2.COLOR_BGR2RGB)
        results = hands.process(imgFlipped)


        imgFlipped.flags.writeable = True
        imgFlipped = cv2.cvtColor(imgFlipped, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    imgFlipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


        #Introduction Text on Screen
        position = ( (int) (img.shape[1]/2) - 450, (int) (img.shape[0]/2 - 100))
        position_shifted_less_down = ((int)(img.shape[1] / 2) - 425, (int)(img.shape[0] / 2))
        position_shifted_down = ( (int) (img.shape[1]/2 - 250), (int) (img.shape[0]/2 + 100))
        cv2.putText(imgFlipped, 'Right-Handed Users Please Type "R"',
                    position, font, 1.5, (218,112,214), 3, cv2.LINE_AA)
        cv2.putText(imgFlipped, 'Left-Handed Users Please Type "L"',
                    position_shifted_less_down, font, 1.5, (218, 112, 214), 3, cv2.LINE_AA)
        cv2.putText(imgFlipped, "At Any Time Press Esc To Quit", position_shifted_down , font, 1, (218,112,214), 3, cv2.LINE_AA)
        cv2.imshow('Webcam Input', imgFlipped)

        # check for the key pressed
        keyInput = cv2.waitKey(30)


        # set the key for the countdown
        # to begin. If "R" is pressed, it starts
        # and we note that they are righthanded
        if keyInput == ord('r'):
            prev = time.time()
            righthanded = True
            countdown(righthanded, prev, TIMER, font, position, path, STOP)

        if keyInput == ord('l'):
            prev = time.time()
            righthanded = False
            countdown(righthanded, prev, TIMER, font, position, path, STOP)
            # Press Esc to exit at any point
        elif keyInput == 27:
            break

# close the camera
cap.release()

# close all the opened windows
cv2.destroyAllWindows() 




##DIFFERENT SOLUTION


#Create an object to hold reference to camera video capturing
# vidcap = cv2.VideoCapture(0)

# #check if connection with camera is successfully
# while vidcap.isOpened():
#     ret, frame = vidcap.read()  #capture a frame from live video

#     #check whether frame is successfully captured
#     if ret:
#         # continue to display window until 'q' is pressed
#         morePictures = True
#         while(morePictures):
#             cv2.imshow("Frame",frame)   #show captured frame
            
#             #press 'q' to break out of the loop
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             # if '):
#             #     morePictures = False
#     #print error if frame capturing was unsuccessful
#     else:
#         print("Error : Failed to capture frame")

# # print error if the connection with camera is unsuccessful
# else:
#     print("Cannot open camera")
