import os
import cv2


def video_to_frame(session_id, dir, frameRate):
    vidcap = cv2.VideoCapture(dir)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        if (count % frameRate == 0):
            if os.path.isfile("result/" + session_id + "/frames/" + session_id + "_" + str(count) + ".jpg") == False:
                cv2.imwrite("result/"+session_id+"/frames/"+session_id+"_"+str(count)+".jpg",
                        image)  # save frame as JPEG file
                print('Save frame: ', count, ': ', success)
            else:
                print("Skipped frame ",count)
        success, image = vidcap.read()
        #print('Read a new frame: ', count, ': ', success)
        count += 1
