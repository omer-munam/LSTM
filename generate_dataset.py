import cv2
import time
vid_capture = cv2.VideoCapture(0)
vid_cod = cv2.VideoWriter_fourcc(*'MP4V')
reg_num = input("Enter Reg Number:  ")
output1 = cv2.VideoWriter("./videos/"+reg_num+".mp4", vid_cod, 20.0, (640,480))

length = 0
while(length<240):
     time.sleep(0.5)
     ret,frame = vid_capture.read()
     output1.write(frame)
     frame = cv2.putText(frame, str(length), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
     cv2.imshow("My cam video", frame)
     length += 1 
     if cv2.waitKey(1) &0XFF == ord('t'):
         break

vid_capture.release()
output1.release()
cv2.destroyAllWindows()