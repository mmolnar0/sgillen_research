import cv2
import cv2.aruco as aruco
import numpy as np

#from scipy.signal import sawtooth

from numpy import sin, pi
import sounddevice as sd

cap = cv2.VideoCapture(0)

DICTIONARY  = aruco.DICT_6X6_1000
aruco_dict =  aruco.Dictionary_get(DICTIONARY)

if not cap.isOpened():
    print("something went wrong! video not open")
    raise SystemExit


re, img = cap.read()

cv2.namedWindow('Markers')
cv2.imshow('Markers', img)
cv2.moveWindow('Markers', 700, 0)


parameters = aruco.DetectorParameters_create()


# Parameters for sine wave
fs = 44100       # sampling rate, Hz
duration = .5   # in seconds
f = 15    # sine frequency, Hz

wave = abs((sin(2*pi*np.arange(fs*duration)*f/fs)).astype(np.float32))
#wave = sawtooth(2*pi*np.arange(fs*duration)*f/fs).astype(np.float32)
wave_play = wave.copy()


baseline = 70
static_gain = .01

while (True):
    re, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    found = aruco.drawDetectedMarkers(img, corners, ids)

    #print(corners)
    #print(ids)



    cv2.imshow('Markers', found)

    if corners:
        tracked_marker = corners[0].squeeze()
        x_track1 = tracked_marker[0, 0]
        x_track2 = tracked_marker[2, 0]

        print(abs(x_track1 - x_track2))
        gain = (abs(x_track1 - x_track2) - baseline)*static_gain

        wave_play = gain*wave
        print()
    #else:
        #wave_play = wave*0
        #sd.stop()

    #status = sd.wait()
    #print("here")
    sd.play(wave_play, fs, blocking=True)

    if cv2.waitKey(1) == ord('q'):
        break

sd.stop
cap.release()