import serial
from start import start_camera  

with serial.Serial('/dev/ttyAMA0', 115200, timeout=1) as ser:

  resp = ser.read()

  while(resp != "V"):
    resp = ser.read()
    print(resp)


  ser.write(b"v")
  print("video started")

  start_camera()
