import unittest
from pathlib import Path

import os, pty, serial, time, sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from run.listen_serial import listen_to_serial_port

class ListenSerialTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.master, slave = pty.openpty()
		s_name = os.ttyname(slave)
		self.test_conn = serial.Serial(s_name)

	def test_switch_from_tracking_to_sound_vision(self):
		listen_to_serial_port(self.test_conn)
		os.write(self.master, b'H')
		print("\n\n\n\nwrote 'H' to serial port\n\n\n\n")
		
		output = os.read(self.master, 1)
		self.assertEqual(output, b'h')

	def tearDown(self):
		video_file = Path("./output_video.mov")

		if video_file.exists():
			os.remove("./output_video.mov")


if __name__ == '__main__':
    unittest.main()
