#Original C source: Copyright (C) Peter B.L. Meijer
#http://www.seeingwithsound.com/im2sound.htm
#Check site for license details.

import math
import struct
import ctypes
import numpy as np
import cv2 as cv

#using PyAL
#from openal import al, alc
import sounddevice as sd

sd.default.samplerate = 21000
sd.default.dtype = 'int16'
sd.default.channels = 1

VIEW = True
FL       =   500   # Lowest  frequency (Hz) in soundscape
FH       =  5000   # Highest frequency (Hz)              
FS       = 44100   # Sample  frequency (Hz)              
T        =  1.05   # Image to sound conversion time (s)  
D        =     1   # Linear|Exponential=0|1 distribution 
HIFI     =     1   # 8-bit|16-bit=0|1 sound quality      

class Example(object):
	def __init__(self):
		# self.listener = Listener()
		# self.sbuffer = buffer_sound()
		self.render = audio_render()
		# self.player = Player()

		try:
			cap = cv.VideoCapture(0)   
			if not cap.isOpened(): 
				raise ValueError('camera ID')
		except ValueError:
			print("Could not open camera", 0)
			raise

		# Setting standard capture size, may fail; resize later
		frame = cap.read()  # Dummy read needed with some devices
		  
		if VIEW:  # Screen views only for debugging
			cv.namedWindow('Large', cv.WINDOW_AUTOSIZE);
			cv.namedWindow('Small', cv.WINDOW_AUTOSIZE);
		  
		key = 0
		while key != 27:  # Escape key
			ret, frame = cap.read()
	   
			if ret == False:
				# Sometimes initial frames fail
				print("Capture failed\n")
				key = cv.waitKey(100)
				continue
			 
			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

			output = self.render.generate(gray)
			sd.play(output, 21000)

			#self.sbuffer.load(self.render.output)
			#self.player.add(self.sbuffer)
			#self.player.play()

		#self.player.remove()
		#self.player.delete()
		#self.sbuffer.delete()
		#self.listener.delete()



class audio_render(object):
#pre-calculate waveform values
	def __init__(self):
		self.speed = 1.5

		self.byte = int(20000 * self.speed)
	#size of image(s)
		self.size = 720

		self.ns = self.byte
		self.scale = 0.5/math.sqrt(self.size)
		self.TwoPi = 6.283185307179586476925287 #double Pi
		self.rnd = 0.211321159122
		self.phi0 = self.TwoPi * self.rnd

	#Generate sine wave
		self.w = np.asarray(range(self.size)) 
		self.w = self.TwoPi * FL * pow(1.0* FH/FL,1.0*self.w/(self.size-1))
		#self.w = self.TwoPi * 500 * np.power(1.0*5000/500, 1.0*self.w/(self.size-1))
		self.w = np.tile(self.w,(self.ns,1))

		self.core = np.asarray(range(self.ns)) #312.5 * data(4096), repeat
		self.core = np.reshape(self.core,(self.byte,1))
		self.core = np.repeat(self.core,(self.size),axis=1)

		self.w = np.sin(self.w * (self.core * (1.0 / self.byte)) + self.phi0)

		self.output = (ctypes.c_short * self.ns)()

#load and convert image, multiply it by waveform and load into ctypes array for ouput
	def generate(self,image):
		self.data = image
	# 	rawimage = self.image.get_image_data()
	# 	pixels = rawimage.get_data('RGBA',rawimage.width*4)
	# #convert image data to numpy array int16
	# 	self.data = np.fromstring(pixels, dtype='uint8')

	# #reshape array into 4 by (width*height) array, then strip color data.
	#	self.data = np.reshape(self.data,(len(self.data)/4,4))
	# 	self.data = self.data[::,:1:]

	#reshape array into X by Y grid to match image X/Y
		# self.data = np.reshape(self.data,(self.size,self.size))
	#convert image data to floating point.
		self.data = self.data[0:self.size, 0:self.size]
		self.data = self.data / 255.0
	#expand image data to match size of wav sample
		self.data = np.repeat(self.data,int(np.ceil(self.byte / float(self.size))),axis=1) 
		self.data = self.data[::,:self.byte:]

		self.data = np.swapaxes(self.data,0,1)
		self.data = np.multiply(self.data,self.w) * 1500.0

		self.data = np.sum(self.data,axis=1)

	#add click tone to beginning
		self.data[:64] = ((2.0 * self.rnd - 1.0) / self.scale) * 32767.0 #pre-compile click sound math earlier.

		self.output = (ctypes.c_short * self.ns)()

		for a in range(0,len(self.output),1):
			self.output[a] = self.output[a]+int(self.data[a])

		return self.output


# class buffer_sound(object):
# 	def __init__(self):
# 		self.channels = 1
# 		self.bitrate = 16
# 		self.samplerate = 21000
# 		self.wavbuf = None
# 		self.alformat = al.AL_FORMAT_MONO16
# 		self.length = None
# ##        formatmap = {
# ##            (1, 8) : al.AL_FORMAT_MONO8,
# ##            (2, 8) : al.AL_FORMAT_STEREO8,
# ##            (1, 16): al.AL_FORMAT_MONO16,
# ##            (2, 16) : al.AL_FORMAT_STEREO16,
# ##        }
# ##        alformat = formatmap[(channels, bitrate)]


# 		self.buf = al.ALuint(0)
# 		al.alGenBuffers(1, self.buf)

# 	def load(self,data):
# 		self.wavbuf = data
# 		self.length = len(data)
# 	#allocate buffer space to: buffer, format, data, len(data), and samplerate
# 		al.alBufferData(self.buf, self.alformat, self.wavbuf, len(self.wavbuf)*2, self.samplerate)

# 	def delete(self):
# 		al.alDeleteBuffers(1, self.buf)



# #load a listener to load and play sounds.
# class Listener(object):
# 	def __init__(self):
# 	#load device/context/listener
# 		self.device = alc.alcOpenDevice(None)
# 		self.context = alc.alcCreateContext(self.device, None)
# 		alc.alcMakeContextCurrent(self.context)

# #set player position
# 	def _set_position(self,pos):
# 		self._position = pos
# 		x,y,z = map(int, pos)
# 		al.alListener3f(al.AL_POSITION, x, y, z)

# 	def _get_position(self):
# 		return self._position

# #delete current listener
# 	def delete(self):
# 		alc.alcDestroyContext(self.context)
# 		alc.alcCloseDevice(self.device)

# 	position = property(_get_position, _set_position,doc="""get/set position""")




# #load sound buffers into an openal source player to play them
# class Player(object):
# #load default settings
# 	def __init__(self):
# 	#load source player
# 		self.source = al.ALuint(0)
# 		al.alGenSources(1, self.source)
# 	#disable rolloff factor by default
# 		al.alSourcef(self.source, al.AL_ROLLOFF_FACTOR, 0)
# 	#disable source relative by default
# 		al.alSourcei(self.source, al.AL_SOURCE_RELATIVE,0)
# 	#capture player state buffer
# 		self.state = al.ALint(0)
# 	#set internal variable tracking
# 		self._volume = 1.0
# 		self._pitch = 1.0
# 		self._position = [0,0,0]
# 		self._rolloff = 1.0
# 		self._loop = False
# 		self.queue = []


# #set rolloff factor, determines volume based on distance from listener
# 	def _set_rolloff(self,value):
# 		self._rolloff = value
# 		al.alSourcef(self.source, al.AL_ROLLOFF_FACTOR, value)

# 	def _get_rolloff(self):
# 		return self._rolloff


# #set whether looping or not - true/false 1/0
# 	def _set_loop(self,lo):
# 		self._loop = lo
# 		al.alSourcei(self.source, al.AL_LOOPING, lo)

# 	def _get_loop(self):
# 		return self._loop
	  

# #set player position
# 	def _set_position(self,pos):
# 		self._position = pos
# 		x,y,z = map(int, pos)
# 		al.alSource3f(self.source, al.AL_POSITION, x, y, z)

# 	def _get_position(self):
# 		return self._position
		

# #set pitch - 1.5-0.5 float range only
# 	def _set_pitch(self,pit):
# 		self._pitch = pit
# 		al.alSourcef(self.source, al.AL_PITCH, pit)

# 	def _get_pitch(self):
# 		return self._pitch

# #set volume - 1.0 float range only
# 	def _set_volume(self,vol):
# 		self._volume = vol
# 		al.alSourcef(self.source, al.AL_GAIN, vol)

# 	def _get_volume(self):
# 		return self._volume

# #queue a sound buffer
# 	def add(self,sound):
# 		al.alSourceQueueBuffers(self.source, 1, sound.buf) #self.buf
# 		self.queue.append(sound)

# #remove a sound from the queue (detach & unqueue to properly remove)
# 	def remove(self):
# 		if len(self.queue) > 0:
# 			al.alSourceUnqueueBuffers(self.source, 1, self.queue[0].buf) #self.buf
# 			al.alSourcei(self.source, al.AL_BUFFER, 0)
# 			self.queue.pop(0)

# #play sound source
# 	def play(self):
# 		al.alSourcePlay(self.source)

# #get current playing state
# 	def playing(self):
# 		al.alGetSourcei(self.source, al.AL_SOURCE_STATE, self.state)
# 		if self.state.value == al.AL_PLAYING:
# 			return True
# 		else:
# 			return False

# #stop playing sound
# 	def stop(self):
# 		al.alSourceStop(self.source)

# #rewind player
# 	def rewind(self):
# 		al.alSourceRewind(self.source)

# #pause player
# 	def pause(self):
# 		al.alSourcePause(self.source)

# #delete sound source
# 	def delete(self):
# 		al.alDeleteSources(1, self.source)

# #Go straight to a set point in the sound file
# 	def _set_seek(self,offset):#float 0.0-1.0
# 		al.alSourcei(self.source,al.AL_BYTE_OFFSET,int(self.queue[0].length * offset))

# #returns current buffer length position (IE: 21000), so divide by the buffers self.length
# 	def _get_seek(self):#returns float 0.0-1.0
# 		al.alGetSourcei(self.source, al.AL_BYTE_OFFSET, self.state)
# 		return float(self.state.value)/float(self.queue[0].length)

# 	rolloff = property(_get_rolloff, _set_rolloff,doc="""get/set rolloff factor""")
# 	volume = property(_get_volume, _set_volume,doc="""get/set volume""")
# 	pitch = property(_get_pitch, _set_pitch, doc="""get/set pitch""")
# 	loop = property(_get_loop, _set_loop, doc="""get/set loop state""")
# 	position = property(_get_position, _set_position,doc="""get/set position""")
# 	seek = property(_get_seek, _set_seek, doc="""get/set the current play position""")

Example()
