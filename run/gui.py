import tkinter as tk
import subprocess
from start import *

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self, text="Welcome to the Video to Audio App")
        self.label.pack(side="top")

        self.start_object_recog = tk.Button(self)
        self.start_object_recog["text"] = "Start Object Recognition"
        self.start_object_recog["command"] = self.start_darknet
        self.start_object_recog.pack(side="right")

        self.start_live_captioning = tk.Button(self)
        self.start_live_captioning["text"] = "Start Live Captioning"
        self.start_live_captioning["command"] = self.start_neuraltalk2
        self.start_live_captioning.pack(side="left")
        
        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.pack(side="bottom")

    def start_darknet(self):
        print("Object Recognition Started. Press ESC to stop recording.")
        start_camera()

    def start_neuraltalk2(self):
        print("Live Captioning Started. Press ESC to stop recording.")
        subprocess.call("cd ~/UBC/CPEN391/project2/neuraltalk2; th videocaptioning.lua", shell=True)

root = tk.Tk()
root.title("Audio to Video App")
app = Application(master=root)
app.mainloop()
