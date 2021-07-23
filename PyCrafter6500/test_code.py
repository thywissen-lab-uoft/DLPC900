import pycrafter6500
import numpy
import PIL.Image

images=[]

images.append((numpy.asarray(PIL.Image.open("testimage.tif"))//129))

# Initialize connection to DMD
dlp=pycrafter6500.dmd()

# Stop the pattern sequence
dlp.stopsequence()

# Change to Pattern-on-the-Fly mode
dlp.changemode(3)

# Settings
exposure=[1000000]*30
dark_time=[0]*30
trigger_in=[False]*30
trigger_out=[1]*30

# Define the sequence
dlp.defsequence(images,exposure,trigger_in,dark_time,trigger_out,0)

# Start the sequence
dlp.startsequence()
