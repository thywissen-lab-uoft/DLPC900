import pycrafter6500
import numpy
import PIL.Image

# Create the bmp image map
images=[]
images.append((numpy.asarray(PIL.Image.open("testimage.tif"))//129))

# Open the USB connection to the DMD
dlp=pycrafter6500.dmd()

# Stop DMD
dlp.stopsequence()

# Patter on the fly mode
dlp.changemode(3)

exposure=[500*10**3] # Exposure time in us
dark_time=[0]*30    # Dak time in us
trigger_in=[True]   # Whether to use an input trigger
trigger_out=[False] # Whether to enable the output trigger
repititions=1       # Number of times to repeat 0 is infinite

dlp.defsequence(images,exposure,trigger_in,dark_time,trigger_out,repititions)
dlp.startsequence()
