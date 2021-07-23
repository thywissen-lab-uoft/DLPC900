import pycrafter6500
import numpy
import PIL.Image

# Create the bmp image map
images=[]

# The cat test image is 255 logic, // is floor dividsion to be 0 and 1
#images.append((numpy.asarray(PIL.Image.open("testimage.tif"))//129))
#images.append((numpy.asarray(PIL.Image.open("../test_images/dot_middle_100_pos.tif"))))
#images.append((numpy.asarray(PIL.Image.open("../test_images/dot_array_3x3_75r_pos.tif"))))
#images.append((numpy.asarray(PIL.Image.open("../test_images/Group_test.bmp"))))
images.append((numpy.asarray(PIL.Image.open("../test_images/test.tif"))))

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
repetitions=0       # Number of times to repeat 0 is infinite

dlp.defsequence(images,exposure,trigger_in,dark_time,trigger_out,repetitions)
dlp.startsequence()