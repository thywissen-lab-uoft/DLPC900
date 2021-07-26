
# coding: utf-8

# # Generating Grayscale Images : testing different techniques

# In[1]:

import numpy as np
import numpy.random as rnd
import heapq as q
import math
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

get_ipython().magic('pylab inline')


# In[2]:

#Standard image shape : DMD dimensions

#Width :
w = 1920 #912
#Height :
h = 1080 #1140


# ## Generating whole-image patterns 

# In[]:
Input = import_imagecolour('input.bmp')
Input = convert_resolution(Input, h, w)

imshow(255. - Input, cmap = 'binary',vmin=0,vmax=255)

# In[3]:

#White image 
img = np.ones((w,h),np.uint32)*0xFFFFFF
img.shape=h,w

#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
pilImage = pilImage.rotate(180) #rotate 180
pilImage.save('white.bmp')


# In[ ]:

#Single spatial frequency (rows)

if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')
    
period = 2
frequency = 1/(period+0.000000001)
#In pixel^-1 (min = 1/1140 ; max = 1)

img = np.zeros((w,h),np.uint32)
img.shape=h,w

for i in range(1140):
    if(math.cos(2*math.pi*frequency*(i))>0):
        img[i,0:912]=0xFFFFFF
            
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
pilImage = pilImage.rotate(180) #rotate 180
pilImage.save('single frequency = 1:'+ str(period) +'.bmp')


# In[ ]:

#Single spatial frequency (columns)

if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')

period = 25
frequency = 1/(period+0.000001)
#In pixel^-1 (min = 1/912 ; max = 1)

img = np.zeros((w,h),np.uint32)
img.shape=h,w

for j in range(912):
    if(math.cos(2*math.pi*frequency*(j))>0):
        img[0:1140,j]=0xFFFFFF
            
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
pilImage.save('single frequency vertical = 1:'+ str(period) +'.bmp')


# In[5]:

#Width :
w = 1920 #912
#Height :
h = 1080 #1140


size = 5


# #### Random pattern :

# In[6]:

#p = white pixel density
#p=0.5

if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')

img = np.zeros((w,h),np.uint32)
img.shape=h,w

for i in range(h):
    for j in range(w):
    
        if(rnd.rand()<p):
            img[i:i+1,j:j+1]=0xFFFFFF
            
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage.save('random p='+ str(p) +'.bmp')


# #### Random with constraint ( fixed number of white pixels in every square of given size ) :

# In[ ]:

#p : density of white pixels
p=1

if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')

#size = 2
n = int(round(p*size**2))

#del img
img = np.zeros((w,h),np.uint32)
img.shape=h,w

for i in range(h//size):
    for j in range(w//size):
               
        c = 0
        
        if(n<size**2/2.):
            
            while c<n:
                k = int(rnd.rand() * size**2)
                
                if(img[i*size + k/size, j*size + k%size] == 0):
                    img[i*size + k/size, j*size + k%size] = 0xFFFFFF
                    c+=1
                    
        else:
            
            for l in range(size**2):
                img[i*size + l/size, j*size + l%size] = 0xFFFFFF
                
            while c<size**2-n:
                k = int(rnd.rand() * size ** 2)
                
                if(img[i*size + k/size, j*size + k%size] == 0xFFFFFF):
                    img[i*size + k/size, j*size + k%size] = 0
                    c+=1
                    
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
pilImage.save('randconstraint p='+ str(p) +'_'+ str(size)+'x'+str(size)+'.bmp')


# #### Regular square pattern :

# In[ ]:

#p : density of white pixels
#p=0.5

#size = 2

if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')

n = int(round(p*size**2))

del img
img = np.zeros((w,h),np.uint32)
img.shape=h,w

for i in range(h//size):
    for j in range(w//size):
        for k in range(n):
            img[i*size + k/size, j*size + k%size] = 0xFFFFFF
            
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage.save('pattern p='+ str(p) + '-'+ str(size)+'x'+str(size)+'.bmp')


# ####  Floyd Steinberg error diffusion :

# In[ ]:

#p : density of white pixels
#p=0.5

if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')

#The error diffusion will be realized on temp array
temp = np.ones((h,w))*p

for i in range(h):
    for j in range(w):
            
        Old_pix = temp[i,j]
        New_pix = int(round(Old_pix)) 
            
        temp[i,j] = New_pix
            
        Error = Old_pix - New_pix

        if(j<w-1): 
            temp[i,j+1] += 7/16. * Error
                
            if(i<h-1):
                temp[i+1,j+1] += 1/16. * Error
            
        if(i<h-1):
            temp[i+1,j] += 5/16. * Error
                
            if(j>0):
                temp[i+1,j-1] += 3/16. * Error
                
                
#The result is transcripted in an image matrix with hexadecimal RGB format
#del img
img = np.zeros((w,h),np.uint32)
img.shape=h,w

for i in range(h):
    for j in range(w):
        if(temp[i,j]==1):
            img[i,j] = 0xFFFFFF;

                
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage.save('FloydStein p='+ str(p) +'.bmp')


# ####  Improving error diffusion :

# In[ ]:

#Managing edges

#p : density of white pixels
#p=0.33

if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')
    
#The error diffusion will be realized on temp array
temp = np.ones((h,w))*p

for i in range(h):
    for j in range(w):
            
        Old_pix = temp[i,j]
        New_pix = int(round(Old_pix)) 
            
        temp[i,j] = New_pix
            
        Error = Old_pix - New_pix
        
        factor = 0.

        if(j<w-1): 
            factor += 7/16.
                
            if(i<h-1):
                factor += 1/16.
            
        if(i<h-1):
            factor += 5/16.
                
            if(j>0):
                factor += 3/16.
        
        if(factor == 0):
            factor = 1.
        
        
        if(j<w-1): 
            temp[i,j+1] += 7/16. * Error / factor
                
            if(i<h-1):
                temp[i+1,j+1] += 1/16. * Error / factor
            
        if(i<h-1):
            temp[i+1,j] += 5/16. * Error / factor 
                
            if(j>0):
                temp[i+1,j-1] += 3/16. * Error / factor
                
                
#The result is transcripted in an image matrix with hexadecimal RGB format
#del img
img = np.zeros((w,h),np.uint32)
img.shape=h,w

for i in range(h):
    for j in range(w):
        if(temp[i,j]==1):
            img[i,j] = 0xFFFFFF;

                
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage.save('FloydStein TESTS p='+ str(p) +'.bmp')

#ARTIFACTS Still present


# In[ ]:

#2 line error diffusion : Stucki filter

if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')

#p : density of white pixels
#p=0.33

#The error diffusion will be realized on temp array
temp = np.ones((h,w))*p

for i in range(h):
    for j in range(w):
            
        Old_pix = temp[i,j]
        New_pix = int(round(Old_pix)) 
            
        temp[i,j] = New_pix
            
        Error = Old_pix - New_pix

        if(j<w-1): 
            temp[i,j+1] += 8/42. * Error
                
            if(i<h-1):
                temp[i+1,j+1] += 4/42. * Error
                
            if(i<h-2):
                temp[i+2,j+1] += 2/42. * Error
                
        if(j<w-2):
            temp[i,j+2] += 4/42. * Error
            
            if(i<h-1):
                temp[i+1,j+2] += 2/42. * Error
                
            if(i<h-2):
                temp[i+2,j+2] += 1/42. * Error
                
                
        if(i<h-1):
            temp[i+1,j] += 8/42. * Error
                
        if(i<h-2):
            temp[i+2,j] += 4/42. * Error
            
        
        if(j>0):
            
            if(i<h-1):
                temp[i+1,j-1] += 4/42. * Error
                
            if(i<h-2):
                temp[i+2,j-1] += 2/42. * Error
                
        if(j>1):
            
            if(i<h-1):
                temp[i+1,j-2] += 2/42. * Error
                
            if(i<h-2):
                temp[i+2,j-2] += 1/42. * Error
                

#The result is transcripted in an image matrix with hexadecimal RGB format
#del img
img = np.zeros((w,h),np.uint32)
img.shape=h,w

for i in range(h):
    for j in range(w):
        if(temp[i,j]==1):
            img[i,j] = 0xFFFFFF;

                
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage.save('Stucki tests  p='+ str(p) +'.bmp')

#Artifacts remain ..



# In[ ]:

#Tests on input image :
Input_image = Image.open("inputs/input1.jpg")
Input = np.array(Input_image)

h,w,k = Input.shape
print('Dimensions: h=' + str(h) +'  x  w=' + str(w))

#Taking a black and white image out of the coloured image
Input_BW = Input[:,:,1]


temp = Input_BW.copy() /255.


for i in range(h):
    for j in range(w):
            
        Old_pix = temp[i,j]
        New_pix = int(round(Old_pix)) 
            
        temp[i,j] = New_pix
            
        Error = Old_pix - New_pix

        if(j<w-1): 
            temp[i,j+1] += 8/42. * Error
                
            if(i<h-1):
                temp[i+1,j+1] += 4/42. * Error
                
            if(i<h-2):
                temp[i+2,j+1] += 2/42. * Error
                
        if(j<w-2):
            temp[i,j+2] += 4/42. * Error
            
            if(i<h-1):
                temp[i+1,j+2] += 2/42. * Error
                
            if(i<h-2):
                temp[i+2,j+2] += 1/42. * Error
                
                
        if(i<h-1):
            temp[i+1,j] += 8/42. * Error
                
        if(i<h-2):
            temp[i+2,j] += 4/42. * Error
            
        
        if(j>0):
            
            if(i<h-1):
                temp[i+1,j-1] += 4/42. * Error
                
            if(i<h-2):
                temp[i+2,j-1] += 2/42. * Error
                
        if(j>1):
            
            if(i<h-1):
                temp[i+1,j-2] += 2/42. * Error
                
            if(i<h-2):
                temp[i+2,j-2] += 1/42. * Error

                
#The result is transcripted in an image matrix with hexadecimal RGB format
#del img
img = np.zeros((w,h),np.uint32)
img.shape=h,w

for i in range(h):
    for j in range(w):
        if(temp[i,j]==1):
            img[i,j] = 0xFFFFFF;

                
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage.save('Stucki test output.bmp')

#Conclusion : Works better on input image ! 
#Efficiently removes bundles of white pixel in Vijin's hair


# Stucki and edge managing Floyd Steinberg do not remove artifacts appearing in fixed density images <br>
# However, Stucki works better on a real image

# ## Grayscaling an input image

# #### Defining the methods to process input image: 

# In[7]:

def randomize(Input_BW):
    Output = np.zeros((w,h),np.uint32)
    Output.shape=h,w
    
    for i in range(h):
        for j in range(w):
            
            if(rnd.rand()*255 < Input_BW[i,j]):
                Output[i,j] = 0xFFFFFF
    
    #Saving the image
    pilImage = Image.frombuffer('RGBA',(w,h),Output,'raw','RGBA',0,1)
    pilImage.save('randomized_output.bmp')
    
    
    
def randconstraint(Input_BW, size):
    Output = np.zeros((w,h),np.uint32)
    Output.shape=h,w
    
    for i in range(h//size):
        for j in range(w//size):
            
            #weight / 255 / size**2 = local density of white pixels
            weight = 0
            for k in range(size**2):
                weight += Input_BW[i*size + k/size, j*size + k%size]
            n = int(round(weight/255.))
            
            c = 0
            if(n<size**2/2.):
                while c<n:
                    k = int(rnd.rand()*size**2)

                    if(Output[i*size + k/size, j*size + k%size] == 0): 
                        Output[i*size + k/size, j*size + k%size] = 0xFFFFFF
                        c+=1

            else:
                for l in range(size**2):
                    Output[i*size + l/size, j*size + l%size] = 0xFFFFFF

                while c<size**2-n:
                    k = int(rnd.rand()*size**2)

                    if(Output[i*size + k/size, j*size + k%size] == 0xFFFFFF): 
                        Output[i*size + k/size, j*size + k%size] = 0
                        c+=1
                        
                
    #Saving the image
    pilImage = Image.frombuffer('RGBA',(w,h),Output,'raw','RGBA',0,1)
    pilImage.save('randconstrained_output_'+ str(size)+'x'+str(size)+'.bmp')

def patternize(Input_BW, size):
    Output = np.zeros((w,h),np.uint32)
    Output.shape=h,w
    
    for i in range(h//size):
        for j in range(w//size):
            
            #weight / 255 = local density of white pixels
            weight = 0
            for k in range(size**2):
                weight += Input_BW[i*size + k/size, j*size + k%size]
            
            n = int(round(weight/255.))
            
            for k in range(n):
                Output[i*size + k/size, j*size + k%size] = 0xFFFFFF
                
    #Saving the image
    pilImage = Image.frombuffer('RGBA',(w,h),Output,'raw','RGBA',0,1)
    pilImage.save('patternized_output_'+ str(size)+'x'+str(size)+'.bmp')
    
def FloydSteinberg(Input_BW):
    
    #The error diffusion will be realized on a copy of the input
    Input_cp = Input_BW.copy()
    
    for i in range(h):
        for j in range(w):

            Old_pix = Input_cp[i,j]
            New_pix = int(round(Old_pix/255.)) * 255.

            Input_cp[i,j] = New_pix

            Error = Old_pix - New_pix

            if(j<w-1): 
                Input_cp[i,j+1] += 7/16. * Error

                if(i<h-1):
                    Input_cp[i+1,j+1] += 1/16. * Error

            if(i<h-1):
                Input_cp[i+1,j] += 5/16. * Error

                if(j>0):
                    Input_cp[i+1,j-1] += 3/16. * Error

                    
    #The result is transcripted in an image matrix with hexadecimal RGB format
    Output = np.zeros((w,h),np.uint32)
    Output.shape=h,w

    for i in range(h):
        for j in range(w):
            if(Input_cp[i,j] == 255):
                Output[i,j] = 0xFFFFFF


    #Saving the image
    pilImage = Image.frombuffer('RGBA',(w,h),Output,'raw','RGBA',0,1)
    pilImage.save('FloydSteinberged.bmp')


    
    
def Stucki(Input_BW, m, max_val):

    #The error diffusion will be realized on a copy of the input
    Input_cp = Input_BW.copy()

    for i in range(h):
        for j in range(w):

            Old_pix = Input_cp[i,j]
            New_pix = int(round(Old_pix/255.)) * 255. 

            Input_cp[i,j] = New_pix

            Error = Old_pix - New_pix

            if(j<w-1): 
                Input_cp[i,j+1] += 8/42. * Error

                if(i<h-1):
                    Input_cp[i+1,j+1] += 4/42. * Error

                if(i<h-2):
                    Input_cp[i+2,j+1] += 2/42. * Error

            if(j<w-2):
                Input_cp[i,j+2] += 4/42. * Error

                if(i<h-1):
                    Input_cp[i+1,j+2] += 2/42. * Error

                if(i<h-2):
                    Input_cp[i+2,j+2] += 1/42. * Error


            if(i<h-1):
                Input_cp[i+1,j] += 8/42. * Error

            if(i<h-2):
                Input_cp[i+2,j] += 4/42. * Error


            if(j>0):

                if(i<h-1):
                    Input_cp[i+1,j-1] += 4/42. * Error

                if(i<h-2):
                    Input_cp[i+2,j-1] += 2/42. * Error

            if(j>1):

                if(i<h-1):
                    Input_cp[i+1,j-2] += 2/42. * Error

                if(i<h-2):
                    Input_cp[i+2,j-2] += 1/42. * Error


    #The result is transcripted in an image matrix with hexadecimal RGB format
    #del img
    Output = np.zeros((w,h),np.uint32)
    Output.shape=h,w

    for i in range(h):
        for j in range(w):
            if(Input_cp[i,j]==255.):
                Output[i,j] = 0xFFFFFF;


    #Saving the image
    pilImage = Image.frombuffer('RGBA',(w,h),Output,'raw','RGBA',0,1)
    pilImage.save('m= '+ str(m) + ' max= ' + str(max_val) +' Stucki.bmp')
    
    
def import_imageBW(string):
    
    Input_image = Image.open(string)
    Input_BW = np.array(Input_image)
    
    return Input_BW


def import_imagecolour(string):
    
    Input_image = Image.open(string)
    Input_np = np.array(Input_image)
    Input_BW = Input_np[:,:,0]
    
    return Input_BW

def crop(x1,y1,x2,y2, imagenp):
    
    cropped_imagenp = np.zeros((x2-x1,y2-y1))
    cropped_imagenp[:,:] = imagenp[x1:x2,y1:y2]
    
    return cropped_imagenp

def divide(Input, white_DMDres):
    
    #Dividing by beam profile (avoiding overshoot 255->0)
    temp = Input / white_DMDres * np.amax(white_DMDres)
    for i in range(h):
        for j in range(w):
            if (temp[i,j] > 255) : temp[i,j] = 255

    return temp


#Parameters from calibration : 
popt = [0.70573506, 2.00252746, 0.02135317]

def inverse_calib(Input):
    
    temp = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            #Avoiding close to Zero values
            if(Input[i,j] < 0.0214*255): 
                Inp_ij = 0.0214*255
            else:
                Inp_ij = Input[i,j]
            temp[i,j] = round(math.exp(math.log((Inp_ij/255.-popt[2])/popt[0])/popt[1])*255)
    
    #Avoiding overshoot
    for i in range(h):
        for j in range(w):
            if (temp[i,j] > 255) : temp[i,j] = 255
                
    return temp

def region_interest(Input, m):
        
    #Aspect ratio of DMD is a bit special, we want a square on the DMD
    m_h = m
    m_w = int(m * 912 * 6.16 / 1140 / 9.85)
    
    temp = Input
    if(m == 0): 
        return temp
    else :
        #Taking region of interest into account : 
        temp_square = np.zeros([h,w])
        for i in range((h-m_h)//2,(h+m_h)//2):
            for j in range((w-m_w)//2,(w+m_w)//2):
                temp_square[i,j] = temp[i,j]
        return temp_square
    


# ### Creating output image

# In[8]:

#Size of the region of interest (m=0 selects the whole image):
m = 400
max_val = 110


# In[9]:

#Receiving the input image

Input = import_imagecolour("inputs/input1.jpg")

h,w = Input.shape
print('Dimensions: h=' + str(h) +'  x  w=' + str(w))
if (h!=1140 or w!=912): print('DIMENSIONS DIFFER FROM DMD')



#Putting grayscale values : changing 255 values to lower max_val
for i in range(h): 
    for j in range(w): 
        if(Input[i,j] == 255.): Input[i,j]= max_val

            
            
fig = plt.figure(figsize=(10,10))
imshow(255 - Input , cmap = 'Greys', vmin = 0, vmax = 255)
plt.show()


# In[9]:

Input = import_imagecolour("inputs/input1.jpg")

h,w = Input.shape
#print h, w
imshow(255 - Input, cmap = 'Greys')


# In[10]:

Stucki(Input, 0,0)


# In[8]:

randomize(Input)
randconstraint(Input, 2)
patternize(Input, 2)
FloydSteinberg(Input)


# In[ ]:

#Dividing by beam profile and applying inverse calib function :

white = import_imageBW('CCD/white187.bmp')
white = crop(230, 260, 780, 1160, white)

h_white,w_white = white.shape

fig = plt.figure(figsize=(15,10))
plt.subplot(121)
imshow(255 - white, cmap = 'Greys')
plt.title('White image (CCD resolution)')

white_DMDres = np.zeros([h,w])
for i in range(h):
    for j in range(w):
        white_DMDres[i,j] = white[int(1.*i/h*h_white),int(1.*j/w*w_white)]

plt.subplot(122)
imshow(255. - white_DMDres, cmap = 'Greys')
plt.title('Transcripted to DMD resolution')
plt.show()


corrected_white_DMDres = inverse_calib(white_DMDres)


#Showing corrected image
fig = plt.figure(figsize=(15,10))

plt.subplot(121)
imshow(255. - corrected_white_DMDres, cmap = 'Greys',vmin = 0, vmax = 255)
plt.title('Corrected white image')

plt.tight_layout()


corrected_Input = divide(Input, corrected_white_DMDres)
corrected_Input = region_interest(corrected_Input, m)

#Showing corrected image
plt.subplot(122)
imshow(255. - corrected_Input, cmap = 'Greys',vmin = 0, vmax = 255)
plt.title('Target divided by beam profile')

plt.tight_layout()
plt.show()


if(np.amax(corrected_Input) == 255) :
    print('Saturation of corrected image : If it is not due to a defect, narrowing region of interest or reducing input image intensity might be a good idea')
    


# In[ ]:

#If correction is desired :

Input = corrected_Input


# In[ ]:

#Saving the image with Stucki error diffusion :
Stucki(Input, m, max_val)


# In[ Another ]:



# In[ n]: Generate a single white dot for Airy Disk fitting 
#width and Height for image on DMD
w = 1080
h = 1920
arrayn = 10
arraym = 6
#Width :


img = np.zeros((h,w),np.uint32)*0xFFFFFF
#img[h//4,w//4] = np.uint32(1) * 0xFFFFFF
m = int(1)

for k in range(arraym):
    for l in range(arrayn):
        kk=k+1
        ll = l+1
        for i in range(kk*h//(arraym+1)-m,kk*h//(arraym+1)+m): 
            for j in range(ll*w//(arrayn+1)-m,ll*w//(arrayn+1)+m): 
                img[i,j] = np.uint32(1) * 0xFFFFFF

w = 912
h = 1140

#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
pilImage.save('dot_RM_6x10.bmp')


# In[n+1]: Another way of generating white dots

def convert_resolution(Input, h_final, w_final):
    
    h_init, w_init = Input.shape
    
    Output = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            Output[i,j] = Input[int(1.*i/h_final*h_init),int(1.*j/w_final*w_init)]
    
    return Output

h_target,w_target = 560,920
#Width :

w = 1920
h = 1080
#Input[]: 
img = np.zeros((h,w))*255


for i in range(h): 
    for j in range(w): 
        if math.sqrt((i-h/2)**2 + (j-w/2)**2) <= 200 :
            img[i,j] = 255


         

w = 1920
h = 1080
 
imgcon = convert_resolution(img,h,w) 
imgcon = np.uint32(imgcon//255) * 0xFFFFFF      

pilImage = Image.frombuffer('RGBA',(w,h),imgcon,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
pilImage.save('dot_middle_circle_200.bmp') 

# In[n+1]: Generating black dots

def convert_resolution(Input, h_final, w_final):
    
    h_init, w_init = Input.shape
    
    Output = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            Output[i,j] = Input[int(1.*i/h_final*h_init),int(1.*j/w_final*w_init)]
    
    return Output

h_target,w_target = 560,920
#Width :

w = 1920
h = 1080
#Input[]: 
img = np.ones((h,w))*255


for i in range(h): 
    for j in range(w): 
        if math.sqrt((i-h/2)**2 + (j-w/2)**2) <= 200 :
            img[i,j] = 0


         

w = 1920
h = 1080
 
imgcon = convert_resolution(img,h,w) 
imgcon = np.uint32(imgcon//255) * 0xFFFFFF  

pilImage = Image.frombuffer('RGBA',(w,h),imgcon,'raw','RGBA',0,1)
#pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
pilImage.save('black_middle_200.bmp') 

     
        
        
# In[ n]: Generate dot array 
#width and Height for image on DMD
w = 1920 #912
#Height :
h = 1080 #1140
#Width :


img = np.ones((h,w),np.uint32)*0xFFFFFF
n = 75 #radius
m = 500

for i in range(h): 
    for j in range(w): 
#        if (i-h//4*0)**2+(j-w//4*0)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
#        if (i-h//4*0)**2+(j-w//4*1)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
#        if (i-h//4*0)**2+(j-w//4*2)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
#        if (i-h//4*0)**2+(j-w//4*3)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
#        if (i-h//4*0)**2+(j-w//4*4)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF

#        if (i-h//4*1)**2+(j-w//4*0)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
        if (i-h//4*1)**2+(j-w//4*1)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
        if (i-h//4*1)**2+(j-w//4*2)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
        if (i-h//4*1)**2+(j-w//4*3)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
#        if (i-h//4*1)**2+(j-w//4*4)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF

#        if (i-h//4*2)**2+(j-w//4*0)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
        if (i-h//4*2)**2+(j-w//4*1)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
        if (i-h//4*2)**2+(j-w//4*2)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
        if (i-h//4*2)**2+(j-w//4*3)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
#        if (i-h//4*2)**2+(j-w//4*4)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF

#        if (i-h//4*3)**2+(j-w//4*0)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
        if (i-h//4*3)**2+(j-w//4*1)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
        if (i-h//4*3)**2+(j-w//4*2)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
        if (i-h//4*3)**2+(j-w//4*3)**2 <= n**2:
            img[i,j] = np.uint32(0) * 0xFFFFFF
#        if (i-h//4*3)**2+(j-w//4*4)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF

#        if (i-h//4*4)**2+(j-w//4*0)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
#        if (i-h//4*4)**2+(j-w//4*1)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
#        if (i-h//4*4)**2+(j-w//4*2)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
#        if (i-h//4*4)**2+(j-w//4*3)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
#        if (i-h//4*4)**2+(j-w//4*4)**2 <= n**2:
#            img[i,j] = np.uint32(1) * 0xFFFFFF
            



#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
#pilImage.save('Dot_middle_60x60.bmp')
pilImage.save('dot_array_3x3_75r_neg.bmp')
       

        

 # In[ n]: Generate grid lines 
#width and Height for image on DMD
w = 1920 #912
#Height :
h = 1080 #1140
#Width :
w, h = (1920, 1080)
hh = np.linspace(0, 1, h)
ww = np.linspace(0, 1, w)
hv, wv = np.meshgrid(hh, ww)

img = np.zeros((h,w),np.uint32)*0xFFFFFF
n = 10


for i in range(h): 
    for j in range(w): 
        if abs(i-h//4*0)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(i-h//4*1)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(i-h//4*2)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(i-h//4*3)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(i-h//4*4)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*0)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*1)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*2)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*3)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*4)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*5)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF



#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
#pilImage.save('Dot_middle_60x60.bmp')
pilImage.save('grid.bmp')
       

# In[ n]: Generate 1D sine wave
#width and Height for image on DMD
w = 1920 #912
#Height :
h = 1080 #1140
#Width :
w, h = (1920, 1080)
hh = np.linspace(0, 1, h)
ww = np.linspace(0, 1, w)
hv, wv = np.meshgrid(hh, ww)

img = np.zeros((h,w),np.uint32)*0xFFFFFF
n = 20
m = 500

for i in range(h): 
    for j in range(w): 
        if abs(i-h//4*0)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(i-h//4*1)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(i-h//4*2)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(i-h//4*3)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(i-h//4*4)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*0)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*1)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*2)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*3)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*4)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
        if abs(j-w//5*5)<= n: 
            img[i,j] = np.uint32(1) * 0xFFFFFF



#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
#pilImage.save('Dot_middle_60x60.bmp')
pilImage.save('sine.bmp')
        
# In[n+1]: Generating black box        
        
 #width and Height for image on DMD
w = 1920 #912
#Height :
h = 1080 #1140
#Width :
w, h = (1920, 1080)
hh = np.linspace(0, 1, h)
ww = np.linspace(0, 1, w)
hv, wv = np.meshgrid(hh, ww)

img = np.zeros((h,w),np.uint32)*0xFFFFFF
boxsize = 300;

for i in range(h//2-boxsize//2,h//2-boxsize//2): 
    for j in range(w//2-boxsize//2,w//2-boxsize//2):
            img[i,j] = np.uint32(1) * 0xFFFFFF        

#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
#pilImage.save('Dot_middle_60x60.bmp')
pilImage.save('box.bmp')

# In[]: boxes from illustrator!

w = 1920 #912
#Height :
h = 1080 #1140
#Width :

imageinput = import_imagecolour("Box-01.tif")
imageinput = convert_resolution(imageinput, h, w)//255
print(imageinput)
#imageinput = int(imageinput)
#imageinput
#imageinput = abs(imageinput - np.ones((h,w),np.uint32))
imshow(imageinput)


w, h = (1920, 1080)
hh = np.linspace(0, 1, h)
ww = np.linspace(0, 1, w)
hv, wv = np.meshgrid(hh, ww)

img = np.zeros((h,w),np.uint32)*0xFFFFFF

for i in range(h): 
    for j in range(w): 
        if int(imageinput[i,j]) == 1: 
            img[i,j] = np.uint32(1) * 0xFFFFFF
       

#img = imageinput*np.uint32(1)*0xFFFFFF
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
#pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
#pilImage.save('Dot_middle_60x60.bmp')
pilImage.save('Box_300.bmp')


# In[]: traget for boxes with a little light in the middle

w = 1920 #912
#Height :
h = 1080 #1140
#Width :

imageinput = import_imagecolour("Box-01.tif")
imageinput = convert_resolution(imageinput, h, w)//255
print(imageinput)


imshow(imageinput)


w, h = (1920, 1080)
hh = np.linspace(0, 1, h)
ww = np.linspace(0, 1, w)
hv, wv = np.meshgrid(hh, ww)

img = np.zeros((h,w))

for i in range(h): 
    for j in range(w): 
        if int(imageinput[i,j]) == 1: 
            img[i,j] = 0.1
       
# In[]: traget for boxes with a little gaussian in the middle
            
w = 1920 #912
#Height :
h = 1080 #1140
#Width :


box = import_imagecolour("Box-01.tif")
box = convert_resolution(box, h, w)//255
x, y = np.meshgrid(np.linspace(0,w,w), np.linspace(0,h,h))
Gau = twoD_Gaussian((x,y), 1, w//2, h//2, 1000, 1000, 0, 0)
Gau = Gau.reshape(-1,w)
print(box)
#imshow(box)
img = np.zeros((h,w))

for i in range(h): 
    for j in range(w): 
        if int(box[i,j]) == 1: 
            Gau[i,j] = 0

img = box+Gau
plt.imshow(img)
plt.colorbar()

# In[]: Ring feature for density fitting

w = 1920 #912
#Height :
h = 1080 #1140
#Width :

w, h = (1920, 1080)
hh = np.linspace(0, 1, h)
ww = np.linspace(0, 1, w)
hv, wv = np.meshgrid(hh, ww)

img = np.ones((h,w))
inner = 150
outer = 300

for i in range(h): 
    for j in range(w): 
        if ((i-h//2)**2+(j-w//2)**2)<=outer**2 and ((i-h//2)**2+(j-w//2)**2)>=inner**2: 
            img[i,j] = 0
        if ((i-h//2)**2+(j-w//2)**2)<=inner**2:
            img[i,j] = 0.5
plt.imshow(img)
plt.colorbar()       