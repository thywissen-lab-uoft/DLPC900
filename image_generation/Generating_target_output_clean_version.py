
# coding: utf-8

# #  Pattern generator to hit target in CCD output

# In[1]:

import numpy as np
import numpy.random as rnd
import heapq as q
import math
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

get_ipython().magic('pylab inline')




# In[2]:

#DMD dimensions

#Width :
w = 1920
#Height :
h = 1080

def twoD_Gaussian(X, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y = X
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def line_tool(x1,y1,x2,y2, imagenp):    
    a = ((x1,y1),(x2,y2))    
    if(abs(x2-x1)>abs(y2-y1)):
        x = np.sign(x2-x1)
        y = 0        
    else:
        x = 0
        y = np.sign(y2-y1)    
    n = max(abs(x2-x1),abs(y2-y1))    
    ret = np.zeros(n)    
    for k in range(n):
        ret[k] = imagenp[a[0][0] + x*k + abs(y)*int(round(1.*k*(a[1][0]-a[0][0])/n))   ,   a[0][1] + y*k + abs(x)*int(round(1.*k*(a[1][1]-a[0][1])/n)) ]        
    return ret


#Methods used to grayscale/correct pattern

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
    
    for i in range(h/size):
        for j in range(w/size):
            
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
    
    for i in range(h/size):
        for j in range(w/size):
            
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


    
    
def Stucki(Input_BW, name, invert=False):

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
    Output = np.zeros((h,w),np.uint32)
    #Output.shape=h,w
    
    for i in range(h):
        for j in range(w):
            if(Input_cp[i,j]==255.):
                Output[i,j] =  0xFFFFFF
            else: 
                Output[i,j] = 0
                    
    print(Output.dtype)
    #Saving the image
    pilImage = Image.frombuffer('RGBA',(w,h),Output,'raw','RGBA',0,1)
    pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
    #pilImage = pilImage.rotate(180) #rotate 180
    pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
    if invert == True:       
        im = pilImage.convert('L')
        im_invert = ImageOps.invert(im)
        im_invert = im_invert.convert('1')
        im_invert.save(name + '_neg' + '.bmp')
    else:
        pilImage.save(name+'.bmp')
    return Input_cp          


    
    
    
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
    temp = Input / white_DMDres * 255.
    for i in range(h):
        for j in range(w):
            if (temp[i,j] > 255) : temp[i,j] = 255

    return temp


#Parameters from calibration : 
popt = [0.70573506, 2.00252746, 0.02135317]

def calib(x):
    return popt[0]*x**popt[1] +popt[2]

def inverse_calib(Input):
    
    temp = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            #Avoiding values smaller than calib function offset (To avoid problems with the log)
            if(Input[i,j] < 0.0214*255): 
                #We put the target value to zero to have real black and avoid white pixels appearing in black regions
                temp[i,j] = 0  
            else:
                temp[i,j] = math.exp(math.log((Input[i,j]/255.-popt[2])/popt[0])/popt[1])*255
    
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
        for i in range((h-m_h)/2,(h+m_h)/2):
            for j in range((w-m_w)/2,(w+m_w)/2):
                temp_square[i,j] = temp[i,j]
        return temp_square


def convert_resolution(Input, h_final, w_final):
    
    h_init, w_init = Input.shape
    
    Output = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            Output[i,j] = Input[int(1.*i/h_final*h_init),int(1.*j/w_final*w_init)]
    
    return Output
    





# In[]:
#We define the substract_offset method to avoid O values in the denominator of the fraction in the DMDnonlin/CCDoffset method
#However, we want a real zero for the target to avoid white pixels in large black areas
#cond = 1 => low values approximated to 1/255
#cond = 0 => low values approximated to 0
def substract_offset(Input, offset, cond):
    
    h,w = Input.shape
    Output = Input - np.ones((h,w))*offset
    
    for i in range(h):
        for j in range(w):
            if(Output[i,j]< 1/255.):
                if(cond==1): Output[i,j] = 1/255.
                elif(cond==0): Output[i,j] = 0
    
    return Output

#Reminder :  
popt = [0.97538718, 2, 0.02836889]

def Create_DMD_pattern_DMDnonlin_CCDoffset(Input,m):
    
    offset = 0#255*popt[2] 
    
     #Beam profile information
#    whitein = import_imageBW('white_0.5V.tif')
#    white = crop(314, 163, 911, 1213,whitein)
    white = 255*np.ones((h,w),dtype=uint8)
    
    white_DMDres = convert_resolution(white, h, w)
    corrected_white_DMDres = substract_offset(white_DMDres, offset, 1)
    
    corrected_Input = substract_offset(Input, offset, 0)
    corrected_Input = divide(corrected_Input, corrected_white_DMDres)
    corrected_Input = (corrected_Input/255.)**(1./popt[1]) * 255
    
    if(np.amax(corrected_Input) == 255) :
        print('Saturation of corrected image : If it is not due to a defect, narrowing region of interest or reducing input image intensity might be a good idea')

    return corrected_Input

#def Create_DMD_pattern_DMDnonlin_CCDoffset_renormalized(Input,m):
#    
#    offset = 255*popt[2] 
#    
#     #Beam profile information
#    white = import_imageBW('white_1.2V.tif')
#    white = crop(314, 163, 911, 1213,white)
#    
#    white_DMDres = convert_resolution(white, h, w)
#    corrected_white_DMDres = substract_offset(white_DMDres, offset, 1)
#    white
#    corrected_Input = popt[0] * Input #No substraction of the CCD offset, but a renormalization to have whole-DMD saturation at target = 1
#    #corrected_Input = Input
#    corrected_Input = divide(corrected_Input, corrected_white_DMDres)
#    corrected_Input = (corrected_Input/255.)**(1./popt[1]) * 255
#    
#    if(np.amax(corrected_Input) == 255) :
#        print('Saturation of corrected image : If it is not due to a defect, narrowing region of interest or reducing input image intensity might be a good idea')
#
#    return corrected_Input


# In[8]:

#Target definition (create your target here)

#
#m = 0
#factor = 0.25 #Target will be multiplied by factor < 1 in order to avoid DMD saturation
#
#
##Creating target profile
#Input = np.zeros((h,w))
#for i in range(h): 
#    for j in range(w//2): 
#        Input[i,j]= 255
#        #Input[i,j]= round(255.*j/w)
#        
#        
#        
##Multiplying target by factor
#Input = Input * factor
#            
#            
#fig = plt.figure(figsize=(10,10))
#imshow(255 - Input , cmap = 'Greys', vmin = 0, vmax = 255)
#plt.title('Target profile')
#plt.show()


# In[4]: INPUT TARGET

#Importing target (if target is saved as a picture)
h,w = 1080,1920
m = 0
factor = 0.75

#Input = import_imageBW('input.bmp')
Input = import_imagecolour('black_middle_200.bmp')
#Input = np.ones((h,w))*255
Input = Input*factor

##Gaussian#############################################
#x, y = np.meshgrid(np.linspace(0,w,w), np.linspace(0,h,h))
#Input = twoD_Gaussian((x,y), 255, w//2, h//2, 150, 150, 0, 0)
#Input = Input.reshape(-1,w)
#Input = 255-Input
#################################################

###Box with little light in the middle##############
#imageinput = import_imagecolour("Box-01.tif")
#imageinput = convert_resolution(imageinput, h, w)//255
#hh = np.linspace(0, 1, h)
#ww = np.linspace(0, 1, w)
#hv, wv = np.meshgrid(hh, ww)
#Input = np.ones((h,w))
#for i in range(h): 
#    for j in range(w): 
#        if int(imageinput[i,j]) == 0: 
#            Input[i,j] = factor
#Input = Input*255
####################################################

###Ring with little light in the middle#############
#img = np.ones((h,w))
#inner = 75
#outer = 200
#for i in range(h): 
#    for j in range(w): 
#        if ((i-h//2)**2+(j-w//2)**2)<=outer**2 and ((i-h//2)**2+(j-w//2)**2)>=inner**2: 
#            img[i,j] = 0
#        if ((i-h//2)**2+(j-w//2)**2)<=inner**2:
#            img[i,j] = 0.0
#Input = img*255
####################################################

#Input = np.loadtxt('Matrix_Out.txt')*255

whitein = import_imageBW('white_0.5V.tif')
white = crop(314, 163, 911, 1213, whitein)

h_import, w_import = Input.shape
print('Dimensions of imported target :', h_import, 'x', w_import)


#Transcripting to DMD resolution
if(h_import != h or w_import != w):
    Input = convert_resolution(Input, h, w)
    print('Converted to DMD resolution : ', h,'x', w)
    
    
#Multiplying target by factor
#Input = Input * factor
    
fig = plt.figure(figsize=(10,10))
imshow(255 - Input , cmap = 'Greys', vmin = 0, vmax = 255)
plt.title('Target profile')
plt.show()


# In[9]: OUTPUT 

#Creating DMD pattern (using non-linear DMD hypothesis with a CCD offset, confirmed by experiments - renormalized)


Output = Create_DMD_pattern_DMDnonlin_CCDoffset(Input, m)
#Output = Create_DMD_pattern_DMDnonlin_CCDoffset_renormalized(Input, m)
 

fig = plt.figure(figsize=(10,10))
imshow(255 - Output , cmap = 'Greys', vmin = 0, vmax = 255)
plt.title('Computed DMD pattern (DMD non lin, CCD offset)')
plt.show()





# In[]: Test Output
imagenp = Output
x1, y1 = 600, 0
x2, y2 = 600, 1700

ret = line_tool(x1, y1,x2,y2, imagenp) / 255. 
x = range(ret.size)


fig = plt.figure(figsize=(10,5))

#Plotting the profile
plt.subplot(121)
plt.plot(x,ret)
plt.ylim(0,1)
plt.title('Intensity profile')

#Showing image with profile line
plt.subplot(122)
imshow(255. - imagenp, cmap = 'binary',vmin=0,vmax=255)
plt.plot([y1,y2],[x1,x2], "r")
plt.plot([y1],[x1],"ro")
plt.plot([y2],[x2],"bo")
plt.title('Image')

plt.tight_layout()
plt.show()



# In[]:

imagenp = white
x1, y1 = 200, 0
x2, y2 = 200,800 

ret = line_tool(x1, y1,x2,y2, imagenp) / 255. 
x = range(ret.size)


fig = plt.figure(figsize=(10,5))

#Plotting the profile
plt.subplot(121)
plt.plot(x,ret)
plt.ylim(0,1)
plt.title('Intensity profile')

#Showing image with profile line
plt.subplot(122)
imshow(255. - imagenp, cmap = 'binary',vmin=0,vmax=255)
plt.plot([y1,y2],[x1,x2], "r")
plt.plot([y1],[x1],"ro")
plt.plot([y2],[x2],"bo")
plt.title('Image')

plt.tight_layout()
plt.show()

# In[6]:

if (m == 0):
    bh = 0
    bw = 0
else :
    bh = (h-m)/2
    bw = (h-m)/2

plt.plot(range(1920), Output[bh + 10,:], 'b')
plt.plot(range(1920), Output[h - bh - 10,:], 'g')
plt.plot(range(1080), Output[:,bw + 10], 'r')
plt.plot(range(1080), Output[:,w - bw - 10], 'c')

ax = np.linspace(0 ,1080, 1080)
a_255 = np.ones(1080)*255
plt.plot(ax, a_255, 'k--')

plt.xlim(0,1080)
plt.ylim(0,300)

plt.title('Checking contours for saturation')
plt.xlabel('Pixel index')
plt.ylabel('Intensity, max = 255')
plt.show()


# In[10]: OUTPUT TO DMD

#Saving the image with Stucki error diffusion :


#name = 'Disk_400_Para_middle_0.5'
name = 'Flat_hole_r_200_contrast_50'
Sresult = Stucki(Output, name, invert=True)
#FloydSteinberg(Output)
imshow(255. - Sresult, cmap = 'binary',vmin=0,vmax=255)


# In[11]: Try iteration



#Input = Stucki(Output,name)
#Output = Create_DMD_pattern_DMDnonlin_CCDoffset(Input, m)
#white = import_imageBW('0130/white0130_0001.tif')
#white = crop(164, 166, 739, 1086,white)
#white_DMDres = convert_resolution(white, h, w)
#sim = np.ones((h,w))
#for i in range(h):
#    for j in range(w):
#        sim[i,j] = white_DMDres[i,j]*popt[0]*((Input[i,j]/255)**2)+popt[2]
#
#fig = plt.figure(figsize=(10,10))
#imshow(255 - sim , cmap = 'Greys', vmin = 0, vmax = 255)
#plt.title('Computed DMD pattern (DMD non lin, CCD offset)')
#plt.show()
#
##Plotting the profile
#ret = line_tool(x1, y1,x2,y2, sim) / 255. 
#x = range(ret.size)
#plt.subplot(121)
#plt.plot(x,ret)
#plt.ylim(0,1)
#plt.title('Intensity profile')
#
##Showing image with profile line
#plt.subplot(122)
#imshow(255. - sim, cmap = 'binary',vmin=0,vmax=255)
#plt.plot([y1,y2],[x1,x2], "r")
#plt.plot([y1],[x1],"ro")
#plt.plot([y2],[x2],"bo")
#plt.title('Image')
#
#plt.tight_layout()
#plt.show()
#
#ret = line_tool(x1, y1,x2,y2, white_DMDres) / 255. 
#x = range(ret.size)
#plt.subplot(121)
#plt.plot(x,ret)
#plt.ylim(0,1)
#plt.title('Intensity profile')
#
##Showing image with profile line
#plt.subplot(122)
#imshow(255. - white_DMDres, cmap = 'binary',vmin=0,vmax=255)
#plt.plot([y1,y2],[x1,x2], "r")
#plt.plot([y1],[x1],"ro")
#plt.plot([y2],[x2],"bo")
#plt.title('Image')
#
#plt.tight_layout()
#plt.show()

# In[]: letters!

w = 1920 #912
#Height :
h = 1080 #1140
#Width :

imageinput = import_imagecolour("A-01.tif")
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

img = np.ones((h,w),np.uint32)*0xFFFFFF

for i in range(h): 
    for j in range(w): 
        if int(imageinput[i,j]) == 1: 
            img[i,j] = np.uint32(0) * 0xFFFFFF
       

#img = imageinput*np.uint32(1)*0xFFFFFF
#Saving the image
pilImage = Image.frombuffer('RGBA',(w,h),img,'raw','RGBA',0,1)
pilImage = pilImage.convert('1') #convert to 1bit monochromatic image
#pilImage = pilImage.rotate(180) #rotate 180
pilImage = pilImage.transpose(Image.FLIP_TOP_BOTTOM)
#pilImage.save('Dot_middle_60x60.bmp')
pilImage.save('A.bmp')



