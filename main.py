#-------IMAGE EDITOR-------------------
'''import all the necessary libraries
1) OpenCV for reading,writing of images and color space conversion from RGB to HSV.
2) NumPy for array operations and vectorization.
3) PyQt5 (version 5) for GUI. 
4) Matplotlib.pyplot for plotting the histogram of images.
'''
import os
import sys,cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

cwd = os.getcwd()

'''
Any image is represented in the form of RGB format with each colour having red,green and blue components(red,green,blue).
This RGB triplet (red,green,blue) is represented by QRgb.
qRgb function creates and return the QRgb triplet values. '''
gray_color_table = [qRgb(i, i, i) for i in range(256)]

class Editor(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '213079003-Image-Editor'    # declares the editor - window's title
        self.number_grids = 3
        self.initialise_User_Interface()   # initialises the User Interface
 
    
    def initialise_User_Interface(self):
        self.setWindowTitle(self.title)    # sets the editor - window's title as defined
        geometry = app.desktop().availableGeometry()        
        # the window is set to the size of the screen i.e. full-screen mode.
        self.setGeometry(geometry)  
        self.setStyleSheet("background-color : Azure")
        windowLayout = QHBoxLayout()       # sets the geometry of the window

# The entire layout is divided into three grids.
# 1) First grid - provides the list of buttons 
# 2) Second grid - for displaying the original image
# 3) Third grid - for displaying the processed image.
                                                      
# The entire layout is taken as an empty list initially and three grids are appended to it.        
        self.layout = []                                
        number_layouts = 3                        
        for i in range(number_layouts):
            self.layout.append(QGridLayout())  
         
#adjusts the grid dimensions            
        self.layout[1].setColumnStretch(1, 2)
        self.layout[2].setColumnStretch(1, 2) 
        
#defining the push buttons using QPushButton widget 
        button_1 = QPushButton('Load Image', self)
        button_2 = QPushButton('Histogram Equalization',self)
        button_3 = QPushButton('Image Negative',self)
        button_4 = QPushButton('Gamma Correction',self)
        button_5 = QPushButton('Log Transform', self)
        button_6 = QPushButton('Image Blur', self)
        button_7 = QPushButton('Image Sharpening', self)
        button_8 = QPushButton('Edge Detection', self)
        button_9 = QPushButton('Undo Last', self)
        button_10 = QPushButton('Undo All', self)
        button_11 = QPushButton('Save', self) 
        button_12 = QPushButton('Exit', self) 
        
#setting colours for the push buttons
        button_1.setStyleSheet("color : blue; background-color : lightpink;")
        button_2.setStyleSheet("color : black; background-color : lightblue;")
        button_3.setStyleSheet("color : blue; background-color : lightpink")
        button_4.setStyleSheet("color : black; background-color : lightblue;")
        button_5.setStyleSheet("color : blue; background-color : lightpink;")
        button_6.setStyleSheet("color : black; background-color : lightblue;")
        button_7.setStyleSheet("color : blue; background-color : lightpink;")
        button_8.setStyleSheet("color : black; background-color : lightblue;")
        button_9.setStyleSheet("color : blue; background-color : lightpink;")
        button_10.setStyleSheet("color : black; background-color : lightblue;")
        button_11.setStyleSheet("color : blue; background-color : lightpink;")
        button_12.setStyleSheet("color : black; background-color : lightblue;")
        
#setting icons for the push buttons
        button_1.setIcon(QIcon(cwd+'\\logos\\load.PNG'))
        button_2.setIcon(QIcon(cwd+'\\logos\\histogram.PNG'))
        button_3.setIcon(QIcon(cwd+'\\logos\\negative.PNG'))
        button_4.setIcon(QIcon(cwd+'\\logos\\gamma.PNG'))
        button_5.setIcon(QIcon(cwd+'\\logos\\log.PNG'))
        button_6.setIcon(QIcon(cwd+'\\logos\\blurring.PNG'))
        button_7.setIcon(QIcon(cwd+'\\logos\\sharpening.JPG'))
        button_8.setIcon(QIcon(cwd+'\\logos\\edge.PNG'))
        button_9.setIcon(QIcon(cwd+'\\logos\\Undo.PNG'))
        button_10.setIcon(QIcon(cwd+'\\logos\\reset1.JPG'))
        button_11.setIcon(QIcon(cwd+'\\logos\\save.JPG'))
        button_12.setIcon(QIcon(cwd+'\\logos\\exit.PNG'))
        
#setting dimensions for the push buttons
        button_1.setIconSize(QSize(25,25))
        button_2.setIconSize(QSize(25,25))
        button_3.setIconSize(QSize(25,25))
        button_4.setIconSize(QSize(25,25))
        button_5.setIconSize(QSize(25,25))
        button_6.setIconSize(QSize(25,25))
        button_7.setIconSize(QSize(25,25))
        button_8.setIconSize(QSize(25,25))
        button_9.setIconSize(QSize(25,25))
        button_10.setIconSize(QSize(25,25))
        button_11.setIconSize(QSize(25,25))
        button_12.setIconSize(QSize(25,25))

# each pushbutton upon clicking performs its respective function.      
        button_1.clicked.connect(self.load_Image)
        button_2.clicked.connect(self.histogram_Equalization)
        button_3.clicked.connect(self.image_Negative)
        button_4.clicked.connect(self.gamma_Correction)
        button_5.clicked.connect(self.log_Transformation)
        button_6.clicked.connect(self.image_Blurring)
        button_7.clicked.connect(self.image_Sharpening)
        button_8.clicked.connect(self.edge_Detection)
        button_9.clicked.connect(self.undo_Last)
        button_10.clicked.connect(self.undo_All)
        button_11.clicked.connect(self.save)
        button_12.clicked.connect(self.exit)
        
#adding each button in the first grid of the layout. 
#This is performed using the fuction addWidget(button,row,column)  
#since all the buttons are added in the first grid, the column= 0        
        self.layout[0].addWidget(button_1,0,0),
        self.layout[0].addWidget(button_2,1,0)
        self.layout[0].addWidget(button_3,2,0)
        self.layout[0].addWidget(button_4,3,0)
        self.layout[0].addWidget(button_5,4,0)
        self.layout[0].addWidget(button_6,5,0)
        self.layout[0].addWidget(button_7,6,0)
        self.layout[0].addWidget(button_8,7,0)
        self.layout[0].addWidget(button_9,8,0)
        self.layout[0].addWidget(button_10,9,0)
        self.layout[0].addWidget(button_11,10,0)
        self.layout[0].addWidget(button_12,11,0)

#defining the names for each grid.
# 1)First layout is named as  Buttons Grid.
# 2)Second layout is named as  Original Image.
# 3)Third layout is named as  Modified Image.

# The labels is taken as an empty list initially and three labels names are appended to it.     
        grid_labels = ["BUTTONS GRID", "ORIGINAL IMAGE", "MODIFIED IMAGE"]
        self.labels=[]
        for i in range(self.number_grids):            
            self.labels.append(QGroupBox(grid_labels[i]))            
            self.labels[i].setLayout(self.layout[i])
            
#adding the defined labels to the grids.      
        for i in range(self.number_grids):
            windowLayout.addWidget(self.labels[i])
            
        self.setLayout(windowLayout)  # sets the layout of editor-window to be a windowLayout
        self.show()     # displays the editor-window on screen
        
# initialize the pyqt QLabel widget for always displaying the original Image
        self.label = QLabel(self)
        
# the pyqt QLabel widget for always displaying the modified Image. 
# Initialized to original image at the beginning. 
# So both the Original and Modified image grids display the original image initially.                    
        self.label_result = QLabel(self)  
                    
# both the declared label widgets are added to 2nd & 3rd grids respectively.                                                
        self.layout[1].addWidget(self.label)                                                          
        self.layout[2].addWidget(self.label_result)

# function to display the modified (output) image.        
    
    def displayOutput(self, output):
                
         # combine V-channel of input image(self.inputImage) with the H,S channels.
        self.hsvImage[:,:,2] = self.inputImage
         # storing the input image(in RGB format) as previous image
        self.previousImage = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
         # combine V-channel of output image (output) with the H,S channels.
        self.hsvImage[:,:,2] = output
         # update the image display with the modified RGB image
        qImage = self.CV_to_QImage(cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB))
         # display the output image
        self.label_result.setPixmap(QPixmap(qImage))
         # set the output as input image.
        self.inputImage = output

# function for selecting the image.   
    def openFile(self):
        fileLoadOptions = QFileDialog.Options()
        fileLoadOptions |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Image File")
#if the user selects a file with extensions .jpg, .png, .jpeg, .bmp , .JPG, .PNG', .JPEG, .BMP , then it is an allowed file.
# if user selects an allowed file
        if fileName.split('.')[1].lower() in ['jpg', 'png', 'jpeg', 'bmp']:   
            # reads the selected file(i.e.image) and resizes to (image_width, image_height)=(480, 360)
            self.originalImage = cv2.resize(cv2.imread(fileName, 1), (480, 360)) 
            
            #OpenCV reads an image in BGR format..and thus convert into RGB for further processing
            self.originalImage = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2RGB) 
            
            # previous image variable for "Undo" Operation and currently set to the original image.                                                                    
            self.previousImage = self.originalImage  
            
            # convert image format from RGB to HSV for operations on V channel
            self.hsvImage = cv2.cvtColor(self.originalImage, cv2.COLOR_RGB2HSV)   
            # inputimage = V channel of original HSV image
            self.inputImage = self.hsvImage[:, :, 2].astype(np.uint8)
            
            # convert OpenCV image into PyQt QImage for displaying in Pixmap
            qImage = self.CV_to_QImage(self.originalImage)     
            
            # set pixmap with the original loaded image in both original and modified grids
            self.label.setPixmap(QPixmap(qImage))
            self.label_result.setPixmap(QPixmap(qImage))
            
#  self.originalImage is the original selected image in RGB format
#  self.previousImage is the variable used to store the previous image in RGB format
#  self.hsvImage is variable used to store the image in HSV format
#  self.inputImage is the selected image having V-channel  

# case when user selects a file with other extensions and raises an error asking for proper file
        elif (fileName):                                                        # case when user selects a file with extension not in "possibleImageTypes"
            buttonReply = QMessageBox.question(self, 'Warning Message', "Wrong File Selection", QMessageBox.Ok, QMessageBox.Ok)

#converts an OpenCV image into  a QImage.   
    def CV_to_QImage(self, img, copy=False):
    #input image must be in OpenCV image format (np.uint8)
         
      # if input image format is OpenCV image format np.uint8
        if img.dtype == np.uint8:
           # grayscale images or the images having two dimensions [height, width]
            if len(img.shape) == 2:
                qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim
            # images having three dimensions [height, width, number of Channels]
            elif len(img.shape) == 3:
                # if the image has three channels
                if img.shape[2] == 3:  
                    qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                # if the image has four channels
                elif img.shape[2] == 4:  
                    qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim

    
    def correlation(self, image, window):
        output = np.zeros(image.shape,dtype=np.uint8)
        # find no of rows and columns in the ndarray of image passed
        img_height = image.shape[0]
        img_width = image.shape[1]

        # compute window size from window passed
        window_size = window.shape[0]
        # compute zero padding requirements and offset to be used during convolution
        zero_padding = window_size - 1
        offset = int(zero_padding / 2)

        # create the zero padded image
        paddedImage =  np.zeros((img_height + zero_padding, img_width+ zero_padding),dtype=np.uint8)
        paddedImage[offset: img_height + offset, offset:img_width +offset] = image

        # loop through the window elements
        # computes correlation as shifted sum of image elements by keeping  window stationary
        
        for i in range(img_height):                                                     # convolution of zero-padded image and gaussian kernel
                for j in range(img_width):                        
                    product = paddedImage[i:i + window_size, j:j + window_size]*window
                    output[i,j] = np.round(np.sum(product))

        #  return the computed image
        return output
    
    def convolution(self, image, window):
        # to perform convolution, flip the window and compute correlation
        window = np.flipud(np.fliplr(window))
        output = self.correlation(image, window)
        #  return the computed image
        return output
    
    
    
 # opens a dialog box for the user to select an Image     
    @pyqtSlot()
    def load_Image(self):
        self.openFile()             

# function performing histogram equalization     
    @pyqtSlot()
    def histogram_Equalization(self):
     # Histrogram Equalization Function has the following algorithm:
     # 1. Finds the histogram of input image. 
     # 2. Computes the probability distribution of each intensity.
     # 3. Computes the cumulative distribution of input image.
     # 4. Assign the new intensity values to pixels by scaling the cdf by 255.
     # 5. Computes the equalised histogram.
     # 6. Plots the input image histogram and equalized image histogram.  
     
# considering a 8-bit image, the total number of discrete intensity levels are 256                                                                        
        L = 256 
        img_height, img_width = self.inputImage.shape # variables storing the image dimensions
        outputImage = np.zeros(self.inputImage.shape)   # numpy array of zeros for output "histogram equalized" image
        original_img_hist = np.zeros((L, 1))    # variable to store the input image histogram
        equalized_img_hist = np.zeros((L, 1))   # variable to store the equalized histogram
        
        # Compute the histogram of input image
        for i in range(L):   
            original_img_hist[i, 0] = np.sum(self.inputImage == i) #counting the number of pixels of each intensity.
         
        # Compute the probability distribution by dividing each intensity count by the total number of pixels
        original_img_hist = original_img_hist/ (img_height*img_width) 
        
        # Initialize the cdf of input image as zero
        cdf = np.zeros(original_img_hist.shape)
        sum_Hist = 0 # variable to store the sum of previous probabilities.
        for i in range(L):
            sum_Hist+= original_img_hist[i,0]                    
            cdf[i,0] = sum_Hist # updates the cdf with the sum of probabilities
       
        ''' compute the transform values for each intensity from [0-255] and assign it 
       to the pixels locations where that intensity is present in input image 
       and the output image is found.'''                                 
                                              
        for i in range(L):                                                   
            outputImage[np.where(self.inputImage == i)] = np.round(((L-1))*cdf[i])
        
        # compute the equalized histogram of output image
        for i in range(L):                                                
            equalized_img_hist[i, 0] = np.sum(outputImage == i)           
        
        outputImage = outputImage.astype(np.uint8)  # change output image type for display purposes

        # plotting the input and ouput histograms
        
        plt.subplot(2,1,1)
        plt.plot(original_img_hist, linewidth=1)
        plt.xlabel('Intensity')
        plt.ylabel('Number of Pixels')
        plt.grid(True)
        plt.subplot(2,1,2)
        plt.plot(equalized_img_hist, linewidth=1)
        plt.xlabel('Intensity')
        plt.ylabel('Number of Pixels')
        plt.grid(True)
        plt.suptitle('Original vs Equalized Image Histograms')
        plt.show()

        self.displayOutput(outputImage) # display the output image  
        
# function performing image negative
    @pyqtSlot()
    def image_Negative(self):
        # convert image from intensity range [0-255] to [0-1]
        normalised_Image = cv2.normalize(self.inputImage.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  
        outputImage = 1 - normalised_Image     # finding output intensities
        max_intensity = np.max(outputImage)    # finding the maximum intensity value  of output image
        outputImage = (outputImage/max_intensity)*255.0 # bring the output image to [0-255] range
        outputImage = outputImage.astype(np.uint8)   # change output image type for display purposes
        
        self.displayOutput(outputImage)     # display the output image 

# function performing gamma correction   
    @pyqtSlot()
    def gamma_Correction(self):
        
     # opens a dialog asking for gamma value and stores the gamma bvalue in a variable "gamma"
        gamma, ok_is_Pressed = QInputDialog.getDouble(self, "Gamma", "Gamma Value:") 
        if ok_is_Pressed:      # compute only if the user presses "OK" 
            # convert image from intensity range [0-255] to [0-1]
            normalised_Image = cv2.normalize(self.inputImage.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) 
            outputImage = np.power(normalised_Image, gamma)     # finding output intensities          
            max_intensity = np.max(outputImage)          # finding the maximum intensity value  of output image
            outputImage = (outputImage/max_intensity)*255.0    # bring the output image to [0-255] range
            outputImage = outputImage.astype(np.uint8)     # change output image type for display purposes                                   
           
            self.displayOutput(outputImage)   # display the output image 

# function performing log transformation   
    @pyqtSlot()
    def log_Transformation(self):
        # convert image from intensity range [0-255] to [0-1]
        normalised_Image = cv2.normalize(self.inputImage.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) 
        outputImage = np.log(1+normalised_Image)    # finding output intensities                                                 
        max_intensity = np.max(outputImage)       # finding the maximum intensity value  of output image                                      
        outputImage = (outputImage/max_intensity)*255.0    # bring the output image to [0-255] range                                  
        outputImage = outputImage.astype(np.uint8)   # change output image type for display purposes                                     
        
        self.displayOutput(outputImage)        # display the output image                                                

# function performing blurring operation    
    @pyqtSlot()
    def image_Blurring(self):
        """
            Blur Image using Gaussian Kernel with kernel size and gaussian variance
            chosen by initUserInterface
        """

        #opens a dialog prompting user for filter size
        filter_size, ok_1_pressed = QInputDialog.getInt(self, "Size", "Value of filter (odd number > 0):")      
        #if user gives odd size of filter        
        if (ok_1_pressed  and filter_size%2!=0):
        # opens a dialog prompting user for sigma     
            sigma, ok_2_pressed = QInputDialog.getDouble(self, "Sigma", "Value of sigma (> 0):")     
            if (ok_2_pressed):
                # creates a gaussian kernel of given window size and sigma
                kernel = GaussianKernel(filter_size, sigma)     
                # perfomes convolution of image and kernel to form the blurred image
                blurredImage = self.convolution(self.inputImage, kernel)
                blurredImage = blurredImage/(kernel.sum())
                blurredImage = blurredImage.astype(np.uint8)  # change output image type for display purposes
                
                self.displayOutput(blurredImage)    # display the output image
                
        #if user gives even size of filter 
        elif (ok_1_pressed  and filter_size%2==0): 
            # displays a warning asking for odd value
            buttonReply = QMessageBox.question(self, 'Warning Message', "Kernel Size has to be an odd number", QMessageBox.Ok, QMessageBox.Ok)
            # again opens the dialog for filter size and produces the blurred image
            filter_size, ok_1_pressed = QInputDialog.getInt(self, "Size", "Value of filter (odd number > 0):")      
            if (ok_1_pressed  and filter_size%2!=0):
                sigma, ok_2_pressed = QInputDialog.getDouble(self, "Sigma", "Value of sigma (> 0):")     
                if (ok_2_pressed):

                    kernel = GaussianKernel(filter_size, sigma)      
                    blurredImage = self.convolution(self.inputImage, kernel)
                    blurredImage = blurredImage/(kernel.sum())
                    blurredImage = blurredImage.astype(np.uint8)                            
                    self.displayOutput(blurredImage)    
    
# function performing sharpening operation     
    @pyqtSlot()
    def image_Sharpening(self):
        """
            Sharpen Image using "Unsharp Masking" by using a Gaussian Kernel with kernel size and gaussian variance
             and a scaling constant k chosen by initUserInterface
        """
        #opens a dialog prompting user for filter size
        filter_size, ok_1_pressed = QInputDialog.getInt(self, "Size", "Value of filter (odd number > 0):")      
        #if user gives odd size of filter        
        if (ok_1_pressed  and filter_size%2!=0):
        # opens dialog boxes  prompting user for sigma and k
            sigma, ok_2_pressed = QInputDialog.getDouble(self, "Sigma", "Value of sigma (> 0):")
            k, ok_3_pressed = QInputDialog.getDouble(self, "k value", "Scaling Constant Value (default = 5):")
            if (ok_2_pressed and ok_3_pressed ):     
                # creates a gaussian kernel of given window size and sigma
                kernel = GaussianKernel(filter_size, sigma) 
                # perfomes convolution of image and kernel to form the blurred image
                blurredImage = self.convolution(self.inputImage, kernel)
                blurredImage = blurredImage/(kernel.sum())
                # converting both blurred and input images into same format
                blurredImage = blurredImage.astype(np.uint8)
                inputimg = self.inputImage.astype(np.uint8)
                # scaled version of mask computed by subtracting the blurred image from original image and scaling by k
                maskImage = k*cv2.subtract(inputimg, blurredImage)
                maskImage = maskImage.astype(np.uint8)
                # sharp image computed by adding scaled mask with original image
                sharpenedImage = cv2.add(self.inputImage, maskImage)
                for i in range(256):                                                   
                    sharpenedImage[np.where(i <0)] = 0
                    sharpenedImage[np.where(i >255)] = 255
                sharpenedImage = sharpenedImage.astype(np.uint8) # change output image type for display purposes
                
                self.displayOutput(sharpenedImage) # display the output image
        #if user gives even size of filter 
        elif (ok_1_pressed  and filter_size%2==0):
            # displays a warning asking for odd value
            buttonReply = QMessageBox.question(self, 'Warning Message', "Kernel Size has to be an odd number", QMessageBox.Ok, QMessageBox.Ok)
            # again opens the dialog for filter size and produces the sharpened image  
            filter_size, ok_1_pressed = QInputDialog.getInt(self, "Size", "Value of filter (odd number > 0):")      
                 
            if (ok_1_pressed  and filter_size%2!=0):
            
                sigma, ok_2_pressed = QInputDialog.getDouble(self, "Sigma", "Value of sigma (> 0):")
                k, ok_3_pressed = QInputDialog.getDouble(self, "k value", "Scaling Constant Value (default = 5):")
                if (ok_2_pressed and ok_3_pressed ):    
                    kernel = GaussianKernel(filter_size, sigma) 
                    blurredImage = self.convolution(self.inputImage, kernel)
                    blurredImage = blurredImage/(kernel.sum())
                    blurredImage = blurredImage.astype(np.uint8)
                    inputimg = self.inputImage.astype(np.uint8)
                    maskImage = k*cv2.subtract(self.inputImage, blurredImage)
                    maskImage = maskImage.astype(np.uint8)
                    sharpenedImage = cv2.add(self.inputImage, maskImage)
                    for i in range(256):                                                   
                        sharpenedImage[np.where(i <0)] = 0
                        sharpenedImage[np.where(i >255)] = 255
                    sharpenedImage = sharpenedImage.astype(np.uint8) 
                    
                    self.displayOutput(sharpenedImage) 

# function performing edge detection         
    @pyqtSlot()
    def edge_Detection(self):
        """
            Edge detection is performed using Fixed 3x3 Gaussian kernel with a 
            standard deviation of 1.6 and using a 3x3 Sobel Operator
             
        """
        
        
        sigma = 1.6       # standard deviation for gaussian blurring 
        filter_size = 3    # kernel size for gaussian blurring
      
        # combine V-channel (self.image) with H,S channel
        self.hsvImage[:,:,2] = self.inputImage
        # convert into rgb
        img = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
        # convert into gray-scale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.int32)
        # storing the dimensions of input image
        img_height, img_width = img.shape
        # creates a gaussian kernel of given window size and sigma
        kernel = GaussianKernel(filter_size, sigma)
    # perform convolution of image and kernel to form the blurred image
        zero_padding = filter_size - 1
        offset = int(zero_padding / 2)
        # zero padded image for convolution with gaussian kernel
        paddedImage =  np.zeros((img_height + zero_padding, img_width+ zero_padding),dtype=np.uint32)
        paddedImage[offset: img_height + offset, offset:img_width +offset] = img
        blurredImage = np.zeros(self.inputImage.shape,dtype=np.uint32)
        
        # loop through the window elements
        # computes correlation as shifted sum of image elements by keeping  window stationary
        for i in range(img_height):                                                 
          for j in range(img_width):
            product = paddedImage[i:i + filter_size, j:j + filter_size]*kernel
            blurredImage[i,j] = np.round(np.sum(product))   
        
        # kernel for finding gradient in X & Y direction
        kernel_Gradient_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)    # vertical edges
        kernel_Gradient_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)    # horizontal edges

        paddedImage[offset: img_height + offset, offset:img_width +offset] = blurredImage
        gradientX = np.zeros(img.shape, dtype=np.int32)
        gradientY = np.zeros(img.shape, dtype=np.int32)

     # convolution for finding gradients
        for i in range(img_height):
            for j in range(img_width):
              horizontal_gradient = paddedImage[i:i + filter_size, j:j + filter_size]*(-1*kernel_Gradient_X)
              vertical_gradient = paddedImage[i:i + filter_size, j:j + filter_size]*(-1*kernel_Gradient_Y)
              gradientX[i,j] = np.round(np.sum(horizontal_gradient))
              gradientY[i,j] = np.round(np.sum(vertical_gradient))

        sobel_operated_img = np.sqrt((np.power(gradientX,2))+ (np.power(gradientY,2)))
        for i in range(256):                                                   
            sobel_operated_img[np.where(i <0)] = 0
            sobel_operated_img[np.where(i >255)] = 255
        edgeDetectedImage = 255 * cv2.normalize(sobel_operated_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        self.displayOutput(edgeDetectedImage)

# function performing undo operation    
    @pyqtSlot()
    def undo_Last(self):
      # combine V-channel (self.image) with H,S channel
      self.hsvImage[:,:,2] = self.inputImage
      
      # storing the image in a temporary variable
      temp = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
      
      #display the previous image
      qImage = self.CV_to_QImage(self.previousImage)
      self.label_result.setPixmap(QPixmap(qImage))
      
      # assign the V-channel of previous image to present image after converting it to HSV
      self.inputImage = cv2.cvtColor(self.previousImage, cv2.COLOR_RGB2HSV)[:,:,2]
      
      # assign present image to previous image variable 
      self.previousImage = temp       

# function performing reset operation    
    @pyqtSlot()
    def undo_All(self):
        # display original image
      qImage = self.CV_to_QImage(self.originalImage)
      self.label_result.setPixmap(QPixmap(qImage))
      # combine V-channel (self.image) with H,S channel
      self.hsvImage[:,:,2] = self.inputImage
      # store the present image as previous image
      self.previousImage = cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2RGB)
      # assign the V-channel of original image to present image after converting it to HSV
      self.inputImage = cv2.cvtColor(self.originalImage, cv2.COLOR_RGB2HSV)[:,:,2]          

# function for saving the image    
    @pyqtSlot()
    def save(self):
        fileSaveOptions = QFileDialog.Options()
        fileSaveOptions |= QFileDialog.DontUseNativeDialog
        # open a dialog box for choosing destination location for saving image
        fileName, _ = QFileDialog.getSaveFileName(self,"Save Image File")
        # combine V-channel with H,S channel  
        self.hsvImage[:,:,2] = self.inputImage
        #convert HSV to BGR since OpenCV default color format is BGR
        # save image at location specified by fileName
        cv2.imwrite(fileName, cv2.cvtColor(self.hsvImage, cv2.COLOR_HSV2BGR))

# function for closing the application    
    @pyqtSlot()
    def exit(self):
        self.close()    # in-built function of QWidget class


#function gor creating gaussian kernel
def GaussianKernel(winSize, sigma):
    """
        generates a gaussian kernel taking window size = winSize
        and standard deviation = sigma as input/control parameters. 
        Returns the gaussian kernel
    """
    kernel = np.zeros((winSize, winSize))     # generate a zero numpy kernel
    # finding each window coefficient 
    for i in range(winSize):
        for j in range(winSize):
            temp = pow(i-winSize//2,2)+pow(j-winSize//2,2)
            kernel[i,j] = np.exp(-1*temp/(2*pow(sigma,2)))
    kernel = kernel/(2*np.pi*pow(sigma,2))
    norm_factor = np.sum(kernel)               # finding the sum of the kernel generated
    kernel = kernel/norm_factor                # dividing by total sum to make the kernel matrix unit-sum
    return kernel

if __name__ == '__main__':
    app = QApplication(sys.argv)    # defines pyqt application object 
    ex = Editor()
    sys.exit(app.exec_())
