import cv2
import keras
import tensorflow as tf
from keras.applications import VGG16, VGG19
from keras.applications import DenseNet121
from keras.applications import ResNet50, ResNet152
from keras.applications import InceptionV3
from efficientnet.keras import EfficientNetB0, EfficientNetB3, EfficientNetB4
from keras.applications import Xception
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Conv2D, MaxPooling2D, Activation, Flatten

class ImageProcessing:
    def __init__(self, img_height, img_width, no_channels, tol=7, sigmaX=8):

        ''' Initialzation of variables'''

        self.img_height = img_height
        self.img_width = img_width
        self.no_channels = no_channels
        self.tol = tol
        self.sigmaX = sigmaX

    def cropping_2D(self, img, is_cropping = False):

        '''This function is used for Cropping the extra dark part of the GRAY images'''

        mask = img>self.tol
        return img[np.ix_(mask.any(1),mask.any(0))]

    def cropping_3D(self, img, is_cropping = False):

        '''This function is used for Cropping the extra dark part of the RGB images'''

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>self.tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # if image is too dark we return the image
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]  #for channel_1 (R)
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]  #for channel_2 (G)
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]  #for channel_3 (B)         
            img = np.stack([img1,img2,img3],axis=-1)
        return img

    def Gaussian_blur(self, img, is_gaussianblur = False):

        '''This function is used for adding Gaussian blur (image smoothing technique) which helps in reducing noise in the image.'''

        img = cv2.addWeighted(img,4,cv2.GaussianBlur(img,(0,0),self.sigmaX),-4,128)
        return img

    def draw_circle(self,img, is_drawcircle = True):

        '''This function is used for drawing a circle from the center of the image.'''

        x = int(self.img_width/2)
        y = int(self.img_height/2)
        r = np.amin((x,y))     # finding radius to draw a circle from the center of the image
        circle_img = np.zeros((img_height, img_width), np.uint8)
        cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        return img

    def image_preprocessing(self, img, is_cropping = True, is_gaussianblur = True):

        """
        This function takes an image -> crops the extra dark part, resizes, draw a circle on it, and finally adds a gaussian blur to the images
        Args : image - (numpy.ndarray) an image which we need to process
           cropping - (boolean) whether to perform cropping of extra part(True by Default) or not(False)
           gaussian_blur - (boolean) whether to apply gaussian blur to an image(True by Default) or not(False)
        Output : (numpy.ndarray) preprocessed image
        """

        if img.ndim == 2:
            img = self.cropping_2D(img, is_cropping)  #calling cropping_2D for a GRAY image
        else:
            img = self.cropping_3D(img, is_cropping)  #calling cropping_3D for a RGB image
        img = cv2.resize(img, (self.img_height, self.img_width))  # resizing the image with specified values
        img = self.draw_circle(img)  #calling draw_circle
        img = self.Gaussian_blur(img, is_gaussianblur) #calling Gaussian_blur
        return img

class final:
    def __init__(self):
        self.img_width = 512
        self.img_height = 512
        self.no_channels = 3
        self.obj = ImageProcessing(self.img_width, self.img_height, self.no_channels)
    def GAP2D(self):
        '''Global average pooling layer'''
        global_average_pooling = GlobalAveragePooling2D()
        return global_average_pooling
    def dropout(self):
        '''Dropout layer'''
        dropout_layer = Dropout(0.5)
        return dropout_layer
    def dense(self):
        '''Dense layer'''
        dense_layer = Dense(5, activation='sigmoid')
        return dense_layer
    def VGG16_(self):
        '''This function is used for building a model architecture of pretrained vgg16 on imagenet data set.'''
        vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (512,512,3))
        for layer in vgg.layers[:13]:
            layer.trainable=False 
        x = global_average_pooling_layer(vgg.layers[-1].output)
        x = dropout_layer(x)
        output = dense_layer(x)
        model = Model(vgg.layers[0].input,output)
        return model
    def VGG19_(self):
        '''This function is used for building a model architecture of pretrained Vgg19 on imagenet data set.'''
        vgg = VGG19(weights = 'imagenet', include_top = False, input_shape = (512,512,3))
        for layer in vgg.layers[:13]:
            layer.trainable=False 
        x = global_average_pooling_layer(vgg.layers[-1].output)
        x = dropout_layer(x)
        output = dense_layer(x)
        model = Model(vgg.layers[0].input,output)
        return model
    def DenseNet(self):
        '''This function is used for building a model architecture of pretrained Densenet121 on imagenet data set.'''
        densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(512,512,3))
        x = global_average_pooling_layer(densenet.layers[-1].output)
        x = dropout_layer(x)
        output = dense_layer(x)
        model = Model(densenet.layers[0].input,output)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.00005), metrics=['accuracy'])
        return model
    def ResNet50_(self):
        '''This function is used for building a model architecture of pretrained Resnet50 on imagenet data set.'''
        resnet = ResNet50(weights='imagenet',include_top=False,layers=keras.layers,input_shape=(512,512,3))
        x = global_average_pooling_layer(resnet.layers[-1].output)
        x = dropout_layer(x)
        output = dense_layer(x)
        model = Model(resnet.layers[0].input,output)
        return model
    def ResNet152_(self):
        '''This function is used for building a model architecture of pretrained Resnet152 on imagenet data set.'''
        resnet = ResNet152(weights='imagenet',include_top=False,layers=keras.layers,input_shape=(512,512,3))
        x = global_average_pooling_layer(resnet.layers[-1].output)
        x = dropout_layer(x) 
        output = dense_layer(x)
        model = Model(resnet.layers[0].input,output)
        return model
    def inceptionv3_(self):
        '''This function is used for building a model architecture of pretrained InceptionV3 on imagenet data set.'''
        inceptionv3 = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (512,512,3))
        x = global_average_pooling_layer(inceptionv3.layers[-1].output)
        x = dropout_layer(x) 
        output = dense_layer(x) 
        model = Model(inceptionv3.layers[0].input,output)
        return model
    def efficientnet_b0(self):
        '''This function is used for building a model architecture of pretrained EfficientB0 on imagenet data set.'''
        efficientnet_ = EfficientNetB0(weights = 'imagenet',include_top = False, input_shape = (512,512,3))
        x = global_average_pooling_layer(efficientnet_.layers[-1].output)
        x = dropout_layer(x) 
        output = dense_layer(x) 
        model = Model(efficientnet_.layers[0].input,output)
        return model
    def efficientnet_b3(self):
        '''This function is used for building a model architecture of pretrained EfficientB3 on imagenet data set.'''
        efficientnet_ = EfficientNetB3(weights = 'imagenet',include_top = False, input_shape = (512,512,3) )
        x = global_average_pooling_layer(efficientnet_.layers[-1].output)
        x = dropout_layer(x) 
        output = dense_layer(x) 
        model = Model(efficientnet_.layers[0].input,output)
        return model
    def efficientnet_b4(self):
        '''This function is used for building a model architecture of pretrained EfficientB4 on imagenet data set.'''
        efficientnet_ = EfficientNetB4(weights = 'imagenet',include_top = False, input_shape = (512,512,3) )
        x = global_average_pooling_layer(efficientnet_.layers[-1].output)
        x = dropout_layer(x) 
        output = dense_layer(x) 
        model = Model(efficientnet_.layers[0].input,output)
        return model
    def xception(self):
        '''This function is used for building a model architecture of pretrained Xception on imagenet data set.'''
        xception_ = Xception(weights = 'imagenet',include_top = False, input_shape = (512,512,3) )
        x = global_average_pooling_layer(xception_.layers[-1].output)
        x = dropout_layer(x) 
        output = dense_layer(x) 
        model = Model(xception_.layers[0].input,output)
        return model
    def test_prediction(self, predicted_labels):
        '''
        Making predictions of the probability scores. The class with more score will be taken as predicted class. 
        Arguments:
        predicted_labels - (np.array) - probability score of given sample
        '''
        predicted_labels = predicted_labels > 0.5
        prediction_ordinal = np.empty(predicted_labels.shape, dtype = int)
        prediction_ordinal[:,4] = predicted_labels[:,4]
        for i in range(3, -1, -1): prediction_ordinal[:, i] = np.logical_or(predicted_labels[:,i], prediction_ordinal[:,i+1])
        predicted_labels = prediction_ordinal.sum(axis = 1)-1
        return predicted_labels 
    def plot_gradient_heatmap(model, img, layer_name = 'last_conv_layer'):

        '''
        This function is used to plot the gradients for model interpretability.
        Reference and code credits: http://www.hackevolve.com/where-cnn-is-looking-grad-cam/
        Arguments:
        model - trained model
        img - (np.array) - image data
        layer_name - (string) - layer from which we need to take gradients
        '''
        # model predictions
        preds = model.predict(img[np.newaxis])   
        preds = preds_raw>0.5
        class_idx = (preds.astype(int).sum(axis=1)-1)[0]
        class_output_tensor = model.output[:,class_idx]

        # taking the output at particular convolutional layer
        layer_output = model.get_layer(layer_name)

        # taking gradients from model
        with tf.GradientTape() as tape:
        grads = model.predict(class_output_tensor, layer_output)[0]
        grads = tape.gradient(grads)[0]
        pooled_grads = K.mean(grads,axis=(0,1,2))
        iterate = K.function([model.input],[pooled_grads, layer_output.output[0]])
        pooled_grads, layer_output = iterate([img[np.newaxis]])
        for i in range(pooled_grads.shape[0]):
            layer_output[:,:,i] *= pooled_grads[i]

        # plotting heatmap
        heatmap = np.mean(layer_ouput, axis=-1)
        heatmap = np.maximum(heatmap,0)
        heatmap /= np.max(heatmap)     # normalizing the heatmap
        heatmap=cv2.resize(heatmap,(viz_img.shape[1],viz_img.shape[0]))  # resizing the image
        heatmap_color = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_SPRING)/255
        heated_img = heatmap_color*0.5 + viz_img*0.5
        return heated_img
    def plotting(self, img1, img2, plot_preprocessed = True, plot_gradients = False):
        plt.figure(figsize=(15,3))
        plt.subplot(1,3,1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        plt.imshow(img1)
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(img2)
        plt.axis('off')
        if plot_gradients:
            plt.subplot(1,3,3)
            img = self.plot_gradient_heatmap(model,img2)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(224,224))
            img = self.obj.draw_circle(img)
            plt.imshow(img)
            plt.axis('off')
        plt.show()
    def prediction(self, img):
        id = img.split('/')[-1]
        img = cv2.imread(img)
        img1 = self.obj.image_preprocessing(img)
        pred = []
        pred.append(self.test_prediction(vgg16.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(vgg19.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(densenet.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(resnet50.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(resnet152.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(inceptionv3.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(efficientb0.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(efficientb3.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(efficientb4.predict(np.expand_dims(img1,axis=0))))
        pred.append(self.test_prediction(xception_.predict(np.expand_dims(img1,axis=0))))
        
        max_label=-1; max_count=-1
        for i in list(set(pred)):
            if pred.count(i)>max_count: max_count = pred.count(i); max_label = i
        print("Predicted as:",max_label)
        
        self.plotting(img,img1,plot_gradients=True)
    def modeling(self,data):
        
        global_average_pooling_layer = self.GAP2D()
        dropout_layer = self.dropout()
        dense_layer = self.dense()
        
        vgg16 = self.VGG16_()
        vgg16.load_weights("/content/drive/My Drive/models/vgg16.h5")
        vgg19 = self.VGG19_()
        vgg19.load_weights("/content/drive/My Drive/vgg19.h5")
        densenet = self.DenseNet()
        densenet.load_weights("/content/drive/My Drive/models/densenet.h5")
        resnet50 = self.ResNet50_()
        resnet50.load_weights("/content/drive/My Drive/models/resnet50.h5")
        resnet152 = self.ResNet152_()
        resnet152.load_weights("/content/drive/My Drive/models/resnet152.h5")
        inceptionv3 = self.inceptionv3_()
        inceptionv3.load_weights("/content/drive/My Drive/models/inceptionv3.h5")
        efficientb0 = self.efficientnet_b0()
        efficientb0.load_weights("/content/drive/My Drive/models/efficientnet_b0.h5")
        efficientb3 = self.efficientnet_b3()
        efficientb3.load_weights("/content/drive/My Drive/models/efficientnet_b3.h5")
        efficientb4 = self.efficientnet_b4()
        efficientb4.load_weights("/content/drive/My Drive/models/efficientnet_b4.h5")
        xception_ = self.xception()
        xception_.load_weights("/content/drive/My Drive/models/xception.h5")

        if len(data)==1:self.prediction(data[0])
        else:
            for i in range(len(data)): self.prediction(data[i])
