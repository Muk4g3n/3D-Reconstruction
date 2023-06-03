from tensorflow.keras.models import Model
from tensorflow.keras import layers,losses
from CBAM import ConvolutionBlockAttentionModule

class AutoEncoder(Model):

    def __init__(self,reducedDimension,originalSize) -> None:
        super(AutoEncoder,self).__init__()
        self.reducedDimension = reducedDimension
        
        
        ########################################## Encoder Layers ##########################################
        
        self.conv_1 = layers.Conv2D(16, (3,3) ,input_shape = originalSize , activation='relu', padding='same')
        self.maxPool_1 = layers.MaxPooling2D((2,2), padding='same')
        self.batchNorm_1 = layers.BatchNormalization()
        
        
        self.conv_2 = layers.Conv2D(32, (3,3) ,input_shape = originalSize , activation='relu', padding='same')
        self.maxPool_2 = layers.MaxPooling2D((2,2), padding='same')
        self.batchNorm_2 = layers.BatchNormalization()
        
        
        self.conv_3 = layers.Conv2D(64, (3,3) ,input_shape = originalSize , activation='relu', padding='same')
        self.maxPool_3 = layers.MaxPooling2D((2,2), padding='same')
        self.batchNorm_3 = layers.BatchNormalization()
        
        
        
        self.conv_4 = layers.Conv2D(128, (3,3) ,input_shape = originalSize , activation='relu', padding='same')
        self.maxPool_4 = layers.MaxPooling2D((2,2), padding='same')
        self.batchNorm_4 = layers.BatchNormalization()
        
        
        
        self.conv_5 = layers.Conv2D(256, (3,3), activation='relu', padding='same')
        self.maxPool_5 = layers.MaxPooling2D((2,2), padding='same')
        self.batchNorm_5 = layers.BatchNormalization()
        
        
        self.latentDense = layers.Dense(reducedDimension)
        
        self.batchNorm_6 = layers.BatchNormalization()
        
        ########################################## Decoder Layers ##########################################
        
        
        self.upDense = layers.Dense(8*8*256)
        self.batchNorm_7 = layers.BatchNormalization()
        
        self.convT_1 = layers.Conv2DTranspose(256, (3,3), strides=2, activation='relu', padding='same')
        self.batchNorm_8 = layers.BatchNormalization()
        
        self.cbam_1 = ConvolutionBlockAttentionModule()
        self.convT_2 = layers.Conv2DTranspose(128, (3,3), strides=2, activation='relu', padding='same')
        self.batchNorm_9 = layers.BatchNormalization()
        
        self.cbam_2 = ConvolutionBlockAttentionModule()
        self.convT_3 = layers.Conv2DTranspose(64, (3,3), strides=2, activation='relu', padding='same')
        self.batchNorm_10 = layers.BatchNormalization()
        
        self.cbam_3 = ConvolutionBlockAttentionModule()
        self.convT_4 = layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')
        self.batchNorm_11 = layers.BatchNormalization()
        
        self.cbam_4 = ConvolutionBlockAttentionModule()
        self.convT_5 = layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same')
        self.batchNorm_12 = layers.BatchNormalization()
        
        
        self.convT_6 = layers.Conv2DTranspose(1, (3,3), activation='sigmoid', padding='same')
        
        
        
        
        def encoder(self,X):
            x1 = self.conv_1(X)
            x1d = self.batchNorm_1(x1)
            x1 = self.maxPool_1(x1d)


            x2 = self.conv_2(x1)
            x2d = self.batchNorm_2(x2)
            x2 = self.maxPool_2(x2d)


            x3 = self.conv_3(x2)
            x3d = self.batchNorm_3(x3)
            x3 = self.maxPool_3(x3d)



            x4 = self.conv_4(x3)
            x4d = self.batchNorm_4(x4)
            x4 = self.maxPool_4(x4d)


            x5 = self.conv_5(x4)
            x5d = self.batchNorm_5(x5)
            x5 = self.maxPool_5(x5d)


            flat = layers.Flatten()(x5)
            latentDim = self.latentDense(flat)
            encodedFeatures = self.batchNorm_6(latentDim)
            return encodedFeatures.numpy()
        
        def decoder(self,latentDim):
            
            up = self.upDense(latentDim)
            reshapedUp = layers.Reshape((8, 8, 256))(up)
            reshapedUp = self.batchNorm_7(reshapedUp)

            x6 = self.convT_1(reshapedUp)
            x6 = self.batchNorm_8(x6)

            x6 = self.cbam_1(x6)
            x7 = self.convT_2(x6)
            x7 = self.batchNorm_9(x7)
            

            x7 = self.cbam_2(x7)
            x8 = self.convT_3(x7)
            x8 = self.batchNorm_10(x8)
            

            x8 = self.cbam_3(x8)
            x9 = self.convT_4(x8)
            x9 = self.batchNorm_11(x9)
            


            x9 = self.cbam_4(x9)
            x10 = self.convT_5(x9)
            x10 = self.batchNorm_12(x10)
            

            decoded = self.convT_6(x10)
            return decoded
    
    def call(self,X):
        
        ### encoder
        x1 = self.conv_1(X)
        x1d = self.batchNorm_1(x1)
        x1 = self.maxPool_1(x1d)
        
        
        x2 = self.conv_2(x1)
        x2d = self.batchNorm_2(x2)
        x2 = self.maxPool_2(x2d)
       
        
        x3 = self.conv_3(x2)
        x3d = self.batchNorm_3(x3)
        x3 = self.maxPool_3(x3d)
       
        
        
        x4 = self.conv_4(x3)
        x4d = self.batchNorm_4(x4)
        x4 = self.maxPool_4(x4d)
        
        
        x5 = self.conv_5(x4)
        x5d = self.batchNorm_5(x5)
        x5 = self.maxPool_5(x5d)
        
        
        flat = layers.Flatten()(x5)
        latentDim = self.latentDense(flat)
        latentDim = self.batchNorm_6(latentDim)
        
        # Decoder
        
        up = self.upDense(latentDim)
        reshapedUp = layers.Reshape((8, 8, 256))(up)
        reshapedUp = self.batchNorm_7(reshapedUp)
        
        x6 = self.convT_1(reshapedUp)
        x6 = self.batchNorm_8(x6)
        x6 = layers.Add()([x6, x5d])
        
        x6 = self.cbam_1(x6)
        x7 = self.convT_2(x6)
        x7 = self.batchNorm_9(x7)
        x7 = layers.Add()([x7, x4d])
        
        x7 = self.cbam_2(x7)
        x8 = self.convT_3(x7)
        x8 = self.batchNorm_10(x8)
        x8 = layers.Add()([x8, x3d])
        
        x8 = self.cbam_3(x8)
        x9 = self.convT_4(x8)
        x9 = self.batchNorm_11(x9)
        x9 = layers.Add()([x9, x2d])
        
        
        x9 = self.cbam_4(x9)
        x10 = self.convT_5(x9)
        x10 = self.batchNorm_12(x10)
        x10 = layers.Add()([x10, x1d])
        
        decoded = self.convT_6(x10)
        return decoded