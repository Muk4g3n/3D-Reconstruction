import tensorflow as tf
from tensorflow.keras import layers, losses  # ,Sequential,metrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from CBAM import ConvolutionBlockAttentionModule
from tensorflow import reduce_mean, abs
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from CONFIGURATION import CONFIGURATION_ViT as CONFIGURATION

import os
import pickle




def calculate_kl_loss(model):
    def _calculate_kl_loss(*args):
        kld = tf.keras.losses.KLDivergence()
        kl_loss = kld(model.inferenceDistribution,model.learnedDistribution)
        return tf.abs(kl_loss)
    return _calculate_kl_loss

def _calculate_reconstruction_loss(y_target, y_predicted):
    mse = losses.MeanSquaredError()
    reconstruction_loss = mse( y_target, y_predicted )
    return reconstruction_loss


def _calculate_porosity_loss(y_target,y_predicted):
    por1 = tf.reduce_mean(y_target,axis=(1, 2))
    por2 = tf.reduce_mean(y_predicted,axis=(1, 2))
    mae = losses.MeanAbsoluteError()
    return mae(por1,por2)
    


class PatchEncoder(Layer):
  def __init__(self, N_PATCHES, HIDDEN_SIZE):
    super(PatchEncoder, self).__init__(name = 'patch_encoder')

    self.linear_projection = Dense(HIDDEN_SIZE)
    self.positional_embedding = Embedding(N_PATCHES, HIDDEN_SIZE )
    self.N_PATCHES = N_PATCHES

  def call(self, x):
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        rates=[1, 1, 1, 1],
        padding='VALID')
    
    patches = tf.reshape(patches, (tf.shape(patches)[0], 256, patches.shape[-1]))
    
    embedding_input = tf.range(start = 0, limit = self.N_PATCHES, delta = 1 )
    output = self.linear_projection(patches) + self.positional_embedding(embedding_input)
    
    return output


class TransformerEncoder(Layer):
  def __init__(self, N_HEADS, HIDDEN_SIZE):
    super(TransformerEncoder, self).__init__(name = 'transformer_encoder')

    self.layer_norm_1 = LayerNormalization()
    self.layer_norm_2 = LayerNormalization()
    
    self.multi_head_att = MultiHeadAttention(N_HEADS, HIDDEN_SIZE )
    
    self.dense_1 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)
    self.dense_2 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)
    
  def call(self, input):
    x_1 = self.layer_norm_1(input)
    x_1 = self.multi_head_att(x_1, x_1)

    x_1 = Add()([x_1, input])

    x_2 = self.layer_norm_2(x_1)
    x_2 = self.dense_1(x_2)
    output = self.dense_2(x_2)
    output = Add()([output, x_1])

    return output


class ViT(Model):
  def __init__(self, N_HEADS, HIDDEN_SIZE, N_PATCHES, N_LAYERS, N_DENSE_UNITS):
    super(ViT, self).__init__(name = 'vision_transformer')
    self.N_LAYERS = N_LAYERS
    self.patch_encoder = PatchEncoder(N_PATCHES, HIDDEN_SIZE)
    self.trans_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
    self.dense_1 = Dense(N_DENSE_UNITS*2, 'gelu',kernel_initializer='glorot_uniform')
    self.dense_2 = Dense(N_DENSE_UNITS, activation = 'sigmoid',kernel_initializer='glorot_uniform')

  def call(self, inputs, training = True):

    x = self.patch_encoder(inputs)

    for i in range(self.N_LAYERS):
      x = self.trans_encoders[i](x)
    x = Flatten()(x)
    x = self.dense_1(x)
    z = self.dense_2(x)
    return z


class Reconstruction():
    """
    Reconstruction represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self,
                 inputShape=CONFIGURATION["INPUT_SHAPE"],
                 latent_space_dim = CONFIGURATION["LATENT_SPACE_DIM"],
                 reducedDimension = CONFIGURATION["REDUCED_DIMENSION"],
                 num_conv_layers = CONFIGURATION["N_FILTERS"],
                 learning_rate = CONFIGURATION["LEARNING_RATE"],
                 batch_size = CONFIGURATION["BATCH_SIZE"],
                 epochs = CONFIGURATION["N_EPOCHS"],
                 opt = CONFIGURATION["OPTIMIZER"],
                 n_heads =   CONFIGURATION["N_HEADS"], 
                  hidden_size =  CONFIGURATION["HIDDEN_SIZE"],
                  n_patches =  CONFIGURATION["N_PATCHES"],
                  enc_layers =  CONFIGURATION["N_LAYERS"],
                )-> None:
        
        
        
        ##### inputs ######
        self.inputShape = inputShape # [256, 256, 1]
        self.latent_space_dim = latent_space_dim # 64
        self.reducedDimension = reducedDimension #256
        self.num_conv_layers = num_conv_layers #5
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.opt = opt
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.n_patches = n_patches
        self.enc_layers = enc_layers
        self.optimizers = {
            'SGD' : SGD, 
            'RMSprop' : RMSprop,
            'Adagrad' : Adagrad,
            'Adadelta' : Adadelta,
            'Adam' : Adam,
            'Adamax' : Adamax,
            'Nadam' : Nadam,
        }
        self.reshapeDims = self.inputShape[0] // 2**self.num_conv_layers
        self.last_filter = 16 * 2 ** (self.num_conv_layers-1)
        
        
        
        ##### Loss weights ######
        self.reconstruction_loss_weight = 1
        self.Kullback_leibler_weight = 0.001
        self.porosity_Loss_weight = 1
        
        self.skipConnections = []

        self.learnedPrior = None
        self.inference = None
        self.Generate = None
        self.Reconstruction=None

        
        
        self.learnedDistribution = None
        self.inferenceDistribution = None
        
        self._build()

    def summary(self):
        self.learnedPrior.summary()
        self.inference.summary()
        self.Generate.summary()
        self.Reconstruction.summary()
        
    def compile(self):
        optimizer = self.optimizers[self.opt](learning_rate=self.learning_rate)
        self.Reconstruction.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[_calculate_reconstruction_loss,
                                    _calculate_porosity_loss,
                                    calculate_kl_loss(self)],
                             experimental_run_tf_function=False)
    
    def train(self, inputs1,inputs2):
        return self.Reconstruction.fit(x=[inputs1,inputs2],
                       y=inputs2,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       )
    
    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.Reconstruction.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations
    
    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        reconstruction = Reconstruction(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        reconstruction.load_weights(weights_path)
        return reconstruction

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.inputShape,
            self.latent_space_dim,
            self.reducedDimension,
            self.num_conv_layers,
            self.learning_rate,
            self.batch_size,
            self.epochs,
            self.opt,
            self.n_heads,
            self.hidden_size,
            self.n_patches,
            self.enc_layers,
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.Reconstruction.save_weights(save_path)

    def _calculate_combined_loss(self, y_target, y_predicted):
        
        
        reconstruction_loss = _calculate_reconstruction_loss(y_target,y_predicted)
        kl_loss = calculate_kl_loss(self)(self.inferenceDistribution, self.learnedDistribution)
        porisity_loss = _calculate_porosity_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss\
                                                         + kl_loss * self.Kullback_leibler_weight\
                                                         + porisity_loss * self.porosity_Loss_weight
        return combined_loss
    


    def _build(self):
        self._build_inference()
        self._build_learned_prior()
        self._build_generate(self.num_conv_layers)

        self._build_reconstruction()

    ######  inference ########
    
    
        
    def _build_inference(self):
        self.inference = ViT(
        N_HEADS = self.n_heads, HIDDEN_SIZE = self.hidden_size, N_PATCHES = self.n_patches,
        N_LAYERS = self.enc_layers, N_DENSE_UNITS = self.latent_space_dim)
        
        
    ######   learned prior ########
        
    def _build_learned_prior(self):
        
        self.learnedPrior =ViT(
        N_HEADS = self.n_heads, HIDDEN_SIZE = self.hidden_size, N_PATCHES = self.n_patches,
        N_LAYERS = self.enc_layers, N_DENSE_UNITS = self.latent_space_dim)
    
    #### generate ######
    
    def _build_generate(self,num_conv_layers):
        

        
        input_generate = layers.Input(shape=self.inputShape, name="generate_input")
        learnedDist = layers.Input(shape=(1,self.latent_space_dim), name="gen_learned_input")
        
        ### ------------------------------------ Encoder ---------------------------------------------
        
        x = input_generate
        for i in range(num_conv_layers):
            conv = layers.Conv2D(16 * 2 ** i, (3, 3), activation='relu', padding='same', name=f"encoder_conv_{i+1}")(x)
            bn = layers.BatchNormalization()(conv)
            mp = layers.MaxPooling2D((2, 2), padding='same')(bn)
            x = mp
            self.skipConnections.append(bn)

        
        flattened = layers.Flatten()(x)
        latentDense = layers.Dense(self.reducedDimension,name ="latent_dense")(flattened)
        conv_out = layers.BatchNormalization()(latentDense)
        
        conv_out = layers.Reshape(( 1 ,self.reducedDimension ),name="reshape_conv")(conv_out)
#         learnedDist = layers.Reshape(( 1 ,self.latent_space_dim ),name="reshape_learned")(learnedDist)
        
        
        concated_input = layers.Concatenate(axis=-1)([conv_out,learnedDist])
        
        reshaped = layers.Reshape((1, self.reducedDimension + self.latent_space_dim),name="reshape_all")(concated_input)
        
        generated = layers.LSTM(self.reducedDimension, return_sequences=True,name="generate_LSTM")(reshaped)
        
        upDense = layers.Dense(self.reshapeDims*self.reshapeDims*self.last_filter,name="up_dense")(generated)
        batchNorm_7 = layers.BatchNormalization()(upDense)
        
        reshapedUp = layers.Reshape((self.reshapeDims,self.reshapeDims,self.last_filter))(batchNorm_7)
        
        generated = reshapedUp
        for i in reversed(range(num_conv_layers)):
            convT = layers.Conv2DTranspose(16 * 2 ** i, (3, 3), strides=2, activation='relu',
                                           padding='same', name=f"decoder_conv_{i+1}")(generated)
            bn = layers.BatchNormalization()(convT)
            skip_conn = self.skipConnections.pop()
            bn = layers.Add()([bn, skip_conn])
            generated = ConvolutionBlockAttentionModule()(bn)
        
        
        gen_out = layers.Conv2DTranspose(1, (3,3), activation='sigmoid', padding='same')(generated)
        
        self.Generate = Model([input_generate,learnedDist], gen_out, name="generate")
        
        
    def _build_reconstruction(self):
        input_learned = layers.Input(shape=self.inputShape, name="learned_input")
        input_inference = layers.Input(shape=self.inputShape, name="inference_input")



        self.learnedDistribution = self.learnedPrior(input_learned)
        self.learnedDistribution = layers.Reshape(( 1 ,self.latent_space_dim ))(self.learnedDistribution)
        self.inferenceDistribution = self.inference(input_inference)
        gen_out = self.Generate([input_learned,self.learnedDistribution])

        self.Reconstruction = Model([input_learned,input_inference],gen_out,name="reconstuction")