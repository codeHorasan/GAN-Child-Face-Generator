import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from random import random
from numpy import load
from matplotlib import pyplot
from numpy.random import randint
import tensorflow as tf
from deepface import DeepFace
from keras.initializers import RandomNormal
from keras.layers import Conv2D, LeakyReLU, Input, Activation, Concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from keras import Model, load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
layer = InstanceNormalization(axis=-1)

image_dir_fms = "TSKinFace_Data/TSKinFace_cropped/FMS"
image_dir_fmd = "TSKinFace_Data/TSKinFace_cropped/FMD"
image_dir_fmsd = "TSKinFace_Data/TSKinFace_cropped/FMSD"
fms_length = 285
fmd_length = 274
fmsd_length = 228
family_list = []
child_list = []

#FMS
for i in range(1,fms_length+1):
    mother_file = "TSKinFace_Data/TSKinFace_cropped/FMS/FMS-{}-M.jpg".format(str(i))
    father_file = "TSKinFace_Data/TSKinFace_cropped/FMS/FMS-{}-F.jpg".format(str(i))
    son_file = "TSKinFace_Data/TSKinFace_cropped/FMS/FMS-{}-S.jpg".format(str(i))
    mother = np.array(mpimg.imread(mother_file)).astype('float32')
    father = np.array(mpimg.imread(father_file)).astype('float32')
    son = np.array(mpimg.imread(son_file)).astype('float32')

    family = np.concatenate((mother,father), axis=1)
    family = cv2.resize(family, (256,256))
    son = cv2.resize(son, (256,256))

    family_list.append(family)
    child_list.append(son)

#FMD
for i in range(1,fmd_length+1):
    mother_file = "TSKinFace_Data/TSKinFace_cropped/FMD/FMD-{}-M.jpg".format(str(i))
    father_file = "TSKinFace_Data/TSKinFace_cropped/FMD/FMD-{}-F.jpg".format(str(i))
    daughter_file = "TSKinFace_Data/TSKinFace_cropped/FMD/FMD-{}-D.jpg".format(str(i))
    mother = np.array(mpimg.imread(mother_file)).astype('float32')
    father = np.array(mpimg.imread(father_file)).astype('float32')
    daughter = np.array(mpimg.imread(daughter_file)).astype('float32')

    family = np.concatenate((mother,father), axis=1)
    family = cv2.resize(family, (256,256))
    daughter = cv2.resize(daughter, (256,256))

    family_list.append(family)
    child_list.append(daughter)

#FMSD
for i in range(1,fmsd_length+1):
    mother_file = "TSKinFace_Data/TSKinFace_cropped/FMSD/FMSD-{}-M.jpg".format(str(i))
    father_file = "TSKinFace_Data/TSKinFace_cropped/FMSD/FMSD-{}-F.jpg".format(str(i))
    daughter_file = "TSKinFace_Data/TSKinFace_cropped/FMSD/FMSD-{}-D.jpg".format(str(i))
    son_file = "TSKinFace_Data/TSKinFace_cropped/FMSD/FMSD-{}-S.jpg".format(str(i))
    mother = np.array(mpimg.imread(mother_file)).astype('float32')
    father = np.array(mpimg.imread(father_file)).astype('float32')
    daughter = np.array(mpimg.imread(daughter_file)).astype('float32')
    son = np.array(mpimg.imread(son_file)).astype('float32')

    family = np.concatenate((mother,father), axis=1)
    family = cv2.resize(family, (256,256))
    daughter = cv2.resize(daughter, (256,256))
    son = cv2.resize(son, (256,256))

    family_list.append(family)
    child_list.append(daughter)
    family_list.append(family)
    child_list.append(son)

file_list = [i for i in range(1,491)]
for i in file_list:
    family_images = os.listdir("family_dataset/{}".format(i))
    son_list = [s for s in family_images if s.startswith("S")]
    daughter_list = [d for d in family_images if d.startswith("D")]
    try:
        mother =  np.array(mpimg.imread("family_dataset/{}/M.jpg".format(str(i)))).astype('float32')
        father =  np.array(mpimg.imread("family_dataset/{}/F.jpg".format(str(i)))).astype('float32')
        family = np.concatenate((mother,father), axis=1)
        family = cv2.resize(family, (256,256))
        #flipped_family = np.fliplr(family)
        if son_list:
            for s in son_list:
                son = np.array(mpimg.imread("family_dataset/{}/{}".format(str(i),s))).astype('float32')
                family_list.append(family)
                child_list.append(son)
                """#horizontal flip
                family_list.append(flipped_family)
                child_list.append(np.fliplr(son))"""
        if daughter_list:
            for d in daughter_list:
                daughter = np.array(mpimg.imread("family_dataset/{}/{}".format(str(i),d))).astype('float32')
                family_list.append(family)
                child_list.append(daughter)
                """#horizontal flip
                family_list.append(flipped_family)
                child_list.append(np.fliplr(daughter))"""
    except Exception as e:
        print("hata:",i, "  ", e)

from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
 
def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)
 
filename = 'family2child_256.npz'
savez_compressed(filename, family_list, child_list)
print('Saved dataset: ', filename)


def define_discriminator(image_shape):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # source image input
  in_image = Input(shape=image_shape)
  # C64
  d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  # C128
  d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = InstanceNormalization(axis=-1)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # C256
  d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = InstanceNormalization(axis=-1)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # C512
  d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = InstanceNormalization(axis=-1)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # second last output layer
  d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
  d = InstanceNormalization(axis=-1)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # patch output
  patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
  #print("patch out shape (Discriminator):", patch_out.shape)
  # define model
  model = Model(in_image, patch_out)
  # compile model
  model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
  return model

def resnet_block(n_filters, input_layer):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # first layer convolutional layer
  g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # second convolutional layer
  g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  # concatenate merge channel-wise with input layer
  g = Concatenate()([g, input_layer])
  #print("concatenated shape: (resnet block)", g.shape)
  return g

def define_generator(image_shape, n_resnet=9):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # image input
  in_image = Input(shape=image_shape)
  # c7s1-64
  g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # d128
  g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # d256
  g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # R256
  for _ in range(n_resnet):
    g = resnet_block(256, g)
  # u128
  g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # u64
  g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  #print("before last : (generator)", g.shape)
  # c7s1-3
  g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  out_image = Activation('tanh')(g)
  #print("last shape: (generator)", g.shape)
  # define model
  model = Model(in_image, out_image)
  return model

def custom_loss(y_true, y_pred):
  return DeepFace.verify(y_true, y_pred, enforce_detection=False, model_name="VGG-Face")["distance"]

def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
  # ensure the model we're updating is trainable
  g_model_1.trainable = True
  # mark discriminator as not trainable
  d_model.trainable = False
  # mark other generator model as not trainable
  g_model_2.trainable = False
  # discriminator element
  input_gen = Input(shape=image_shape)
  gen1_out = g_model_1(input_gen)
  output_d = d_model(gen1_out)
  # identity element
  input_id = Input(shape=image_shape)
  output_id = g_model_1(input_id)
  # forward cycle
  output_f = g_model_2(gen1_out)
  # backward cycle
  gen2_out = g_model_2(input_id)
  output_b = g_model_1(gen2_out)
  # define model graph
  model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
  # define optimization algorithm configuration
  opt = Adam(lr=0.0002, beta_1=0.5)
  # compile model with weighting of least squares loss and L1 loss
  model.compile(loss=['mse', 'mae', 'mae', 'mae', custom_loss], loss_weights=[1, 5, 10, 10, 5], optimizer=opt)
  return model

def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

def summarize_performance(step, g_model, trainX, name, n_samples=3):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot real images
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()

def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return np.asarray(selected)

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	# define properties of the training run
	n_epochs, n_batch, = 100, 1
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# prepare image pool for fakes
	poolA, poolB = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# update fakes from pool
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
		# update generator B->A via adversarial and cycle loss
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# update discriminator for A -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# update generator A->B via adversarial and cycle loss
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# update discriminator for B -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# summarize performance
		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		# evaluate the model performance every so often
		if (i+1) % (bat_per_epo * 1) == 0:
			# plot A->B translation
			summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
			# plot B->A translation
			summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (bat_per_epo * 5) == 0:
			# save the models
			save_models(i, g_model_AtoB, g_model_BtoA)

dataset = load_real_samples('family2child_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

from keras.models import load_model

def plot_images(src_img, gen_img, tar_img):
  images = np.vstack((src_img, gen_img, tar_img))
  # scale from [-1,1] to [0,1]
  images = (images + 1) / 2.0
  titles = ['Family', 'Generated', 'Child']
  for i in range(len(images)):
    plt.subplot(1, 3, 1 + i)
    plt.axis('off')
    plt.imshow(images[i])
    plt.title(titles[i])
  plt.show()

[X1, X2] = load_real_samples('family2child_256.npz')
print('Loaded', X1.shape, X2.shape)

model = load_model('g_model_AtoB_003935.h5', custom_objects={"InstanceNormalization": layer})

ix = np.random.randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
gen_image = model.predict(src_image)
plot_images(src_image, gen_image, tar_image)

plt.imshow(NormalizeData(gen_image[0]))#.astype('uint8'))
plt.show()