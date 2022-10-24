import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from deepface import DeepFace


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
  """#horizontal flip
  family_list.append(np.fliplr(family))
  child_list.append(np.fliplr(son))"""

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
  """#horizontal flip
  family_list.append(np.fliplr(family))
  child_list.append(np.fliplr(daughter))"""

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
  """#horizontal flip
  family_list.append(np.fliplr(family))
  child_list.append(np.fliplr(daughter))"""
  """#horizontal flip
  family_list.append(np.fliplr(family))
  child_list.append(np.fliplr(son))"""

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

# helper display function
def imshow(img):
    plt.imshow(img/255.0)

def randomly_display_images(family_list, child_list):
    fig = plt.figure(figsize=(20, 4))
    plot_size=20
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
        if idx < 10:
            imshow(child_list[idx])
        else:
            imshow(family_list[idx])

np.savez_compressed('data.npz', family_list, child_list)
data = np.load('data.npz')
source_list, target_list = data['arr_0'], data['arr_1']

def custom_loss(y_true, y_pred):
  return DeepFace.verify(y_true, y_pred, enforce_detection=False, model_name="VGG-Face")["distance"]

def define_discriminator_64x64(image_shape=(64,64,3)):
	init = RandomNormal(stddev=0.02)
	in_source_image = Input(shape=image_shape)
	in_target_image = Input(shape=image_shape)
	merged = Concatenate()([in_source_image, in_target_image])
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	model = Model([in_source_image, in_target_image], patch_out)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.8])
	return model


def define_discriminator(image_shape):
	init = RandomNormal(stddev=0.02)
	in_src_image = Input(shape=image_shape)
	in_target_image = Input(shape=image_shape)
	merged = Concatenate()([in_src_image, in_target_image])
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	model = Model([in_src_image, in_target_image], patch_out)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model


def encoder_block(layer_in, n_filters, batchnorm=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	g = LeakyReLU(alpha=0.2)(g)
	return g
 
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Concatenate()([g, skip_in])
	g = Activation('relu')(g)
	return g

def define_generator_64(image_shape=(64,64,3)):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	e1 = encoder_block(in_image, 64, batchnorm=False)
	e2 = encoder_block(e1, 128)
	e3 = encoder_block(e2, 256)
	e4 = encoder_block(e3, 512)
	e5 = encoder_block(e4, 512)
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
	b = Activation('relu')(b)
	d1 = decoder_block(b, e5, 512)
	d2 = decoder_block(d1, e4, 512, dropout=False)
	d3 = decoder_block(d2, e3, 256, dropout=False)
	d4 = decoder_block(d3, e2, 128, dropout=False)
	d5 = decoder_block(d4, e1, 64, dropout=False)
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
	out_image = Activation('tanh')(g)
	model = Model(in_image, out_image)
	return model

def define_generator(image_shape=(256,256,3)):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	e1 = encoder_block(in_image, 64, batchnorm=False)
	e2 = encoder_block(e1, 128)
	e3 = encoder_block(e2, 256)
	e4 = encoder_block(e3, 512)
	e5 = encoder_block(e4, 512)
	e6 = encoder_block(e5, 512)
	e7 = encoder_block(e6, 512)
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	model = Model(in_image, out_image)
	return model

def define_gan_64(g_model, d_model, image_shape):
	in_source = Input(shape=image_shape)
	d_model.trainable = False
	gen_out = g_model(in_source)
	dis_out = d_model([in_source, gen_out])
	model = Model(in_source, [dis_out, gen_out])
	opt = Adam(lr=0.005, beta_1=0.7)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[10,100])
	return model

def define_gan(g_model, d_model, image_shape):
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	in_src = Input(shape=image_shape)
	gen_out = g_model(in_src)
	dis_out = d_model([in_src, gen_out])
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae', custom_loss], optimizer=opt, loss_weights=[1,100,50])
	return model


def load_real_samples(filename):
  data = np.load(filename)
  X1, X2 = data['arr_0'], data['arr_1']
  # scale from [0,255] to [-1,1]
  X1 = (X1 - 127.5) / 127.5
  X2 = (X2 - 127.5) / 127.5
  return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
  trainA, trainB = dataset
  ix = np.random.randint(0, trainA.shape[0], n_samples)
  X1, X2 = trainA[ix], trainB[ix]
  y = np.ones((n_samples, patch_shape, patch_shape, 1))
  return [X1,X2], y

def generate_fake_samples(g_model, samples, patch_shape):
  X = g_model.predict(samples)
  y = np.zeros((len(X), patch_shape, patch_shape, 1))
  return X, y

np.savez_compressed('data.npz', family_list, child_list)
data = np.load('data.npz')
dataset = load_real_samples('data.npz')
g_model = define_generator()
[X1real,X2real], y_real  = generate_real_samples(dataset, 2, g_model.output_shape[1])

def summarize_performance(step, g_model, dataset, n_samples=3):
  [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
  X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
  X_realA = (X_realA + 1) / 2.0
  X_realB = (X_realB + 1) / 2.0
  X_fakeB = (X_fakeB + 1) / 2.0
  for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(X_realA[i])
  for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(X_fakeB[i])
  for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + n_samples*2 + i)
    plt.axis('off')
    plt.imshow(X_realB[i])
  filename1 = 'plot_%06d.png' % (step+1)
  plt.savefig(filename1)
  plt.close()
  filename2 = 'model_%06d.h5' % (step+1)
  g_model.save(filename2)
  print('>Saved: %s and %s' % (filename1, filename2))

def plot_images(src_img, gen_img, tar_img):
  images = np.vstack((src_img, gen_img, tar_img))
  images = (images + 1) / 2.0
  titles = ['Source', 'Generated', 'Expected']
  for i in range(len(images)):
    plt.subplot(1, 3, 1 + i)
    plt.axis('off')
    plt.imshow(images[i])
    plt.title(titles[i])
  plt.show()

def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=32):
  n_patch = d_model.output_shape[1]
  trainA, _ = dataset
  bat_per_epo = int(len(trainA) / n_batch)
  n_steps = bat_per_epo * n_epochs
  for i in range(n_steps):
    [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
    d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
    X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
    d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
    g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
    if (i+1) % (bat_per_epo * 10) == 0:
      print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
      summarize_performance(i, g_model, dataset)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

image_shape = dataset[0].shape[1:]

d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)

train(d_model, g_model, gan_model, dataset, n_epochs=1, n_batch=1)

def inference_and_plot(img1_path, img2_path):
    m = np.array(Image.open(img1_path))
    m = cv2.resize(m, (256,256))
    m = (m - 127.5) / 127.5
    f = np.array(Image.open(img2_path))
    f = cv2.resize(f, (256,256))
    f = (f - 127.5) / 127.5
    family = np.concatenate((f,m), axis=1)
    family = cv2.resize(family, (256,256))
    """backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    dad_face = DeepFace.detectFace(img_path = "F.jpg", target_size = (64, 64), detector_backend = backends[2])
    mom_face = DeepFace.detectFace(img_path = "M.jpg", target_size = (64, 64), detector_backend = backends[2])"""
    plt.imshow(family);
