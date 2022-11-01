
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from keras.optimizers import Adam
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input

if not os.path.isdir('gen_images/'):
	os.mkdir('gen_images/')

def discriminator_model(optimizer, img_rows, img_cols, channels):

	model = Sequential()
	###YOUR CODE HERE###
	# Architecture & Compile

	return model

def generator_model(optimizer, img_rows, img_cols, channels):

	model = Sequential()
	###YOUR CODE HERE###
	# Architecture & Compile

	return model

def gan_model(img_rows, img_cols, channels):
	
	# Adam Optimizer
	learning_rate = 0.001
	beta = 0.5
	optimizer = Adam(learning_rate, beta)

	# Discriminator & Generator Models
	discriminator = discriminator_model(optimizer, img_rows, img_cols, channels)
	generator = generator_model(optimizer, img_rows, img_cols, channels)

	discriminator.trainable = False

	# GAN Model
	image_input = Input(shape=(100,))
	gan = generator(image_input)
	gan = discriminator(gan)
	gan = Model(image_input, gan)
	gan.compile(loss='binary_crossentropy', optimizer=optimizer)
	
	# Return Models
	return generator, discriminator, gan

def save_imgs(generator, epoch):
	row, column = 5, 5
	noise = np.random.normal(0, 1, (row * column, 100))
	gen_imgs = generator.predict(noise)

	gen_imgs = 0.5 * gen_imgs + 0.5

	fig, axs = plt.subplots(row, column)
	cnt = 0
	for i in range(row):
		for j in range(column):
			axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig("gen_images/epoch_%d.png" % epoch)
	plt.close()

def train(generator, discriminator, gan, epochs, batch_size=128, save_interval=50):
	
	# Load the dataset
	(X_train, _), (_, _) = mnist.load_data()

	# Rescale -1 to 1
	X_train = (X_train.astype(np.float32) - 127.5) / 127.5
	X_train = np.expand_dims(X_train, axis=3)

	half_batch = int(batch_size / 2)

	for epoch in range(epochs):
	    idx = np.random.randint(0, X_train.shape[0], half_batch)
	    imgs = X_train[idx]

	    noise = np.random.normal(0, 1, (half_batch, 100))
	    gen_imgs = generator.predict(noise)

	    # Train the Discriminator
	    discriminator_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
	    discriminator_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
	    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

	    #  Train Generator
	    noise = np.random.normal(0, 1, (batch_size, 100))

	    valid_y = np.array([1] * batch_size)
	    generator_loss = gan.train_on_batch(noise, valid_y)

	    print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, discriminator_loss[0], 100*discriminator_loss[1], generator_loss))

	    if epoch % save_interval == 0:
	        save_imgs(generator, epoch)

if __name__ == "__main__":

	img_rows, img_cols, channels = 28, 28, 1
	
	epochs = 10000
	batch_size = 32
	save_interval = 1000
	
	generator, discriminator, gan = gan_model(img_rows, img_cols, channels)
	
	train(generator, discriminator, gan, epochs+1, batch_size, save_interval)
