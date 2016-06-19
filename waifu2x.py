#coding:utf-8

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU
import json
from PIL import Image
from scipy import misc
import numpy as np

class Waifu2x():
	def __init__(self, model_paths):
		self.params = []
		for path in model_paths:
			with open(path, 'rb') as f:
				self.params.append(json.load(f))

	def _loadModel(self, input_shape, param_id):
		model = Sequential()
		model.add(Convolution2D(
			self.params[param_id][0]['nOutputPlane'], 
			self.params[param_id][0]['kH'], 
			self.params[param_id][0]['kW'], 
			init='zero',
			border_mode='same', 
			weights=[np.array(self.params[param_id][0]['weight']), np.array(self.params[param_id][0]['bias'])], 
			bias=True, 
			input_shape=input_shape))
		model.add(LeakyReLU(0.1))
		for param in self.params[param_id][1:]:
			model.add(Convolution2D(
				param['nOutputPlane'], 
				param['kH'], 
				param['kW'], 
				init='zero',
				border_mode='same', 
				weights=[np.array(param['weight']), np.array(param['bias'])], 
				bias=True))
			model.add(LeakyReLU(0.1))
		return model

	def _loadImageY(self, path, is_noise):
		im = Image.open(path).convert('YCbCr')

		if is_noise:
			im = misc.fromimage(im).astype('float32')
		else:
			im = misc.fromimage(im.resize((2*im.size[0], 2*im.size[1]), resample=Image.NEAREST)).astype('float32')

		x = np.reshape(np.array(im[:,:,0]), (1, 1, im.shape[0], im.shape[1])) / 255.0

		return im, x

	def _loadImageRGB(self, path, is_noise):
		im = Image.open(path)

		if is_noise:
			im = misc.fromimage(im).astype('float32')
		else:
			im = misc.fromimage(im.resize((2*im.size[0], 2*im.size[1]), resample=Image.NEAREST)).astype('float32')

		x = np.array([[im[:,:,0], im[:,:,1], im[:,:,2]]])/255.0

		return im, x
		
	def generate(self, img_path, output, param_id=0, is_noise=False):
		input_channel = self.params[param_id][0]['nInputPlane']

		im = None
		x = None
		# Loading image from img_path.
		if input_channel == 1:
			im, x = self._loadImageY(img_path, is_noise)
		else:
			im, x = self._loadImageRGB(img_path, is_noise)

		# Define model from the image.
		model = self._loadModel((input_channel, im.shape[0], im.shape[1]), param_id)

		# Generate new value.
		y = model.predict(x)

		# Return value to image.
		if input_channel == 1:
			im[:,:,0] = np.clip(y, 0, 1)*255
			misc.toimage(im, mode='YCbCr').convert("RGB").save(output)
		else:
			im = np.clip(y, 0, 1)[0]*255
			misc.toimage(im, mode='RGB').save(output)
