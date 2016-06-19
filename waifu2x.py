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

	def _loadImage(self, path, is_noise):
		im = Image.open(path).convert('YCbCr')
		if is_noise:
			im = misc.fromimage(im).astype('float32')
		else:
			im = misc.fromimage(im.resize((2*im.size[0], 2*im.size[1]), resample=Image.NEAREST)).astype('float32')
		x = np.reshape(np.array(im[:,:,0]), (1, 1, im.shape[0], im.shape[1])) / 255.0
		return im, x
		
	def generate(self, img_path, output, param_id=0, is_noise=False):

		# Loading image from img_path.
		im, x = self._loadImage(img_path, is_noise)

		# Define model from the image.
		model = self._loadModel((1, im.shape[0], im.shape[1]), param_id)

		# Generate new value.
		y = model.predict(x)

		# Return value to image.
		im[:,:,0] = np.clip(y, 0, 1)*255

		# Store image.
		misc.toimage(im, mode='YCbCr').convert("RGB").save(output)

