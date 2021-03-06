import tensorflow as tf
import keras
import keras_contrib
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import torch

import numpy as np
import cv2
import six

from refinenet.resnet import rf_lw50, rf_lw101, rf_lw152
from refinenet.helpers import prepare_img
from style_transfer import layers
from style_transfer import utils

if __name__ == '__main__':

	has_cuda = torch.cuda.is_available()
	n_classes = 7
	model_inits = {
		'rf_lw50_person'   : rf_lw50,
		'rf_lw101_person'  : rf_lw101,
		'rf_lw152_person'  : rf_lw152,
		}
	models = dict()
	for key,fun in six.iteritems(model_inits):
		net = fun(n_classes, pretrained=True).eval()
		if has_cuda:
			net = net.cuda()
		models[key] = net
	rf = ['rf_lw50_person', 'rf_lw101_person', 'rf_lw152_person']
	mnets = [models.get(rf[0]), models.get(rf[1]), models.get(rf[2])]

	custom_objects = {
		'InstanceNormalization':
			InstanceNormalization,
		'DeprocessStylizedImage': layers.DeprocessStylizedImage
	}
	transfer_net = keras.models.load_model(
		'style_transfer/leaf.h5',
		custom_objects=custom_objects
	)
	inputs = [transfer_net.input, keras.backend.learning_phase()]
	outputs = [transfer_net.output]
	transfer_style = keras.backend.function(inputs, outputs)

	img = cv2.imread('./img/tmp.jpg')
	res = img.shape
	# img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LANCZOS4)
	seg_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
	if has_cuda:
		seg_inp = seg_inp.cuda()
	
	segs = []
	for mnet in mnets:
		seg = mnet(seg_inp)[0].data.cpu().numpy().transpose(1, 2, 0).argmax(axis=2).astype(np.uint8)
		segs.append(seg)

	style_inp = np.expand_dims(img, axis=0)
	style = transfer_style([style_inp, 1])[0]
	style = np.uint8(style[0])

	for i in range(len(segs)):
		seg = segs[i]
		seg = cv2.resize(seg, (res[1], res[0]), interpolation = cv2.INTER_LANCZOS4)
		mask = np.uint8( np.ones(res) * np.expand_dims(seg, axis=2) * 255 ) / 255
		masked = np.uint8( (1 - mask) * style + mask * img )
		window12 = np.hstack((img, np.uint8(mask * 255) ) )
		window34 = np.hstack((style, masked))
		output = np.vstack((window12, window34))
		# output = cv2.resize(output, (res[1], res[0]), interpolation = cv2.INTER_LANCZOS4)
		cv2.imwrite('./img/' + rf[i] + '_result.jpg', output)