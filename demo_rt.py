import keras.backend.tensorflow_backend as KTF
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

	config = tf.ConfigProto()  
	config.gpu_options.allow_growth=True
	session = tf.Session(config=config)
	KTF.set_session(session)

	#初始化语义分割模型
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
		models[key] = net #dict
	mnet = models.get('rf_lw50_person')	#choose model

	#初始化风格迁移模型
	custom_objects = {
		'InstanceNormalization':
			InstanceNormalization,
		'DeprocessStylizedImage': layers.DeprocessStylizedImage
	}
	transfer_net = keras.models.load_model(
		'style_transfer/leaf.h5',	#choose model
		custom_objects=custom_objects
	)
	inputs = [transfer_net.input, keras.backend.learning_phase()]
	outputs = [transfer_net.output]
	transfer_style = keras.backend.function(inputs, outputs)

	cap = cv2.VideoCapture(0)
	while True:	#逐帧处理循环
		ret, frame = cap.read()
		frame = frame[60:420:,:,:]

		#语义分割
		seg_inp = torch.tensor(prepare_img(frame).transpose(2, 0, 1)[None]).float()
		if has_cuda:
			seg_inp = seg_inp.cuda()
		seg = mnet(seg_inp)[0].data.cpu().numpy().transpose(1, 2, 0).argmax(axis=2).astype(np.uint8)

		#风格迁移
		style_inp = np.expand_dims(frame, axis=0)
		style = transfer_style([style_inp, 1])[0]
		style = np.uint8(style[0])

		#合并
		seg = cv2.resize(seg, (640, 360),interpolation = cv2.INTER_LANCZOS4)
		mask = np.uint8( np.ones((360, 640, 3)) * np.expand_dims(seg, axis=2) * 255 ) / 255
		masked = np.uint8( (1 - mask) * style + mask * frame )
		window12 = np.hstack((frame, np.uint8(mask * 255) ) )
		window34 = np.hstack((style, masked))
		output = np.vstack((window12, window34))

		#显示
		cv2.imshow("", cv2.resize(output, (1280, 720)) )
		if cv2.waitKey(1) & 0xFF == ord('q'): #Press Q to exit
			break

	cap.release()
	cv2.destroyAllWindows()