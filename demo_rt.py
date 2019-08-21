import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras
import keras_contrib

import torch

import numpy as np
import cv2
import six

from models.resnet import rf_lw50, rf_lw101, rf_lw152
from utils.helpers import prepare_img
from style_transfer import layers
from style_transfer import utils

if __name__ == '__main__':

	#限制Tensorflow
	config = tf.ConfigProto()  
	config.gpu_options.allow_growth=True    #不全部占满显存, 按需分配
	session = tf.Session(config=config)
	KTF.set_session(session)

	#初始化语义分割模型
	has_cuda = torch.cuda.is_available()	#判断有没有配置了cuda
	# cmap = np.load('./utils/cmap.npy')	#cmap是ndarray，256×3，起到将模型输出的矩阵转为RGB矩阵的作用
	n_classes = 7 #7是让模型输出7种语义，不可随意增减。Person是7，VOC是21，NYU是40，Context是60
	model_inits = {	# key / constructor
		'rf_lw50_person'   : rf_lw50,
		'rf_lw101_person'  : rf_lw101,
		'rf_lw152_person'  : rf_lw152,
		}
	models = dict()	#构造dict
	for key,fun in six.iteritems(model_inits):
		net = fun(n_classes, pretrained=True).eval()
		if has_cuda:
			net = net.cuda()	#将网络的权值移动到显卡
		models[key] = net #dict
	mnet = models.get('rf_lw50_person')	#指定使用哪个分割模型

	#初始化风格迁移模型
	custom_objects = {
		'InstanceNormalization':
			keras_contrib.layers.normalization.InstanceNormalization,
		'DeprocessStylizedImage': layers.DeprocessStylizedImage
	}
	transfer_net = keras.models.load_model(
		'example/leaf.h5',	#指定使用哪个迁移模型
		custom_objects=custom_objects
	)
	inputs = [transfer_net.input, keras.backend.learning_phase()]
	outputs = [transfer_net.output]
	transfer_style = keras.backend.function(inputs, outputs)	#transfer_style为模型函数

	cap = cv2.VideoCapture(0)	#初始化摄像头
	while True:	#逐帧处理循环
		ret, frame = cap.read()	#得到帧，是(640, 480, 3)的ndarray
		frame = frame[60:420:,:,:]	#摄像头是宽屏，裁掉上下黑边部分，得到(640, 360, 3)的ndarray

		#下面处理语义分割
		seg_inp = torch.tensor(prepare_img(frame).transpose(2, 0, 1)[None]).float()	#torch.Size([1, 3, 360, 640])
		if has_cuda:
			seg_inp = seg_inp.cuda()	#将图片数据移动到显卡
		seg = mnet(seg_inp)[0].data.cpu().numpy().transpose(1, 2, 0).argmax(axis=2).astype(np.uint8)	#(90, 160, 7) → (90, 160)
		# seg = cmap[seg.argmax(axis=2).astype(np.uint8)]# (90, 160, 7) → (90, 160) → (90, 160, 3)	#RGB化，不需要7种所以忽略

		#下面处理风格迁移
		style_inp = np.expand_dims(frame, axis=0)	#扩展维度到(1, 360, 640, 3)
		style = transfer_style([style_inp, 1])[0]	#风格迁移
		style = np.uint8(style[0])	#降维，转数据类型

		#合并
		seg = cv2.resize(seg, (640, 360),interpolation = cv2.INTER_LANCZOS4)	#将四分之一小的分割输出图放大回原尺寸
		mask = np.uint8( np.ones((360, 640, 3)) * np.expand_dims(seg, axis=2) * 255 ) / 255	#扩展维度并转为二值矩阵
		masked1 = np.uint8( mask * frame )	#蒙版操作
		masked2 = np.uint8( (1 - mask) * style )	#蒙版操作
		window12 = np.hstack((frame, np.uint8(mask * 255) ) )	#将蒙版转为黑白图像，然后跟原图拼接一起，放在上面
		window34 = np.hstack((masked1, masked2))	#拼接下面两张图
		output = np.vstack((window12, window34))	#上下拼接一起

		#显示
		cv2.imshow("", cv2.resize(output, (1280, 720)) )	#显示
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	