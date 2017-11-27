import codecs
import sys
import tqdm
import numpy as np
from utils import *
from Spacial import *

def get_model(path, detail = False, texture_d = np.array([1.,1.,1.]), texture_a = np.array([1.,1.,1.]), texture_s = np.array([1.,1.,1.]), ratio_s = 0.5, ratio_d = 0.4, ratio_a = 0.1,specular = False):
	"""
	Load model from path
	@ path 			: path to load
	@ detail 		: whether to use complete triangle or use simple triangle
	"""
	file = codecs.open(path, mode = 'r')
	lines = file.readlines()
	vertex = []
	model_list = []
	normal = []
	print("loading model......")
	for oriline in tqdm(lines):
		line = oriline.strip()
		if 'v ' == line[:2]:
			elements = line.split()[1:]
			ver = np.array([float(ele) for ele in elements])
			if len(ver) != 3:
				raise TypeError("obj read error, wrong type v" + oriline)
			vertex.append(ver)
		elif detail and 'vn' == line[:2]:
			elements = line.split()[1:]
			ver = np.array([float(ele) for ele in elements])
			if len(ver) != 3:
				raise TypeError("obj read error, wrong type vn" + oriline)
			normal.append(ver)
		elif 'f' == line[:1]:
			elements = line.split()[1:]
			ver_list = np.array([vertex[int(ele.split('/')[0]) - 1] for ele in elements])
			if len(ver_list) < 3:
				raise TypeError("obj read error, wrong type f" + oriline)
			if detail:
				nver_list = np.array([normal[int(ele.split('/')[2]) - 1] for ele in elements])
				mod = Complete_Polynominal(ver_list,nver_list,texture_s = np.array([texture_s]*3), 
					texture_d = np.array([texture_d]*3), texture_a = np.array([texture_a]*3), ratio_s = ratio_s, ratio_d = ratio_d, ratio_a = ratio_a,specular = specular)
			else:
				mod = Polynominal(ver_list, texture_s = texture_s, texture_d = texture_d, texture_a = texture_a, ratio_s = ratio_s, ratio_d = ratio_d, ratio_a = ratio_a,specular = specular)
			model_list.extend(mod)
	print("loading sucess!")
	vertex = np.array(vertex)
	return model_list, vertex.max(axis = 0), vertex.min(axis = 0)