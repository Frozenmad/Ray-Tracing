from PIL import Image
import numpy as np

mini = 1e-3

def Matrix2Image(data):
	"""
	Transfer h*w*3 matrix to image
	input:
	@data	: an ndarray with shape [h,w,3]
	output:
	@im 	: an PIL image with RGB from input data
	"""
	data = (data * 255)
	h = data.shape[0]
	w = data.shape[1]
	im = Image.new("RGB",(h,w))
	for i in range(h):
		for j in range(w):
			im.putpixel((j,i),(int(data[i][j][0]),int(data[i][j][1]),int(data[i][j][2])))
	return im

def get_max(arr1,arr2):
	"""
	Get the max value from arr1 and arr2
	"""
	arr3 = np.zeros(arr1.shape)
	for i in range(len(arr1)):
		arr3[i] = max(arr1[i],arr2[i])
	return arr3

def get_min(arr1,arr2):
	"""
	Get the min value from arr1 and arr2
	"""
	arr3 = np.zeros(arr1.shape)
	for i in range(len(arr1)):
		arr3[i] = min(arr1[i],arr2[i])
	return arr3

def lenth(vec):
	"""
	Get the lenth of vec
	"""
	return np.sqrt(np.inner(vec,vec))

def normal(vec):
	"""
	Normal the vec
	"""
	if lenth(vec):
		return vec/lenth(vec)
	else:
		return np.array([0.,0.,0.])

def same_point(point1,point2):
	"""
	To judge whether two points are same
	"""
	return lenth(point2-point1) < mini

def check_value_zero(inp):
	"""
	check if input is zero
	"""
	return np.abs(inp) < mini

def check_vec_zero(inp):
	"""
	check if a vector is a zero vector
	"""
	return same_point(inp,np.array([0.,0.,0.]))

def different_side(v_1,v_2,v_3):
	"""
	Check if v_2 and v_3 are on the different side of v_1
	"""
	v_test_1 = np.cross(v_1,v_2)
	v_test_2 = np.cross(v_1,v_3)
	if np.inner(v_test_1,v_test_2) <= 0:
		return True
	return False