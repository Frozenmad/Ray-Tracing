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
	im = Image.new("RGB",(w,h))
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

def safe_arc_tan(x1,x2):
	"""
	calculate the arc_tan value of x1/x2
	if x2 == 0, it will return pi/2 or -pi/2, depending on the sign of x1
	"""
	if check_value_zero(x1) and check_value_zero(x2):
		raise ValueError('Arc tan input 0 error')
	size = np.sqrt(x1**2 + x2**2)
	x1 = x1 / size
	x2 = x2 / size
	if check_value_zero(x2):
		return x1 * np.pi/2
	return np.arctan(x1/x2)

def auto_camera(camera, maxs, mins, axis = 0, resolution_width = 100, resolution_height = 100, match_size = True):
	"""
	set the camera automatically
	@ camera 			: the camera you want to set
	@ maxs,mins			: the max and min measure of your model
	@ resolution_width 	: the resolution width
	@ resolution_height : the resolution height
	@ match_size 		: whether to auto match the size, if True, the height will be auto craeted
	"""
	position = np.zeros(3)
	for i in range(3):
		if i == axis:
			position[i] =  maxs[i] + maxs[i] * 0.3
		else:
			position[i] =  (maxs[i] + mins[i]) / 2
	camera.setPosition(position)
	camera.setPosDepth(maxs[axis] * 0.1)
	if axis == 0:
		camera.setheight((maxs[2] - mins[2]) * 1.2)
		camera.setwidth((maxs[1] - mins[1]) * 1.2)
		camera.setRotation([[0.,0.,np.pi]])
	elif axis == 1:
		camera.setheight((maxs[2] - mins[2]) * 1.2)
		camera.setwidth((maxs[0] - mins[0]) * 1.2)
		camera.setRotation([[0.,0.,-np.pi/2]])
	else:
		camera.setheight((maxs[0] - mins[0]) * 1.2)
		camera.setwidth((maxs[1] - mins[1]) * 1.2)
		camera.setRotation([[0.,np.pi/2,0.]])
	if match_size:
		resolution_height = int(resolution_width * camera.height / camera.width)
	camera.setResolution_height(resolution_height)
	camera.setResolution_width(resolution_width)
	
	return camera