from PIL import Image
import numpy as np
from utils import *
from Spacial import *
import time


path_to_save = 'my_space.png'
"""
test = np.zeros([100,100,3])

for i in range(100):
	for j in range(100):
		test[i][j][0] = (j+100-i)/200
		test[i][j][1] = (j+100-i)/200
		test[i][j][2] = (j+100-i)/200

im = Matrix2Image(test)

im.show()
"""

arr = np.array

my_space = Spacial()

my_circle = Circle(np.array([0,0,0]),2,np.array([1.,1.,1.]),np.array([1.,0.,0.]),np.array([1.,0.,0.]))

my_circle2 = Circle(np.array([0,0,6]),3,np.array([1.,1.,1.]),np.array([0.,1.,0.]),np.array([0.,1.,0.]))

my_Triangle = Polynominal([arr([-10,10,-10]),arr([-10,10,10]),arr([-10,-10,10]),arr([-10,-10,-10])],np.array([1.,1.,1.]),np.array([1.,1.,1.]),np.array([1.,1.,1.]))

my_space.AddObject(my_circle)

Light_1 = PointLight(arr([0.,0.,40.]))
Light_2 = Parallel_light(orientation=arr([-1.,0.,-1.]))

my_space.AddLight(Light_2)

my_space.AddObject(my_circle2)

my_space.AddObjects(my_Triangle)

before = time.time()
render = my_space.Render(orth = True, resolution_width = 200, resolution_height = 200, width = 20, height = 20)
after = time.time()

print("Total time used to render: {}".format(after-before))

im = Matrix2Image(render)

im.show()

im.save(path_to_save)

#my_space.render_test(orth = True, y = 0, z = 9.5)