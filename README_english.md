# Ray-Tracing Realization
_2017 fall SJTU CG course assignment2_
## Copyright
author : Chaoyu Guan

E-mail : frozenmad2015@outlook.com

License : [MIT License](LICENSE)

> ATTENTION : Chaoyu Guan is the original author. Please give credit to the original author when you use it elsewhere.

## Related files
##### `main.py`
The main file we running on, you can modify it or create a new one.
##### `Spacial.py`
Realization of space, geometries, and lights.
##### `utils.py`
Some useful tools used in this project.
##### `load_model.py`
Allow you to load model from .obj files.
## Environment dependencies
* *python 2.7 or 3.x*
* *numpy*
* *PIL*
* *tqdm*

You can get these libs by simply entering the code bellow to you command line:
```
pip install --upgrade Pillow numpy tqdm
```
## Simplified Usage
### Step 0 Prepare the environment
Import the libraries which we need to use.
```
from PIL import image
import numpy as np
from Spacial import *
from utils import *
# If you need to load model from .obj files, you need the following lib as well
# from load_model.py import *
```
Let's make our code more concise by renaming the np.array:
```
arr = np.array
```
Then, we can get start!
### Step 1 Make a new space
First of all, you need to create a space for placing your geometries and rendering:
```
my_space = Spacial()
```
### Step 2 Create and add a camera
Every space has a default camera, which is located on `(50,0,0)`，the orientation is `(-1,0,0)`，up vector is `(0,0,1)`，the distance of foreground is 20，and the background is infinite

If you need to set the parameter of your camera, you can create a new one and set it to your sapce as follow:
```
my_camera = Camera(position = arr([50,0,10]), orientation = arr([[np.pi/2,0,np.pi]]))
my_space.setCamera(my_camera)
```
For more info of parameter of class `Camera`, please refer to `Spacial.py`
### Step 2 Create and add geometries
Then, you can add some geometries to your space. For example:
```
my_sphere = Circle(arr([0,0,0]),2))
my_quad = Polynominal([arr([-10,10,-10]),arr([-10,10,10]),arr([-10,-10,10]),arr([-10,-10,-10])])
```
The supported geometries are:
###### Primal geometries
* Sphere  `Circle()`
* Simple triangle  `Triangle()`
* Triangle `CompleteTriangle()`
* Plane `Plane()`

###### Combine geometries
* Simple polynominal  `Polynominal()`
* Polynominal `Complete_Polynominal()`
* Cube `Cube()`
* Load from .obj file `get_model(path)`  # This need to import load_model.py

For the parameters of these geometries, please refer to `Spacial.py`.

> ATTTENTION : The primal geometries should be added into your space by `my_space.AddObject()`, and the combine geometries should use `my_space.AddObjects()`.

Then, you need to add your objects to your space:
```
my_space.AddObject(my_shpere)
my_space.AddObjects(my_quad)
```
### Step 4 Create and add Light (Optional)
This step will guide you to add light. Of course you can render without light, just rendering the ambient light of the objects. But it is still recommended to add the light.

First you need to create a light:
```
Light = PointLight(arr([0.,0.,40.]))
```
The supported light are as follow:
* Point light `PointLight()`
* Parallel light `Parallel_light()`

Please refer to `Spacial.py` for more info of parameters.

Then, you need to add the light to your space:
```
my_space.AddLight(Light)
```
You can also use `my_space.AddLights()` to add several lights together.

### Step 5 Render
Use `my_space.Render()` to render, This function returns a float RGB matrix
```
im_matrix = my_space.Render(max_iter = 3)
```

### Step 6 Show and save picture
You can use the function `Matrix2Image()` from `utils.py` to convert a float RGB matrix to a picture of PIL, then you can show and save it by using PIL lib.
```
im = Matrix2Image(im_matrix)
im.show()
im.save('example.png')
```
## Something to know
Because I use python to implement ray-tracing, the computation speed is low. This is just for the new learners to better learn the ray-tracing algorithm.

Tests shows that, given one sphere, two simple triangles and one light, the resolution is 200\*200, the cost is about 30s.

**AGAIN : Please give credit to the original author when you use it elsewhere**
