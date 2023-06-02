# knn-upscaler (and denoiser)

A upscaler using the k-nearest-neighbor approach based on the assumption of similar structures in images at different scales.  
It will also remove some noise and compression artifacts.  
Just for fun, makes little sense in times of Deep Learning super resolution.

Here are some example comparisons:

<img src="./examples/comparison2.png" width="800px">  
<img src="./examples/comparison1.png" width="800px">  
lowres - upscaled - original:  
<img src="./examples/butterfly.png" width="1200px">

The upscaling process is very dependent on the parameter,
with the correct ones it can also be used to upscale pixel arts (left is the upscaled):
<img src="./examples/pixelart.png" width="1000px" style="image-rendering: pixelated;">

It also works pretty well on generated images:  
<img src="./examples/flowers.png" width="1000px" style="image-rendering: pixelated;">


## How it works

1. The algorithm scales the image down  
2. Then it iterates over each pixel and looks for a similar image patch (parameter -c for context pixels around the center pixel) in the scaled down version.  
3. For the k best image patches, it takes the corresponding 4 pixels of the center pixel in the original image and calculates their average. These 4 pixels are now the resulting pixels in the upscaled version.  

For this reason, the complexity is O(n²) (n = number of pixels), which is very bad for image upscaling.  
Therefore, the upscaler has some parameters to speed it up, sometimes at the expense of quality. For this, it will not compare patches if the center pixel distance is already over a given threshold (--stop) and the assumption that similar structures are more likely to be found in the same area of the image (--area-size).