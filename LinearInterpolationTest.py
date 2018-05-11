#%% Test script for the tri-linear interpolation
import SimpleITK as sitk
import tensorflow as tf
import LinearInterpolation 
import numpy as np
#%%
image = sitk.ReadImage("/home/jiancong/Desktop/projects/commonCNNTools/LinearInterpolation/test.nii.gz")
image = sitk.GetArrayFromImage(image)
shape = image.shape
#%%

image_ph = tf.placeholder(tf.float32, shape = [None, None, None, None, 1])

shape_ph = tf.shape(image_ph)
batch_size = shape_ph[0]
height = shape_ph[1]
width = shape_ph[2]
depth = shape_ph[3]

#y = tf.linspace(0.0, tf.cast(height, tf.float32), height)
#x = tf.linspace(0.0, tf.cast(width, tf.float32), width)
#z = tf.linspace(0.0, tf.cast(depth, tf.float32), depth)
y = tf.linspace(0.0, tf.cast(height, tf.float32) - 1, 2 * height)
x = tf.linspace(0.0, tf.cast(width, tf.float32) - 1, 2 * width)
z = tf.linspace(0.0, tf.cast(depth, tf.float32) - 1, 2 * depth)
x_t, y_t, z_t = tf.meshgrid(x, y, z)

x_t = tf.expand_dims(x_t, axis=0)
y_t = tf.expand_dims(y_t, axis=0)
z_t = tf.expand_dims(z_t, axis=0)

#
#x_t = tf.cast(tf.floor(x_t), tf.int32)
#y_t = tf.cast(tf.floor(y_t), tf.int32)
#z_t = tf.cast(tf.floor(z_t), tf.int32)
#
#zero = tf.zeros([], dtype=tf.int32)
#x_t = tf.clip_by_value(x_t, zero, width - 1)
#y_t = tf.clip_by_value(y_t, zero, height - 1)
#z_t = tf.clip_by_value(z_t, zero, depth - 1)

#%% Test the gather function
img_gather = LinearInterpolation.get_pixel_value_3D(image_ph, x_t, y_t, z_t)


sess = tf.Session()
img_gather_np = sess.run([img_gather], feed_dict = {image_ph: np.reshape(image, (1, shape[0], shape[1], shape[2], 1)).astype(np.float32)})
img_gather_np = img_gather_np[0]

#%%
np.max(img_gather_np)
np.min(img_gather_np)
#%%

out = LinearInterpolation.trilinear_sampler(image_ph, x_t, y_t, z_t, normalized_coordinate = False)


#%%
sess = tf.Session()
out_np, x_t_np = sess.run([out, x_t], feed_dict = {image_ph: np.reshape(image, (1, shape[0], shape[1], shape[2], 1)).astype(np.float32)})

out_np = out_np[0]

#%%
#out_np =  img_gather_np
out_np = np.squeeze(out_np)
out_sitk = sitk.GetImageFromArray(out_np)
sitk.WriteImage(out_sitk, "/home/jiancong/Desktop/projects/commonCNNTools/LinearInterpolation/test_double.nii.gz")

#%%
np.max(out_np)
np.min(out_np)