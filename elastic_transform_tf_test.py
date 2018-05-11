#%% Test script for the elastic transformation
import numpy as np
import tensorflow as tf
import elastic_transform_tf
import SimpleITK as sitk

ep = elastic_transform_tf.elastic_param()
ep.rotation_x = 0.001
ep.rotation_y = 0.001
ep.rotation_z = 0.05

ep.trans_x = 0.01
ep.trans_y = 0.01
ep.trans_z = 0.01

ep.scale_x = 0.1
ep.scale_y = 0.1
ep.scale_z = 0.1

ep.df_x = 0.1
ep.df_y = 0.1
ep.df_z = 0.1

image_sitk = sitk.ReadImage("/home/jiancong/Desktop/projects/commonCNNTools/LinearInterpolation/test.nii.gz")
image = sitk.GetArrayFromImage(image_sitk)
shape = image.shape
#%%

image_ph = tf.placeholder(tf.float32, shape = [None, None, None, 1])
image_aug, y_t, x_t, z_t = elastic_transform_tf.elastic_transform_3D(image_ph, ep)

sess = tf.Session()

#%%
image_aug_np, y_t_np, x_t_np, z_t_np = sess.run([image_aug, y_t, x_t, z_t], feed_dict = {image_ph: np.reshape(image, (shape[0], shape[1], shape[2], 1)).astype(np.float32)})
#image_aug_np = image_aug_np[0]

out_np = np.squeeze(image_aug_np)
out_sitk = sitk.GetImageFromArray(out_np)

out_sitk.CopyInformation(image_sitk)
sitk.WriteImage(out_sitk, "/home/jiancong/Desktop/projects/commonCNNTools/LinearInterpolation/test_aug2.nii.gz")

#%%
out_np = y_t_np
out_sitk = sitk.GetImageFromArray(out_np)
sitk.WriteImage(out_sitk, "/home/jiancong/Desktop/projects/commonCNNTools/LinearInterpolation/test_y_grid.nii.gz")

out_np = x_t_np
out_sitk = sitk.GetImageFromArray(out_np)
sitk.WriteImage(out_sitk, "/home/jiancong/Desktop/projects/commonCNNTools/LinearInterpolation/test_x_grid.nii.gz")

out_np = z_t_np
out_sitk = sitk.GetImageFromArray(out_np)
sitk.WriteImage(out_sitk, "/home/jiancong/Desktop/projects/commonCNNTools/LinearInterpolation/test_z_grid.nii.gz")


















