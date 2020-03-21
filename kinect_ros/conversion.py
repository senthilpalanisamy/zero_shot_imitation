# coding: utf-8
import tf
import numpy as np

quarter_c_a = [-0.110, -0.118, 0.614, 0.772]
t_c_a = [0.956, 0.165, -0.252]
quater_b_a = [0.031, -0.293, 0.785, 0.545]
t_b_a = [1.279, 0.223, -0.163]

T_c_a = tf.transformations.quaternion_matrix(quarter_c_a)
T_c_a[:3, 3] = t_c_a
T_b_a = tf.transformations.quaternion_matrix(quater_b_a)
quater_b_a = [0.013, 0.001, 0.726, 0.687]
T_b_a = tf.transformations.quaternion_matrix(quater_b_a)
T_b_a[:3, 3] = t_b_a
T_b_c = T_b_a.dot(np.linalg.inv(T_c_a))
