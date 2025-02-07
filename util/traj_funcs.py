import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import torch


def draw_semicircle(start_coord, radius, steps=196):
    x0, y0, z0 = start_coord
    center_x = x0
    center_z = z0

    # 创建一个从 0 到 pi 的角度数组
    start = 0.2
    end = 0.8
    angles = np.linspace(start * np.pi, end * np.pi, steps)
    #angles = np.linspace(np.pi, 2*np.pi, steps)

    # 计算每个点在半圆上的 x 和 z 坐标
    x = (center_x + radius) - radius * np.cos(angles)
    z = center_z - radius * np.sin(angles)
    y = np.full_like(x, y0)

    return x, y, z


def draw_T_shape(start_coord, bar_length, bar_height, steps=120):
    x0, y0, z0 = start_coord
    # 横杠的两个顶点
    vertex1 = np.array([x0, y0, z0])
    vertex2 = np.array([x0, y0, z0 - bar_length])
    # 竖杠的底部顶点
    vertex3 = np.array([x0, y0 - bar_height, z0 - bar_length / 2])
    
    assert steps % 3 == 0
    steps_per_section = steps // 3
    # 生成T形状的坐标点
    # 横杠
    y_values_horizontal = np.linspace(vertex1[1], vertex2[1], steps_per_section)
    z_values_horizontal = np.linspace(vertex1[2], vertex2[2], steps_per_section)
    # 竖杠
    y_values_vertical = np.linspace(vertex2[1], vertex3[1], steps_per_section * 2)
    z_values_vertical = np.ones_like(y_values_vertical) * (z0 - bar_length / 2)
    
    # 组合坐标
    y_values = np.concatenate([y_values_horizontal, y_values_vertical])
    z_values = np.concatenate([z_values_horizontal, z_values_vertical])
    # x坐标保持不变
    x_values = np.ones_like(y_values) * x0
    
    return x_values, y_values, z_values


def draw_equilateral_triangle(start_coord, side_length, steps=120):
    x0, y0, z0 = start_coord
    # 计算三个顶点
    vertex1 = np.array([x0, y0, z0])
    vertex2 = np.array([x0, y0, z0 - side_length])
    # 根据等边三角形的高度计算第三个顶点
    height = side_length * np.sqrt(3) / 2
    vertex3 = np.array([x0, y0 + height, z0 - side_length / 2])
    assert steps % 3 == 0
    steps = steps // 3
    # 生成三条边的坐标点
    y_values = np.concatenate([
        np.linspace(vertex1[1], vertex2[1], steps),
        np.linspace(vertex2[1], vertex3[1], steps),
        np.linspace(vertex3[1], vertex1[1], steps)
    ])
    z_values = np.concatenate([
        np.linspace(vertex1[2], vertex2[2], steps),
        np.linspace(vertex2[2], vertex3[2], steps),
        np.linspace(vertex3[2], vertex1[2], steps)
    ])
    # x坐标保持不变
    x_values = np.ones_like(z_values) * x0
    
    return x_values, y_values, z_values

def draw_square(start_coord, side_length, steps=120):
    x0, y0, z0 = start_coord
    # 计算四个顶点
    vertex1 = np.array([x0, y0, z0])
    vertex2 = np.array([x0, y0, z0 + side_length])
    vertex3 = np.array([x0, y0 + side_length, z0 + side_length])
    vertex4 = np.array([x0, y0 + side_length, z0])
    assert steps % 4 == 0
    steps_per_side = steps // 4
    # 生成四条边的坐标点
    y_values = np.concatenate([
        np.linspace(vertex1[1], vertex2[1], steps_per_side),
        np.linspace(vertex2[1], vertex3[1], steps_per_side),
        np.linspace(vertex3[1], vertex4[1], steps_per_side),
        np.linspace(vertex4[1], vertex1[1], steps_per_side)
    ])
    z_values = np.concatenate([
        np.linspace(vertex1[2], vertex2[2], steps_per_side),
        np.linspace(vertex2[2], vertex3[2], steps_per_side),
        np.linspace(vertex3[2], vertex4[2], steps_per_side),
        np.linspace(vertex4[2], vertex1[2], steps_per_side)
    ])
    # x坐标保持不变
    x_values = np.ones_like(y_values) * x0
    
    return x_values, y_values, z_values



def rotate(tensor, angle_degrees = 10):
    angle_radians = math.radians(angle_degrees)
    # Define the rotation matrix for a 10 degree rotation around the z-axis
    rotation_matrix = torch.tensor([
    [math.cos(angle_radians), 0, math.sin(angle_radians)],
    [0, 1, 0],
    [-math.sin(angle_radians), 0, math.cos(angle_radians)]])
    rotated_points = torch.matmul(tensor.squeeze(1), rotation_matrix).unsqueeze(1)
    return rotated_points

def draw_up_down_line(start_z = .0, radius=0.1, theta_step = 20,steps=196):
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)  # 调整这里可以控制螺旋的紧密程度
    z_values = start_z + radius * np.sin(theta)
    return z_values

def draw_straight_line(start_coord, step_length, steps=196):
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    y_values = np.ones_like(x_values)* y0
    z_values = np.ones_like(x_values) * z0
    return y_values, x_values, z_values

def draw_curve_line(start_coord, radius, step_length, theta_step = 50, steps=196):
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)  # 调整这里可以控制螺旋的紧密程度
    z_values = z0 + radius * np.sin(theta)
    y_values = np.ones_like(x_values) * y0
    return z_values, y_values, x_values

def draw_curve_line2(start_coord, radius, step_length, theta_step = 50, steps=196):
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    theta = np.linspace(0, 2 * np.pi * steps / theta_step, steps)  # 调整这里可以控制螺旋的紧密程度
    z_values = z0 + radius * np.cos(theta)
    y_values = np.ones_like(x_values) * y0
    return z_values, y_values, x_values


def draw_circle(start_coord, radius, steps=196):
    # start_coord: x-y-z, where y is the height of the scene
    x0, y0, z0 = start_coord
    center_x = x0
    center_z = z0

    # Creating an array for angles from 0 to 2pi
    angles = np.linspace(0, 2 * np.pi, steps)

    # Calculating the x and z coordinates for each point on the circle
    x = (center_x - radius) + radius * np.cos(angles)
    z = center_z + radius * np.sin(angles)
    y = np.full_like(x, y0) 
    return x,y,z

def draw_ellipse(start_coord, a, b, steps=196):
    x0, y0, z0 = start_coord

    # 计算椭圆中心坐标
    center_x = x0 - a
    center_z = z0 + b

    # 创建一个从 0 到 2pi 的角度数组
    angles = np.linspace(0, 2 * np.pi, steps)

    # 计算每个点在椭圆上的 y 和 z 坐标
    x = center_x + a * np.cos(angles)
    z = center_z - b + b * np.sin(angles)
    y = np.full_like(x, y0)

    return x, y, z


def draw_spiral(start_coord, radius, step_length, steps=196):
    x0, y0, z0 = start_coord
    x_values = np.linspace(x0, x0 + step_length * steps, steps)
    theta = np.linspace(0, 2 * np.pi * steps / 20, steps)  # 调整这里可以控制螺旋的紧密程度
    y_values = y0 + radius * np.cos(theta)
    z_values = z0 + radius * np.sin(theta)
    return y_values, x_values, z_values
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot(x_values, y_values, z_values)
    #ax.set_xlabel('X Axis')
    #ax.set_ylabel('Y Axis')
    #ax.set_zlabel('Z Axis')
    #plt.show()# 示例参数
#start_coord = (0, 0, 1)  # 起始坐标
#radius = 0.5               # 螺旋半径
#step_length = 0.05          # X方向的步长draw_spiral(start_coord, radius, step_length)


def generate_u_shape_trajectory(N, hs, x_final, y_final):
    # 计算x, y, z轴的坐标
    y = np.linspace(0, y_final, N)
    
    # 使用调整的cos函数的输出值取幂来使弧度更大
    #x = x_final * (1 - np.cos(2*np.pi * y / y_final) )#**2)
    x = 0.5 * np.sin(2*np.pi * y)
    
    z = hs # - hs[0]
    
    return x, y, z

def plot_trajectory(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

# 测试
#N = 100
#hs = np.linspace(0, 10, N)
#len_param = 3
#x, y, z = generate_u_shape_trajectory(N, hs, len_param, len_param)
#plot_trajectory(x, y, z)