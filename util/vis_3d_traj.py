import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Assuming array "data" is your data array of shape (61, 22, 3)

def plot_3d_static(data):
    # data.shape = (N_frame, P_points, xyz_3d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in range(data.shape[0]):
        xs = data[frame, :, 0]
        ys = data[frame, :, 1]
        zs = data[frame, :, 2]
        ax.scatter(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# Call the function with your array
#plot_3d(data)  # Replace data with your actual array
lines_connect = [(0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (9, 14), (14, 17), (17, 19), (19, 21),
                 (9, 13), (13, 16), (16, 18), (18, 20), (2, 5), (5, 8), (8, 11), (1, 4), (4, 7), (7, 10)]  # This list will store the line objects

def is_joint_predict(masked_joint_list, jointID, frame):
    for frameInfo in masked_joint_list:
        if jointID == frameInfo[2] and frame > frameInfo[0] and frame < frameInfo[1]:
            return True
    return False

colorSave = [ [[], []] for x in range(6)]
color_choice = [('DarkRed', "Pink"), ('DarkRed', "LightGreen"), ('DarkRed', "yellow"), ('blue', "DimGray"), ('ForestGreen', "green"), ('Gold', "yellow")]

def update(num, data, ax, highLight_data=None, masked_joint_list = None):
    global lines_connect
    
    ax.clear()
    ax.scatter(data[num, :, 0], data[num, :, 1], data[num, :, 2], s=10)

    # Draw line segments and store the line objects
    for joint1, joint2 in lines_connect:
        ax.plot(data[num, [joint1, joint2], 0], data[num, [joint1, joint2], 1], data[num, [joint1, joint2], 2], color='blue')

    if highLight_data is not None:
        for i in range(6):
            if is_joint_predict(masked_joint_list, i, num):
                colorSave[i][0].append(num)
            else:
                colorSave[i][1].append(num)
            ax.scatter(highLight_data[colorSave[i][0], i, 0], highLight_data[colorSave[i][0], i, 1], highLight_data[colorSave[i][0], i, 2], color=color_choice[i][0], s=30)
            ax.scatter(highLight_data[colorSave[i][1], i, 0], highLight_data[colorSave[i][1], i, 1], highLight_data[colorSave[i][1], i, 2], color=color_choice[i][1], s=30)
            
            # Clear colorSave for future frames
            colorSave[i][0] = [n for n in colorSave[i][0] if n <= num]
            colorSave[i][1] = [n for n in colorSave[i][1] if n <= num]

    min_v = min(data[:, :, 0].min(), data[:, :, 1].min())
    max_v = max(data[:, :, 0].max(), data[:, :, 1].max())
    ax.set_xlim(min_v * 0.5, max_v * 0.5)
    ax.set_ylim(min_v * 0.5, max_v * 0.5)
    #ax.set_xlim(-1.0, 1.0)
    #ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(data[:, :, 2].min(), data[:, :, 2].max())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(False)
    ax.axis(False)
    
    x = np.array([min_v, max_v])
    y = np.array([min_v, max_v])
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * np.min(data[0, :, 2])
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.5)


def animate_3d(data, highLight_data=None, masked_joint_list = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = animation.FuncAnimation(fig, update, frames=range(data.shape[0]), fargs=(data, ax, highLight_data, masked_joint_list), repeat=False)

    
    
    ani.save('animation.mp4', writer='ffmpeg')

    plt.show()