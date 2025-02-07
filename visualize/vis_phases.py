import numpy as np
import matplotlib.pyplot as plt

def visulize_phases_from_signal(data):
    x_length = data.shape[1]
    fig_width = 0.1 * x_length
    fig_height = 4  # You can adjust this value if needed
    
    # Set the figure size
    plt.figure(figsize=(fig_width, fig_height))
    
    for i in range(3, data.shape[0], 10):
        if i >= data.shape[0]:
            break
        plt.plot(data[i, :], label=f'Curve {i+1}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().axes.get_yaxis().set_visible(False)
    #plt.legend()
    plt.show()
    
def visulize_phases_from_signal_old(data):
    for i in range(0, data.shape[0], 8):
    #for i in range(1, 128, 8): #8
        if i >= data.shape[0]:
            break
        plt.plot(data[i, :], label=f'Curve {i+1}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    #plt.title('Visualization of Curves')
    #plt.legend()
    plt.show()

    #exit()

def visulize_phases(params, data_idx_chose, num_phases, linspace_start, linspace_end, linspace_num):
    t = np.linspace(linspace_start, linspace_end, linspace_num)
    for i in range(num_phases - 10):
        p, f, a, b = params
        
        p = p[data_idx_chose, i, 0].detach().cpu().numpy()
        f = f[data_idx_chose, i, 0].detach().cpu().numpy()
        a = a[data_idx_chose, i, 0].detach().cpu().numpy()
        b = b[data_idx_chose, i, 0].detach().cpu().numpy()

        # Calculate the function values for each time point
        #y = a * np.sin(f * t + p) + b
        y = a * np.sin(  (t + p) * 2.0 * 3.1415926535898) + b
        #y = a * np.sin(2.0 * 3.1415926535898 * (f * t + p)) + b

        # Plot the graph
        plt.plot(t, y)
    #plt.xlabel('t (Time)')
    #plt.ylabel('y = a*sin(f*t+p)+b')
    #plt.title('Graph of a*sin(f*t+p)+b')
    plt.grid(True)
    plt.show()