import numpy as np
import tkinter as tk
from scipy.interpolate import interp1d

def interpolate_line(points, num_samples):
    if len(points) < 2:
        return np.array(points)
    distances = np.sqrt(np.diff([p[0] for p in points])**2 + np.diff([p[1] for p in points])**2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    sample_distances = np.linspace(0, cumulative_distances[-1], num_samples)
    sample_x = np.interp(sample_distances, cumulative_distances, [p[0] for p in points])
    sample_y = np.interp(sample_distances, cumulative_distances, [p[1] for p in points])
    return np.vstack((sample_x, sample_y)).T

class LineSamplerApp:
    def __init__(self, root, num_samples=30):
        self.root = root
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack()
        self.drawing_points = []
        self.lines = []
        self.current_line = []
        self.height_control_points = [(50, 350), (175, 350), (225, 350), (350, 350)]
        self.dragging_point_index = None
        self.drawing = True
        self.num_samples = num_samples

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        adjust_height_button = tk.Button(root, text="Adjust Height", command=self.enter_height_adjust_mode)
        adjust_height_button.pack()
        
        save_button = tk.Button(root, text="Save 3D Points", command=self.save_points)
        save_button.pack()

        self.draw_height_control_points()

    def on_click(self, event):
        if self.drawing:
            self.current_line.append((event.x, event.y))
        else:
            self.on_control_point_click(event)

    def on_drag(self, event):
        if self.drawing and self.current_line:
            self.current_line.append((event.x, event.y))
            self.canvas.create_line(self.current_line[-2], self.current_line[-1], fill="black")
        elif not self.drawing and self.dragging_point_index is not None:
            self.on_control_point_drag(event)

    def on_release(self, event):
        if self.drawing and self.current_line:
            self.lines.append(self.current_line)
            self.current_line = []
        elif not self.drawing:
            self.on_control_point_release(event)

    def enter_height_adjust_mode(self):
        self.drawing = False
        # 绑定新的事件处理器以处理控制点的点击和拖拽
        self.canvas.bind("<Button-1>", self.on_control_point_click)
        self.canvas.bind("<B1-Motion>", self.on_control_point_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_control_point_release)

    def on_control_point_click(self, event):
        for i, point in enumerate(self.height_control_points):
            if (point[0] - 5 <= event.x <= point[0] + 5) and (point[1] - 5 <= event.y <= point[1] + 5):
                self.dragging_point_index = i
                return

    def on_control_point_drag(self, event):
        if self.dragging_point_index is not None:
            self.height_control_points[self.dragging_point_index] = (event.x, event.y)
            self.draw_height_control_points()

    def on_control_point_release(self, event):
        self.dragging_point_index = None

    def draw_height_control_points(self):
        self.canvas.delete("height_control")
        for point in self.height_control_points:
            self.canvas.create_oval(point[0]-5, point[1]-5, point[0]+5, point[1]+5, fill="red", tags="height_control")
        self.redraw_height_curve()

    def redraw_height_curve(self):
        self.canvas.delete("height_curve")
        if len(self.height_control_points) >= 4:  # Use cubic spline interpolation if there are enough points
            x, y = zip(*self.height_control_points)
            curve = interp1d(x, y, kind='cubic', fill_value="extrapolate")
            xs = np.linspace(x[0], x[-1], 100)
            ys = curve(xs)
            for i in range(len(xs) - 1):
                self.canvas.create_line(xs[i], ys[i], xs[i+1], ys[i+1], fill="red", tags="height_curve")

    def save_points(self):
        all_samples = []
        for line in self.lines:
            sampled_line = interpolate_line(line, self.num_samples // len(self.lines))
            #print(sampled_line)
            all_samples.extend(sampled_line)
            all_samples.extend([(-1, -1)] * 5)  # 每个笔画之间插入分隔符
        #exit()

        if all_samples[-5:] == [(-1, -1)] * 5:
            all_samples = all_samples[:-5]

        
        if len(self.height_control_points) >= 2:
            x, y = zip(*self.height_control_points)
            curve = interp1d(x, y, kind='linear', fill_value="extrapolate")
            
            xs = [point[0] for point in all_samples if point[0] != -1 or point[1] != -1]
            zs = curve(xs)
        else:
            
            zs = np.zeros(len([point for point in all_samples if point != (-1, -1)]))

        
        samples_3d = []
        z_index = 0
        for point in all_samples:
            print(point)
            if point[0] == -1 and point[1] == -1:
                samples_3d.append([-1, -1, -1])
            else:
                samples_3d.append([point[0], point[1], zs[z_index]])
                #samples_3d.append([point[0], point[1], .0])
                z_index += 1

        np.save("new_draw_3d.npy", np.array(samples_3d))
        print("3D points saved.")


root = tk.Tk()
app = LineSamplerApp(root)
root.mainloop()



