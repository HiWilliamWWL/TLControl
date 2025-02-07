import os
import time
import open3d as o3d
import numpy as np

def o3d_plot(o3d_items: list, show_coord=True, **kwargs):
    if show_coord:
        _items = o3d_items + [o3d_coord(**kwargs)]
    else:
        _items = o3d_items
    view = o3d.visualization.VisualizerWithKeyCallback()
    view.create_window()
    for item in _items:
        view.add_geometry(item)
    view.run()
    
    
def o3d_pcl(pcl: np.ndarray = None, color: list = None, colors: list = None, last_update=None):
    _pcl = last_update
    if _pcl is None:
        _pcl = o3d.geometry.PointCloud()

    if pcl is not None and pcl.size != 0:
        if pcl.shape[0] > 1000000:
            # auto downsample
            pcl = pcl[np.random.choice(
                np.arange(pcl.shape[0]), size=1000000, replace=False)]
        _pcl.points = o3d.utility.Vector3dVector(pcl)
        if color is not None:
            _pcl.paint_uniform_color(color)
        if colors is not None:
            _pcl.colors = o3d.utility.Vector3dVector(colors)
    return _pcl


def o3d_mesh(mesh=None, color: list = None,
             last_update=None):
    _mesh = last_update
    if _mesh is None:
        _mesh = o3d.geometry.TriangleMesh()

    if mesh is not None:
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            _mesh.vertices = mesh.vertices
            _mesh.triangles = mesh.triangles
        else:
            _mesh.vertices = o3d.utility.Vector3dVector(mesh[0])
            if mesh[1] is not None:
                _mesh.triangles = o3d.utility.Vector3iVector(mesh[1])
        if color is not None:
            _mesh.paint_uniform_color(color)
        _mesh.compute_vertex_normals()
    return _mesh


def o3d_lines(pcl: np.ndarray = None, lines: np.ndarray = None,
              color: list = [1, 0, 0], colors: list = None,
              last_update=None):
    _lines = last_update
    if _lines is None:
        _lines = o3d.geometry.LineSet()

    if pcl is not None:
        _lines.points = o3d.utility.Vector3dVector(pcl)
    if lines is not None:
        _lines.lines = o3d.utility.Vector2iVector(lines)
        if colors is None:
            colors = np.repeat([color], lines.shape[0], axis=0)
        _lines.colors = o3d.utility.Vector3dVector(colors)
    return _lines


def o3d_coord(size=0.1, origin=[0, 0, 0]):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)


def pcl2box(pcl: np.ndarray, radius: float = 0.0075):
    verts = []
    faces = []
    curr_faces = [
        [0, 1, 2], [1, 3, 2], [0, 4, 1], [1, 4, 5], [4, 6, 5], [5, 6, 7], [
            2, 3, 6], [3, 7, 6], [0, 2, 4], [2, 6, 4], [1, 5, 3], [3, 5, 7]
    ]
    for i in range(pcl.shape[0]):
        for u in [-1, 1]:
            for v in [-1, 1]:
                for w in [-1, 1]:
                    verts.append(
                        pcl[i] + np.array([u*radius, v*radius, w*radius]))
        verts_idxs = np.array(range(i*8, i*8+8))

        for face in curr_faces:
            faces.append(verts_idxs[face])

    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def pcl2sphere(pcl: np.ndarray, radius: float = 0.0075):
    verts = None
    faces = None
    for i in range(pcl.shape[0]):
        # The API used to be o3d.geometry.create_mesh_sphere
        current_mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
        current_faces = np.asarray(
            current_mesh.triangles)+(verts.shape[0] if verts is not None else 0)
        current_vertices = np.asarray(current_mesh.vertices) + pcl[i]
        if faces is None:
            faces = current_faces
        else:
            faces = np.vstack((faces, current_faces))  # N_face*3, int
        if verts is None:
            verts = current_vertices
        else:
            verts = np.vstack((verts, current_vertices))  # N*3
    return verts, faces


class O3DItemUpdater():
    def __init__(self, func) -> None:
        self.func = func
        self.update_item = func()

    def update(self, params: dict):
        self.func(last_update=self.update_item, **params)

    def get_item(self):
        return self.update_item


class O3DStreamPlot():
    pause = False
    speed_rate = 1

    def __init__(self, width=1600, height=1200, with_coord=True) -> None:
        self.view = o3d.visualization.VisualizerWithKeyCallback()
        self.view.create_window(width=width, height=height)
        self.ctr = self.view.get_view_control()
        self.render = self.view.get_render_option()
        try:
            self.render.point_size = 3.0
        except:
            print('No render setting')

        self.with_coord = with_coord
        self.first_render = True
        self.frame_idx = 0
        self.plot_funcs = dict()
        self.updater_dict = dict()
        self.init_updater()
        self.init_plot()
        self.init_key_cbk()

    def init_updater(self):
        self.plot_funcs = dict(exampel_pcl=o3d_pcl, example_mesh=o3d_mesh)
        raise RuntimeError(
            "'O3DStreamPlot.init_updater' method should be overriden")

    def init_plot(self):
        for updater_key, func in self.plot_funcs.items():
            updater = O3DItemUpdater(func)
            self.view.add_geometry(updater.get_item())
            if self.with_coord:
                self.view.add_geometry(o3d_coord())
            self.updater_dict[updater_key] = updater

    def init_key_cbk(self):
        key_map = dict(
            w=87, a=65, s=83, d=68, h=72, l=76, space=32, one=49, two=50, four=52
        )
        key_cbk = dict(
            w=lambda v: v.get_view_control().rotate(0, 40),
            a=lambda v: v.get_view_control().rotate(40, 0),
            s=lambda v: v.get_view_control().rotate(0, -40),
            d=lambda v: v.get_view_control().rotate(-40, 0),
            h=lambda v: v.get_view_control().scale(-2),
            l=lambda v: v.get_view_control().scale(2),
            space=lambda v: exec(
                "O3DStreamPlot.pause = not O3DStreamPlot.pause"),
            one=lambda v: exec("O3DStreamPlot.speed_rate = 1"),
            two=lambda v: exec("O3DStreamPlot.speed_rate = 2"),
            four=lambda v: exec("O3DStreamPlot.speed_rate = 4"),
        )

        for key, value in key_map.items():
            self.view.register_key_callback(value, key_cbk[key])

    def init_show(self):
        self.view.reset_view_point(True)
        self.first_render = False

    def update_plot(self):
        
        self.view.update_geometry(None)
        if self.first_render:
            self.init_show()
        self.view.poll_events()
        self.view.update_renderer()
        
        self.ctr.set_zoom(0.2) 

    def show(self, gen=None, fps: float = 30, save_path: str = ''):
        # print("[O3DStreamPlot] rotate: W(left)/A(up)/S(down)/D(right); resize: L(-)/H(+); pause/resume: space; speed: 1(1x)/2(2x)/4(4x)")

        if gen is None:
            gen = self.generator()
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        tick = time.time()
        while True:
            duration = 1/(fps*self.speed_rate)
            while time.time() - tick < duration:
                continue

            # print("[O3DStreamPlot] {} FPS".format(1/(time.time() - tick)))

            tick = time.time()

            try:
                if not self.pause:
                    update_dict = next(gen)
            except StopIteration as e:
                break

            for updater_key, update_params in update_dict.items():
                if updater_key not in self.updater_dict.keys():
                    continue
                self.updater_dict[updater_key].update(update_params)
            self.update_plot()
            if save_path:
                self.view.capture_screen_image(os.path.join(
                    save_path, '{}.png'.format(self.frame_idx)), True)
            self.frame_idx += 1

        # self.close_view()

    def show_manual(self, update_dict):
        for updater_key, update_params in update_dict.items():
            if updater_key not in self.updater_dict.keys():
                continue
            self.updater_dict[updater_key].update(update_params)
        self.update_plot()

    def close_view(self):
        self.view.close()
        self.view.destroy_window()

    def generator(self):
        raise RuntimeError(
            "'O3DStreamPlot.generator' method should be overriden")


class EvaluateStreamPlot(O3DStreamPlot):
    updater_dict = {}
    save_path = ''

    def __init__(self, *args, **kwargs) -> None:
        EvaluateStreamPlot.save_path = kwargs.pop("save_path", "./")
        super().__init__(width=1600, *args, **kwargs)

        EvaluateStreamPlot.updater_dict = self.updater_dict
        self.idx = 0

        def save(v):
            save_path = EvaluateStreamPlot.save_path

            o3d.io.write_triangle_mesh(os.path.join(save_path, "radar_pcl{}.ply".format(
                self.idx)), EvaluateStreamPlot.updater_dict["radar_mesh"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "depth0_pcl{}.ply".format(
                self.idx)), EvaluateStreamPlot.updater_dict["depth0_mesh"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "depth1_pcl{}.ply".format(
                self.idx)), EvaluateStreamPlot.updater_dict["depth1_mesh"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "pred_smpl{}.ply".format(
                self.idx)), EvaluateStreamPlot.updater_dict["pred_smpl"].update_item)
            o3d.io.write_triangle_mesh(os.path.join(save_path, "label_smpl{}.ply".format(
                self.idx)), EvaluateStreamPlot.updater_dict["label_smpl"].update_item)
            self.idx += 1
        self.view.register_key_callback(66, save)  # 66: b

    def init_updater(self):
        self.plot_funcs = dict(
            radar0=o3d_pcl,
            radar0_mesh=o3d_mesh,
            depth0=o3d_pcl,
            depth0_mesh=o3d_mesh,
            depth1=o3d_pcl,
            depth1_mesh=o3d_mesh,
            depth2=o3d_pcl,
            depth2_mesh=o3d_mesh,
            depth3=o3d_pcl,
            depth3_mesh=o3d_mesh,
            pred_smpl=o3d_mesh,
            label_smpl=o3d_mesh,
            vert_pcl=o3d_pcl,
            vert_lines=o3d_lines,
            grid_box=o3d_mesh,
            cluster_sphere=o3d_mesh,
            feat_lines=o3d_lines,
        )

    def init_show(self):
        # if self.line_width is not None:
        #     mat = o3d.visualization.rendering.MaterialRecord()
        #     mat.shader = "unlitLine"
        #     mat.line_width = 10  # note that this is scaled with respect to pixels,
        
        super().init_show()
        '''
        self.ctr.set_up(np.array([[0], [0], [1]]))
        self.ctr.set_front(np.array([[0], [-1], [0]]))
        
        # self.ctr.set_up(np.array([[0], [-1], [0]]))
        # self.ctr.set_front(np.array([[0], [0], [-1]]))
        self.ctr.set_lookat(np.array([0, 0, 0]))
        self.ctr.set_zoom(1)
        '''
        
        
        self.ctr.set_up(np.array([[0], [0], [1]]))  # Changing the 'up' vector
        self.ctr.set_front(np.array([[0], [-1], [0]]))  # Changing the 'front' vector
        self.ctr.set_lookat(np.array([0, 0, 0]))  # Point the camera focuses on
        self.ctr.set_zoom(0.2) 
        
        