
from pathlib import Path
import numpy as np
import math 
from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer

from .entity_mapper import EntityMapper

from occwl.io import load_step

class ColorMap:
    def __init__(self):
        self.color_values = [
            [255, 0, 0],  # Blue
            [0, 255, 0],  # Green
            [0, 0,255]    # Red
        ]

    def interpolate_value(self, a, b, t):
        return (b - a) * t + a

    def interpolate_color(self, t):
        # clamp t to [0, 1]
        t = max(0.0, min(1.0, float(t)))
        num_colors = len(self.color_values)
        tp = t * (num_colors - 1)
        index_before = int(math.floor(tp))
        index_after = min(int(math.ceil(tp)), num_colors - 1)
        tint = tp - index_before

        c0 = self.color_values[index_before]
        c1 = self.color_values[index_after]
        return [self.interpolate_value(c0[c], c1[c], tint) for c in range(3)]

class MultiSelectJupyterRenderer(JupyterRenderer):
    def __init__(self, *args, **kwargs):
        super(MultiSelectJupyterRenderer, self).__init__(*args, **kwargs)
            
    def click(self, value):
        """ called whenever a shape  or edge is clicked
        """
        try:
            obj = value.owner.object
            self.clicked_obj = obj
            if self._current_mesh_selection != obj:
                if obj is not None:
                    self._shp_properties_button.disabled = False
                    self._toggle_shp_visibility_button.disabled = False
                    self._remove_shp_button.disabled = False
                    id_clicked = obj.name  # the mesh id clicked
                    self._current_mesh_selection = obj
                    self._current_selection_material_color = obj.material.color
                    obj.material.color = self._selection_color
                    # selected part becomes transparent
                    obj.material.transparent = True
                    obj.material.opacity = 0.5
                    # get the shape from this mesh id
                    selected_shape = self._shapes[id_clicked]
                    self._current_shape_selection = selected_shape
                # then execute calbacks
                for callback in self._select_callbacks:
                    callback(self._current_shape_selection)
        except Exception as e:
            self.html.value = f"{str(e)}"

class JupyterSegmentationViewer:
    def __init__(self, file_path: Path, seg_folder=None, logit_folder=None):
        self.file_path = file_path
        assert file_path.exists()
    
        solids = self.load_step()
        assert len(solids) == 1, "Expect only 1 solid"
        self.solid = solids[0]
        self.entity_mapper = EntityMapper(self.solid.topods_shape())

        self.seg_folder = seg_folder
        self.logit_folder = logit_folder

        self.bit8_colors = [
            [235, 85, 79],  # ExtrudeSide
            [220, 198, 73], # ExtrudeEnd
            [113, 227, 76], # CutSide
            [0, 226, 124],  # CutEnd
            [23, 213, 221], # Fillet
            [92, 99, 222],  # Chamfer
            [176, 57, 223], # RevolveSide
            [238, 61, 178]  # RevolveEnd
        ]

        self.color_map = ColorMap()

        self.selection_list = []

    def format_color(self, c):
        return '#%02x%02x%02x' % (c[0], c[1], c[2])

    def load_step(self):
        return load_step(self.file_path)

    def select_face_callback(self, face):
        """
        Callback from the notebook when we select a face
        """
        face_index = self.entity_mapper.face_index(face)
        self.selection_list.append(face_index)

    def select_faces_by_indices(self, indices_to_select):
        """
        Programmatically select faces by their indices.
        """
        self.selection_list.clear()
        self.selection_list.extend(indices_to_select)
        print(f"{len(self.selection_list)} faces selected programmatically.")

    def view_solid(self):
        """
        Just show the solid.  No need to show any segmentation data
        """
        renderer = MultiSelectJupyterRenderer()
        renderer.register_select_callback(self.select_face_callback)
        renderer.DisplayShape(
            self.solid.topods_shape(), 
            topo_level="Face", 
            render_edges=True, 
            update=True,
            quality=1.0
        )

    
    def highlight_faces_with_indices(self, indices):
        """
         Подсветить грани с заданными индексами
        """
        indices = set(indices)

        highlighted_color = self.format_color([0, 255, 0])
        other_color = self.format_color([156, 152, 143])

        faces = self.solid.faces()
        colors = []

        for face in faces:
            face_index = self.entity_mapper.face_index(face.topods_shape())
            if face_index in indices:
                colors.append(highlighted_color)
            else:
                colors.append(other_color)
        self._display_faces_with_colors(self.solid.faces(), colors)

    def display_faces_with_heatmap(self, values, interval=None):
        if interval is None:
            min_val = np.min(values)
            max_val = np.max(values)
            interval = [min_val, max_val]
        
        interval_length = interval[1] - interval[0]
        
        # Handle the case where all values are the same (e.g., comparing identical models)
        if interval_length == 0:
            norm_values = np.zeros_like(values, dtype=float)
        else:
            norm_values = (values - interval[0]) / interval_length
        
        norm_values = np.clip(norm_values, 0.0, 1.0)

        faces = self.solid.faces()
        colors = []

        for face in faces:
            face_index = self.entity_mapper.face_index(face.topods_shape())
            norm_value = norm_values[face_index]
            color_list = self.color_map.interpolate_color(norm_value)
            int_color_list = [int(v) for v in color_list]
            color = self.format_color(int_color_list)
            colors.append(color)

        self._display_faces_with_colors(self.solid.faces(), colors)


    def _view_segmentation(self, face_segmentation):
        colors = []
        for segment in face_segmentation:
            color = self.format_color(self.bit8_colors[segment])
            colors.append(color)
        self._display_faces_with_colors(self.solid.faces(), colors)


    def _display_faces_with_colors(self, faces, colors):
        """
        Display the solid with each face colored
        with the given color
        """
        renderer = JupyterRenderer()
        output = []
        for face, face_color in zip(faces, colors):
            result = renderer.AddShapeToScene(
                face.topods_shape(), 
                shape_color=face_color, 
                render_edges=True, 
                edge_color="#000000",
                quality=1.0
            )
            output.append(result)

        # Add the output data to the pickable objects or nothing get rendered
        for elem in output:
            renderer._displayed_pickable_objects.add(elem)                                         

        # Now display the scene
        renderer.Display()