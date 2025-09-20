
import numpy as np
from scipy.spatial import KDTree
from OCC.Core.STEPControl import STEPControl_Reader

from OCC.Extend import TopologyUtils
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FACE


from ...utils.scale_utils import scale_solid_to_unit_box 

from .entity_mapper import EntityMapper

class SDFComputer:
    def __init__(self, step_file, output_dir, scale_body=True):
        self.step_file = step_file
        self.output_dir = output_dir
        self.scale_body = scale_body
        self.uv_sample_resolution = 128
        self.n_sdf_samples_per_face = 500


    def process(self):
        """
        Process the file and extract the derivative data
        """
        # Load the body from the STEP file
        body = self.load_body_from_step()

        # We want to apply a transform so that the solid
        # is centered on the origin and scaled so it just fits
        # into a box [-1, 1]^3
        if self.scale_body:
            body = scale_solid_to_unit_box(body)

        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)

        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return

        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return
                
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return

        entity_mapper = EntityMapper(body)
        

        # TODO: Перевести на gpu
        uv_samples_faces, sdf_values_faces = self.extract_uv_samples_and_sdf(body, entity_mapper)

        output_pathname = self.output_dir / f"{self.step_file.stem}.npz"
        np.savez_compressed(
            output_pathname, 
            uv_faces=uv_samples_faces,      
            sdf_faces=sdf_values_faces  
        )

    def load_body_from_step(self):
        """
        Load the body from the step file.  
        We expect only one body in each file
        """
        step_filename_str = str(self.step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape


    def extract_uv_samples_and_sdf(self, body, entity_mapper):
        """
        For each face we sample UV points, classify them as inside/outside
        and compute the SDF values. 

        We return three lists of arrays, one per face

            - uv_samples_faces  [F, n_samples, 2]  нормализованные UV
            - sdf_values_faces  [F, n_samples]     таргеты SDF
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        uv_samples_faces = []
        # xyz_targets_faces = []
        sdf_values_faces = []

        for face in top_exp.faces(): 

            uv_samples = sample_uv_extended(self.uv_sample_resolution, extend=0.1)

            inside_uv, outside_uv = self.query_cad_kernel_face(face, uv_samples)
            
            sdf_points, sdf_values = compute_sdf(inside_uv, outside_uv)

            # xyz_targets = self.compute_xyz_from_uv_face(face, sdf_points)
            sdf_points, sdf_values = bias_sample_sdf(sdf_points, sdf_values, self.n_sdf_samples_per_face, boundary_ratio=0.4)
            uv_samples_faces.append(sdf_points)
            # xyz_targets_faces.append(xyz_targets)
            sdf_values_faces.append(sdf_values)

        uv_samples_faces  = np.stack(uv_samples_faces,  axis=0)  # [F, n, 2]
        # xyz_targets_faces = np.stack(xyz_targets_faces, axis=0)  # [F, n, 3]
        sdf_values_faces  = np.stack(sdf_values_faces,  axis=0)  # [F, n]

        return uv_samples_faces, sdf_values_faces

    def query_cad_kernel_face(self, face, uv_samples):
        """
        Делим UV на inside/outside относительно ТРИМОВ грани через CAD-ядро.
        """
        uv = ensure_uv_2d(uv_samples)
        surf = BRepAdaptor_Surface(face)
        u0, u1 = surf.FirstUParameter(), surf.LastUParameter()
        v0, v1 = surf.FirstVParameter(), surf.LastVParameter()
        clf = BRepClass_FaceClassifier()

        mask = []
        for uv_ in uv:
            uu = float(u0 + float(uv_[0]) * (u1 - u0))
            vv = float(v0 + float(uv_[1]) * (v1 - v0))
            p2d = gp_Pnt2d(uu, vv)
            clf.Perform(face, p2d, 1e-9)
            mask.append(clf.State() == TopAbs_IN)

        mask = np.array(mask, dtype=bool)
        inside  = uv[mask]
        outside = uv[~mask]
        return ensure_uv_2d(inside), ensure_uv_2d(outside)

    def compute_xyz_from_uv_face(self, face, uv_coords):
        uv = ensure_uv_2d(uv_coords)
        if uv.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        surf = BRepAdaptor_Surface(face)
        u0, u1 = surf.FirstUParameter(), surf.LastUParameter()
        v0, v1 = surf.FirstVParameter(), surf.LastVParameter()
        uu = u0 + uv[:, 0] * (u1 - u0)
        vv = v0 + uv[:, 1] * (v1 - v0)
        out = np.zeros((uv.shape[0], 3), dtype=np.float32)
        for i in range(uv.shape[0]):
            p = surf.Value(float(uu[i]), float(vv[i]))
            out[i, 0] = p.X(); out[i, 1] = p.Y(); out[i, 2] = p.Z()
        return out


    def check_unique_coedges(self, top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                
                # We want to detect the case where the coedges
                # are not unique
                if tup in coedge_set:
                    return False

                coedge_set.add(tup)

        return True
        
    def check_closed(self, body):
        # In Open Cascade, unlinked (open) edges can be identified
        # as they appear in the edges iterator when ignore_orientation=False
        # but are not present in any wire
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0


    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set


    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set


    def check_manifold(self, top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True

def ensure_uv_2d(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        return x
    if x.ndim == 1 and x.shape[0] == 2:
        return x.reshape(1, 2)
    return x.reshape(-1, 2)

# https://github.com/zhenshihaowanlee/Self-supervised-BRep-learning-for-CAD/blob/main/Example_data/sampled_data.py

def sample_uv_extended(resolution, extend=0.1):
    u = np.linspace(-extend, 1 + extend, resolution)
    v = np.linspace(-extend, 1 + extend, resolution)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    return np.stack([uu.flatten(), vv.flatten()], axis=-1)

def compute_sdf(inside_points, outside_points):
    inside = ensure_uv_2d(inside_points)
    outside = ensure_uv_2d(outside_points)

    if inside.shape[0] == 0 and outside.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if inside.shape[0] == 0:
        return outside, -np.zeros((outside.shape[0],), dtype=np.float32)
    if outside.shape[0] == 0:
        return inside, np.zeros((inside.shape[0],), dtype=np.float32)

    inside_tree = KDTree(outside)
    outside_tree = KDTree(inside)
    d_inside, _ = inside_tree.query(inside)
    d_outside, _ = outside_tree.query(outside)

    sdf_inside = d_inside.astype(np.float32)
    sdf_outside = -d_outside.astype(np.float32)
    sdf_points = np.concatenate([inside, outside], axis=0)
    sdf_values = np.concatenate([sdf_inside, sdf_outside], axis=0)
    return ensure_uv_2d(sdf_points), sdf_values.astype(np.float32)

def bias_sample_sdf(sdf_points, sdf_values, n_samples, boundary_ratio=0.4):
    pts = ensure_uv_2d(sdf_points)
    vals = np.asarray(sdf_values, dtype=np.float32).reshape(-1)
    if pts.shape[0] == 0:
        return pts, vals
    idx = np.argsort(np.abs(vals))
    nb = int(n_samples * boundary_ratio)
    nb = max(0, min(nb, idx.size))
    i_boundary = idx[:nb]
    i_pool = idx[nb:]
    if i_pool.size:
        i_pool = np.random.permutation(i_pool)
    need_rand = max(0, n_samples - nb)
    i_sel = np.concatenate([i_boundary, i_pool[:min(need_rand, i_pool.size)]], axis=0)
    i_sel = i_sel.astype(int)
    return pts[i_sel], vals[i_sel]