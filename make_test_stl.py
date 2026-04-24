import trimesh, numpy as np

# L-bracket: two boxes joined
b1 = trimesh.creation.box(extents=[100, 20, 60])
b2 = trimesh.creation.box(extents=[20, 80, 60])
b2.apply_translation([40, 50, 0])
bracket = trimesh.util.concatenate([b1, b2])
bracket.export('l_bracket.stl')