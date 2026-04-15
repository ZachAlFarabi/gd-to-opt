import trimesh

# Simple box — 100 x 80 x 60 mm
box = trimesh.creation.box(extents=[100, 80, 60])
box.export('test_bracket.stl')
print('Saved test_bracket.stl')

# Or a cylinder if you want something more interesting
# cyl = trimesh.creation.cylinder(radius=30, height=80)
# cyl.export('test_cylinder.stl')

# Or an L-bracket shape (two boxes unioned)
# import numpy as np
# b1 = trimesh.creation.box(extents=[100, 20, 60])
# b2 = trimesh.creation.box(extents=[20, 60, 60])
# b2.apply_translation([0, 40, 0])
# bracket = trimesh.util.concatenate([b1, b2])
# bracket.export('test_lbracket.stl')