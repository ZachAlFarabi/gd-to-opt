import numpy as np
import trimesh

def create_heat_sink(
    base_length=100.0,
    base_width=80.0,
    base_thickness=5.0,
    fin_height=30.0,
    fin_thickness=2.0,
    fin_spacing=6.0,
    num_fins=10
):
    """
    Creates a heat sink mesh using trimesh.
    
    Units are arbitrary (e.g., mm).
    """

    meshes = []

    # --- Base plate ---
    base = trimesh.creation.box(
        extents=[base_length, base_width, base_thickness]
    )
    base.apply_translation([0, 0, base_thickness / 2])
    meshes.append(base)

    # --- Fins ---
    total_fin_width = num_fins * fin_thickness + (num_fins - 1) * fin_spacing
    start_x = -total_fin_width / 2 + fin_thickness / 2

    for i in range(num_fins):
        x_pos = start_x + i * (fin_thickness + fin_spacing)

        fin = trimesh.creation.box(
            extents=[fin_thickness, base_width, fin_height]
        )

        # Position fin on top of base
        fin.apply_translation([
            x_pos,
            0,
            base_thickness + fin_height / 2
        ])

        meshes.append(fin)

    # --- Combine all meshes ---
    heat_sink = trimesh.util.concatenate(meshes)

    return heat_sink


if __name__ == "__main__":
    mesh = create_heat_sink(
        base_length=120,
        base_width=80,
        base_thickness=5,
        fin_height=40,
        fin_thickness=2,
        fin_spacing=5,
        num_fins=12
    )

    # Export to STL
    mesh.export("heat_sink.stl")

    # Optional: show in viewer
    mesh.show()