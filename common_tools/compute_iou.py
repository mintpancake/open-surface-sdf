import trimesh

def compute_boolean(mesh_list, operation):
    if operation == "intersection":
        return trimesh.boolean.intersection(mesh_list)
    elif operation == "union":
        return trimesh.boolean.union(mesh_list)
    else:
        RuntimeError("operation is wrong")

def compute_iou(mesh_file_list, file_type = "ply"):
    mesh_obj_list = []
    for mesh_file in mesh_file_list:
        mesh_obj = trimesh.exchange.load.load_mesh(mesh_file, file_type=file_type)
        mesh_obj_list.append(mesh_obj)

    intersected = compute_boolean(mesh_obj_list, "intersection")
    union = compute_boolean(mesh_obj_list, "union")
    return intersected.volume/union.volume

if __name__ == "__main__":
    iou_l1_smallbatch = compute_iou(["normalized_fandisk.obj", "normalized_homer.obj"], file_type="obj")
    print(iou_l1_smallbatch)


