from controller import Supervisor

def extract_color_from_node(node):
    shape = None
    children = node.getField("children")
    if not children:
        return None

    for i in range(children.getCount()):
        child = children.getMFNode(i)
        if child.getTypeName() == "Shape":
            shape = child
            break

    if not shape:
        return None

    appearance = shape.getField("appearance")
    if not appearance:
        return None

    appearance_node = appearance.getSFNode()
    if not appearance_node or appearance_node.getTypeName() != "Appearance":
        return None

    material = appearance_node.getField("material")
    if not material:
        return None

    material_node = material.getSFNode()
    if not material_node or material_node.getTypeName() != "Material":
        return None

    color_field = material_node.getField("diffuseColor")
    if color_field:
        return color_field.getSFColor()

    return None

def extract_shape_from_node(node):
    children = node.getField("children")
    if children:
        for i in range(children.getCount()):
            shape = children.getMFNode(i)
            if shape.getTypeName() == "Shape":
                geometry = shape.getField("geometry")
                if geometry:
                    return geometry.getSFNode().getTypeName()
    return None

def scan_world_objects(supervisor):
    root = supervisor.getRoot()
    children_field = root.getField("children")

    object_data = []
    for i in range(children_field.getCount()):
        node = children_field.getMFNode(i)
        if not node:
            continue

        node_id = node.getId()
        name_field = node.getField("name")
        if name_field:
            name = name_field.getSFString()
        else: name = ''
        typename = node.getTypeName()
        position = node.getPosition()
        color = extract_color_from_node(node)
        shape = extract_shape_from_node(node)

        object_data.append({
            "id": node_id,
            "name": name,
            "typename": typename,
            "position": [round(p, 3) for p in position] if position else None,
            "color": [round(c, 3) for c in color] if color else None,
            "shape": shape
        })

    return object_data

# Run as main script
if __name__ == "__main__":
    supervisor = Supervisor()
    TIME_STEP = int(supervisor.getBasicTimeStep())

    print("Scanning world...")
    supervisor.step(TIME_STEP)
    data = scan_world_objects(supervisor)

    for obj in data:
        print(f"Object ID: {obj['id']}")
        print(f"  Name: {obj['name']}")
        print(f"  TypeName: {obj['typename']}")
        print(f"  Position: {obj['position']}")
        print(f"  Color: {obj['color']}")
        print(f"  Shape: {obj['shape']}")
        print()
