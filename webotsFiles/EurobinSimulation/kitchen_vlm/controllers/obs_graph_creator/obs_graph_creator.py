from controller import Robot, Supervisor
# from rdflib import Graph, URIRef, Literal, Namespace, RDF
# from PIL import Image
# import base64
# import io
# import datetime

TIME_STEP = 32

robot = Robot()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)
camera.recognitionEnable(TIME_STEP)
supervisor = Supervisor()



def get_object_info_from_id(supervisor, object_id):
    """
    Given a supervisor instance and an object ID (from recognition),
    return a dictionary of information about the corresponding node.
    """
    node = supervisor.getFromId(object_id)
    if node is None:
        print(f"No node found with ID {object_id}")
        return None

    info = {}

    def get_field(name):
        field = node.getField(name)
        return field.getSFString() if field and field.getTypeName() == 'SFString' else (
            field.getSFVec3f() if field and field.getTypeName() == 'SFVec3f' else None
        )

    info['DEF'] = node.getDef()
    info['Type'] = node.getTypeName()
    info['Position'] = node.getPosition()  # global position
    info['Orientation'] = node.getOrientation()
    info['Velocity'] = node.getVelocity()
    info['Model'] = get_field("model")
    info['Name'] = get_field("name")
    info['Translation'] = get_field("translation")
    info['Rotation'] = get_field("rotation")

    # More exotic fields? Sure. Just add them here.
    
    return info


# def update_knowledge_graph_with_observation(
    # owl_path, 
    # object_list, 
    # image_data, 
    # save_path="orka_updated.owl"):
    # """
    # Loads an OWL ontology and appends object observations and image.
    # object_list: list of dicts with keys: id, model, position, size, color
    # image_data: raw image bytes (e.g., from camera.getImage())
    # """

    # Load the OWL file
    # g = Graph()
    # g.parse(owl_path, format='xml')

    # Use a namespace — modify to match your ontology
    # ORKA = Namespace("http://www.orca-ontology.org#")
    # OBS = Namespace("http://example.org/observation#")

    # g.bind("orka", ORKA)
    # g.bind("obs", OBS)

    # timestamp = datetime.datetime.utcnow().isoformat()

    # Add observation data
    # for obj in object_list:
        # obj_uri = URIRef(f"{OBS}Object_{obj['id']}_{timestamp}")
        # g.add((obj_uri, RDF.type, ORKA.ObservedObject))
        # g.add((obj_uri, ORKA.hasModel, Literal(obj.get("model", "unknown"))))
        # g.add((obj_uri, ORKA.hasPosition, Literal(str(obj.get("position")))))
        # g.add((obj_uri, ORKA.hasSize, Literal(str(obj.get("size")))))
        # g.add((obj_uri, ORKA.hasColor, Literal(str(obj.get("color")))))
        # g.add((obj_uri, ORKA.timestamp, Literal(timestamp)))

    # Encode image in base64 and store
    # if image_data:
        # image = Image.frombytes('RGB', (640, 480), image_data)
        # buffer = io.BytesIO()
        # image.save(buffer, format="JPEG")
        # image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        # img_uri = URIRef(f"{OBS}Image_{timestamp}")
        # g.add((img_uri, RDF.type, ORKA.CameraImage))
        # g.add((img_uri, ORKA.imageData, Literal(image_base64)))
        # g.add((img_uri, ORKA.timestamp, Literal(timestamp)))

    # Save the updated graph
    # g.serialize(destination=save_path, format='xml')
    # print(f"Graph updated and saved to {save_path}")
    # return g


while robot.step(TIME_STEP) != -1:
    objects = camera.getRecognitionObjects()
    
    print(f"Number of recognized objects: {len(objects)}")
    
    for i, obj in enumerate(objects):
        print(f"\nObject {i + 1}")
        object_id = obj.getId()
        print(f"  ID: {obj.getId()}")
        pos = obj.getPosition()
        print(f"  Position (relative): x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
        
        orient = obj.getOrientation()
        print(f"  Orientation: [{orient[0]:.2f}, {orient[1]:.2f}, {orient[2]:.2f}, {orient[3]:.2f}]")
        
        size = obj.getSize()
        print(f"  Size: height={size[0]:.2f}, width={size[1]:.2f}")
        
        pos_img = obj.getPositionOnImage()
        print(f"  Position on image: x={pos_img[0]}, y={pos_img[1]}")
        
        size_img = obj.getSizeOnImage()
        print(f"  Size on image: width={size_img[0]}, height={size_img[1]}")

        print(f"  Number of colors: {obj.getNumberOfColors()}")
        
        colors = obj.getColors()
        for j in range(obj.getNumberOfColors()):
            r = colors[j * 3]
            g = colors[j * 3 + 1]
            b = colors[j * 3 + 2]
            print(f"  Color {j + 1}: R={r:.2f}, G={g:.2f}, B={b:.2f}")
        
        model = obj.getModel()
        print(f"  Model: {model}")
        info = get_object_info_from_id(supervisor, object_id)

        if info:
            for key, value in info.items():
                print(f"\t\t{key}: {value}")