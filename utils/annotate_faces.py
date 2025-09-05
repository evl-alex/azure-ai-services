import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def annotate_faces(image_file, detected_faces):
    print("\nAnnotating faces in image...")

    # Prepare image for drawing
    fig = plt.figure(figsize=(8, 6))
    plt.axis("off")
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)
    color = "lightgreen"

    # Annotate each face in the image
    face_count = 0
    for face in detected_faces:
        face_count += 1
        r = face.face_rectangle
        bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle(bounding_box, outline=color, width=5)
        annotation = "Face number {}".format(face_count)
        plt.annotate(annotation, (r.left, r.top), backgroundcolor=color)

    # Save annotated image
    plt.imshow(image)
    output_dir = os.path.join("img", "results")
    os.makedirs(output_dir, exist_ok=True)
    outputfile = os.path.join(output_dir, "detected_faces.jpg")
    fig.savefig(outputfile)
    print(f"  Results saved in {outputfile}\n")
