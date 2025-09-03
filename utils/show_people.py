import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def show_people(image_filename, detected_people):
    print("\nAnnotating people...")

    # Prepare image for drawing
    image = Image.open(image_filename)
    fig = plt.figure(figsize=(image.width / 100, image.height / 100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for detected_person in detected_people:
        if detected_person.confidence > 0.2:
            # Draw object bounding box
            r = detected_person.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    output_dir = os.path.join('img', 'results')
    os.makedirs(output_dir, exist_ok=True)
    peoplefile = os.path.join(output_dir, 'people.jpg')
    fig.savefig(peoplefile)
    print('  Results saved in', peoplefile)
