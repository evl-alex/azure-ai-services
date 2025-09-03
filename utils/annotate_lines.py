import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

def annotate_lines(image_file, detected_text):
    print(f'\nAnnotating lines of text in image...')

     # Prepare image for drawing
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for line in detected_text.blocks[0].lines:
        # Draw line bounding polygon
        r = line.bounding_polygon
        rectangle = ((r[0].x, r[0].y),(r[1].x, r[1].y),(r[2].x, r[2].y),(r[3].x, r[3].y))
        draw.polygon(rectangle, outline=color, width=3)

    # Save image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    output_dir = os.path.join('img', 'results')
    os.makedirs(output_dir, exist_ok=True)
    textfile = os.path.join(output_dir, 'lines.jpg')
    fig.savefig(textfile)
    print('  Results saved in', textfile)
