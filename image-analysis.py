import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from utils import init_env_and_file


def main():
    try:
        endpoint, key, file_name, file_data = init_env_and_file(
            "VISION_ENDPOINT",
            "VISION_KEY",
        )

        cv_client = ImageAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        result = cv_client.analyze(
            image_data=file_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.READ,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE,
            ],
        )

        # Get image captions
        if result.caption is not None:
            print("\nCaption:")
            print(
                " Caption: '{}' (confidence: {:.2f}%)".format(
                    result.caption.text, result.caption.confidence * 100
                )
            )

        if result.dense_captions is not None:
            print("\nDense Captions:")
            for caption in result.dense_captions.list:
                print(
                    " Caption: '{}' (confidence: {:.2f}%)".format(
                        caption.text, caption.confidence * 100
                    )
                )

        # Get Image text
        if result.read is not None:
            print("\nText:")

            for line in result.read.blocks[0].lines:
                print(f" {line.text}")
            annotate_lines(file_name, result.read)

            # Find individual words in each line
            print("\nIndividual words:")
            for line in result.read.blocks[0].lines:
                for word in line.words:
                    print(f"  {word.text} (Confidence: {word.confidence:.2f}%)")
            # Annotate the words in the image
            annotate_words(file_name, result.read)

        # Get image tags
        if result.tags is not None:
            print("\nTags:")
            for tag in result.tags.list:
                print(
                    " Tag: '{}' (confidence: {:.2f}%)".format(
                        tag.name, tag.confidence * 100
                    )
                )

        # Get objects in the image
        if result.objects is not None:
            print("\nObjects in image:")
            for detected_object in result.objects.list:
                print(
                    " {} (confidence: {:.2f}%)".format(
                        detected_object.tags[0].name,
                        detected_object.tags[0].confidence * 100,
                    )
                )

        # Get people in the image
        if result.people is not None and len(result.people.list) > 0:
            print("\nPeople in image:")

            for detected_person in result.people.list:
                if detected_person.confidence > 0.2:
                    print(
                        " {} (confidence: {:.2f}%)".format(
                            detected_person.bounding_box,
                            detected_person.confidence * 100,
                        )
                    )
            show_people(file_name, result.people.list)

    except Exception as ex:
        print(ex)


def annotate_words(image_file, detected_text):
    print(f"\nAnnotating individual words in image...")

    # Prepare image for drawing
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width / 100, image.height / 100))
    plt.axis("off")
    draw = ImageDraw.Draw(image)
    color = "cyan"

    for line in detected_text.blocks[0].lines:
        for word in line.words:
            # Draw word bounding polygon
            r = word.bounding_polygon
            rectangle = (
                (r[0].x, r[0].y),
                (r[1].x, r[1].y),
                (r[2].x, r[2].y),
                (r[3].x, r[3].y),
            )
            draw.polygon(rectangle, outline=color, width=3)

    # Save image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    output_dir = os.path.join("img", "results")
    os.makedirs(output_dir, exist_ok=True)
    textfile = os.path.join(output_dir, "words.jpg")
    fig.savefig(textfile)
    print("  Results saved in", textfile)


def annotate_lines(image_file, detected_text):
    print(f"\nAnnotating lines of text in image...")

    # Prepare image for drawing
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width / 100, image.height / 100))
    plt.axis("off")
    draw = ImageDraw.Draw(image)
    color = "cyan"

    for line in detected_text.blocks[0].lines:
        # Draw line bounding polygon
        r = line.bounding_polygon
        rectangle = (
            (r[0].x, r[0].y),
            (r[1].x, r[1].y),
            (r[2].x, r[2].y),
            (r[3].x, r[3].y),
        )
        draw.polygon(rectangle, outline=color, width=3)

    # Save image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    output_dir = os.path.join("img", "results")
    os.makedirs(output_dir, exist_ok=True)
    textfile = os.path.join(output_dir, "lines.jpg")
    fig.savefig(textfile)
    print("  Results saved in", textfile)


def show_people(image_filename, detected_people):
    print("\nAnnotating people...")

    # Prepare image for drawing
    image = Image.open(image_filename)
    fig = plt.figure(figsize=(image.width / 100, image.height / 100))
    plt.axis("off")
    draw = ImageDraw.Draw(image)
    color = "cyan"

    for detected_person in detected_people:
        if detected_person.confidence > 0.2:
            # Draw object bounding box
            r = detected_person.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    output_dir = os.path.join("img", "results")
    os.makedirs(output_dir, exist_ok=True)
    peoplefile = os.path.join(output_dir, "people.jpg")
    fig.savefig(peoplefile)
    print("  Results saved in", peoplefile)


if __name__ == "__main__":
    main()
