from dotenv import load_dotenv
import os
import sys
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from utils import show_people, annotate_words, annotate_lines

def main():
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        load_dotenv()
        endpoint = os.getenv('VISION_ENDPOINT')
        key = os.getenv('VISION_KEY')

        if not endpoint or not key:
            raise ValueError("Missing ENDPOINT or KEY environment variables. Check your .env file.")

        # Validate CLI args
        if len(sys.argv) < 2:
            raise ValueError("Usage: python image-analysis.py <path-to-image>")
        image_file = sys.argv[1]

        if not os.path.exists(image_file):
            print(f"File not found: {image_file}")
            exit(2)

        cv_client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key))

        with open(image_file, "rb") as f:
            image_data = f.read()

        print(f'\nAnalyzing {image_file} ...\n')

        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.READ,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ],
        )

        # Get image captions
        if result.caption is not None:
            print("\nCaption:")
            print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))
            
        if result.dense_captions is not None:
            print("\nDense Captions:")
            for caption in result.dense_captions.list:
                print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))
        
        # Get Image text
        if result.read is not None:
            print("\nText:")
            
            for line in result.read.blocks[0].lines:
                print(f" {line.text}")
            annotate_lines(image_file, result.read)
            
            # Find individual words in each line
            print ("\nIndividual words:")
            for line in result.read.blocks[0].lines:
                for word in line.words:
                    print(f"  {word.text} (Confidence: {word.confidence:.2f}%)")
            # Annotate the words in the image
            annotate_words(image_file, result.read)

        # Get image tags
        if result.tags is not None:
            print("\nTags:")
            for tag in result.tags.list:
                print(" Tag: '{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))

        # Get objects in the image
        if result.objects is not None:
            print("\nObjects in image:")
            for detected_object in result.objects.list:
                print(" {} (confidence: {:.2f}%)".format(detected_object.tags[0].name, detected_object.tags[0].confidence * 100))

        # Get people in the image
        if result.people is not None and len(result.people.list) > 0:
            print("\nPeople in image:")

            for detected_person in result.people.list:
                if detected_person.confidence > 0.2:
                    print(" {} (confidence: {:.2f}%)".format(detected_person.bounding_box, detected_person.confidence * 100))
            show_people(image_file, result.people.list)
  
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()
