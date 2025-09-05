import os

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeDetection01,
)
from azure.core.credentials import AzureKeyCredential
from utils import init_env_and_file


def main():
    try:
        endpoint, key, file_name, file_data = init_env_and_file(
            "FACE_ENDPOINT", "FACE_KEY"
        )

        face_client = FaceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

        result = face_client.detect(
            file_data,
            detection_model=FaceDetectionModel.DETECTION01,
            recognition_model=FaceRecognitionModel.RECOGNITION01,
            return_face_id=False,
            return_face_attributes=[
                FaceAttributeTypeDetection01.HEAD_POSE,
                FaceAttributeTypeDetection01.OCCLUSION,
                FaceAttributeTypeDetection01.ACCESSORIES,
            ],
        )

        face_count = 0

        if len(result) > 0:
            print(len(result), "faces detected.")
            for face in result:
                face_count += 1
                print("\nFace number {}".format(face_count))
                face_attributes = getattr(face, "face_attributes", None)
                if face_attributes:
                    print(
                        " - Head Pose (Yaw): {}".format(face_attributes.head_pose.yaw)
                    )
                    print(
                        " - Head Pose (Pitch): {}".format(
                            face_attributes.head_pose.pitch
                        )
                    )
                    print(
                        " - Head Pose (Roll): {}".format(face_attributes.head_pose.roll)
                    )
                    print(
                        " - Forehead occluded?: {}".format(
                            face_attributes.occlusion["foreheadOccluded"]
                        )
                    )
                    print(
                        " - Eye occluded?: {}".format(
                            face_attributes.occlusion["eyeOccluded"]
                        )
                    )
                    print(
                        " - Mouth occluded?: {}".format(
                            face_attributes.occlusion["mouthOccluded"]
                        )
                    )
                    if face_attributes.accessories:
                        print(" - Accessories:")
                        for accessory in face_attributes.accessories:
                            print("   - {}".format(accessory.type))

            # Annotate faces in the image
            annotate_faces(file_name, detected_faces=result)

    except Exception as ex:
        print(ex)


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


if __name__ == "__main__":
    main()
