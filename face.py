from dotenv import load_dotenv
import os
import sys
from typing import cast
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection01
from azure.core.credentials import AzureKeyCredential
from utils import annotate_faces

# attrs = cast(FaceAttributes, face.face_attributes)
features = [FaceAttributeTypeDetection01.HEAD_POSE,
             FaceAttributeTypeDetection01.OCCLUSION,
             FaceAttributeTypeDetection01.ACCESSORIES]

def main():
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        load_dotenv()
        endpoint = os.getenv('FACE_ENDPOINT')
        key = os.getenv('FACE_KEY')

        if not endpoint or not key:
            raise ValueError("Missing ENDPOINT or KEY environment variables. Check your .env file.")

        # Validate CLI args
        if len(sys.argv) < 2:
            raise ValueError("Usage: python image-analysis.py <path-to-image>")
        image_file = sys.argv[1]

        if not os.path.exists(image_file):
            print(f"File not found: {image_file}")
            exit(2)

        face_client = FaceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key))

        with open(image_file, "rb") as f:
            image_data = f.read()

        print(f'\nAnalyzing {image_file} ...\n')
        


        result = face_client.detect(
            image_data,
            detection_model=FaceDetectionModel.DETECTION01,
            recognition_model=FaceRecognitionModel.RECOGNITION01,
            return_face_id=False,
            return_face_attributes=[
                FaceAttributeTypeDetection01.HEAD_POSE,
                FaceAttributeTypeDetection01.OCCLUSION,
                FaceAttributeTypeDetection01.ACCESSORIES
            ],
        )
        
        face_count = 0
        
        if len(result) > 0:
            print(len(result), 'faces detected.')
            for face in result:
                face_count += 1
                print('\nFace number {}'.format(face_count))
                face_attributes = getattr(face, "face_attributes", None)
                if face_attributes:
                    print(' - Head Pose (Yaw): {}'.format(face_attributes.head_pose.yaw))
                    print(' - Head Pose (Pitch): {}'.format(face_attributes.head_pose.pitch))
                    print(' - Head Pose (Roll): {}'.format(face_attributes.head_pose.roll))
                    print(' - Forehead occluded?: {}'.format(face_attributes.occlusion["foreheadOccluded"]))
                    print(' - Eye occluded?: {}'.format(face_attributes.occlusion["eyeOccluded"]))
                    print(' - Mouth occluded?: {}'.format(face_attributes.occlusion["mouthOccluded"]))
                    if face_attributes.accessories:
                        print(' - Accessories:')
                        for accessory in face_attributes.accessories:
                            print('   - {}'.format(accessory.type))
            
            # Annotate faces in the image
            annotate_faces(image_file, detected_faces=result)
  
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()
