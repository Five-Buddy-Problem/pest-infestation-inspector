from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    model = YOLO("best.pt")

    # # Option 1: Evaluate results using the .yaml file
    # test_metrics = model.val(data="C:\\Users\\Atabay\\PycharmProjects\\datasets\\eupoecilia_ambiguella\\data.yaml", split="test")
    # print(test_metrics)

    """
        Results:
            Box(P): 0.998
            Recall (R): 0.974
            mAP50: 0.994
            mAP50-95: 0.856
            composite fitness score: 0.87
        Speed Metrics:
            Preprocessing: ~0.8 ms per image
            Inference: ~26.3 ms per image
            Postprocessing: ~0.4 ms per image
    """

    # # Option 2: Run inference on the entire test folder.
    # CAREFUL! This will open every test image as a pop-up on the default application
    # results = model("dataset/eupoecilia_ambiguella/test/images")
    #
    # for result in results:
    #     result.show()
    #

    # Option 3: Show a single image
    # Pass the image path to the model
    results = model("eupoecilia_ambiguella/eupoecilia_ambiguella_444.jpg")


    # Save the annotated image as a .jpg file
    annotated_image = results[0].plot()
    cv2.imwrite("results/test_5.jpg", annotated_image)