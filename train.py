from ultralytics import YOLO


def main():
    # Card finding model

    model = YOLO("yolo26n-pose.pt")
    model.train(
        data="dataset_pose_640/data.yaml",
        epochs=5, # quick test run change to train real model
        imgsz=640,
        batch=16,
        workers=6,
        device=0,
        patience=25,
        optimizer="auto",
        project="runs",
        name="card_pose_model"
    )


    ## Name/number finder model
    #model = YOLO("yolo26n-obb.pt")
    #model.train(
    #    resume=True,
    #    data="dataset_obb_640/data.yaml",
    #    epochs=180,
    #    imgsz=640,
    #    batch=18,
    #    workers=6,
    #    device=0,
    #    patience=25,
    #    project="runs",
    #    name="name_number_obb_model"
    #)




if __name__ == "__main__":
    main()
