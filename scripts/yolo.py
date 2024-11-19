from ultralytics import YOLO, checks, hub

model = YOLO("models/yolov11m-fish_model.pt")

def train_yolo():
    checks()

    hub.login('1572b4afca7a8f8edf3315d1bc1ea4b48404bf9000')

    model = YOLO('https://hub.ultralytics.com/models/qReiuPj4S03FfAxTxPjo')
    results = model.train()
    return results


def get_result_per_image(path):
    # Run inference
    results = model(path)

    # Print image.jpg results in JSON format
    print(results[0].to_json())
    


if __name__ == '__main__':
    # train_yolo()

    get_result_per_image()
