from ultralytics import YOLO, checks, hub


def train_yolo():
    checks()

    hub.login('1572b4afca7a8f8edf3315d1bc1ea4b48404bf9000')

    model = YOLO('https://hub.ultralytics.com/models/qReiuPj4S03FfAxTxPjo')
    results = model.train()
    return results


if __name__ == '__main__':
    train_yolo()