import cv2

import pyvirtualcam


def take_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite("photo.jpg", frame)
    cap.release()


def capture_video():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Read error")
            break

        cv2.imshow("Cam", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def create_virtual_cam():
    cap = cv2.VideoCapture(0)

    _, frame1 = cap.read()
    with pyvirtualcam.Camera(
        width=frame1.shape[1], height=frame1.shape[0], fps=30
    ) as cam:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Read error")
                break

            # Modify image
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cam.send(image)
            cam.sleep_until_next_frame()

            cv2.imshow("Cam", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # capture_video()
    create_virtual_cam()
