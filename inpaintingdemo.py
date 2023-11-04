import cv2
import mediapipe as mp
import pyvirtualcam

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # depends on input device, usually 0


def get_coords(point, im):
    x = int(im.shape[1] * point.x)
    y = int(im.shape[0] * point.y)
    return (x, y)


def add_eye(minX, maxX, minY, maxY, eye, im):
    eye = cv2.resize(eye, (maxX - minX, maxY - minY))
    alpha = eye[:, :, 3] / 255.0
    alpha_inv = 1 - alpha
    for c in range(0, 3):
        im[minY:maxY, minX:maxX, c] = (
            alpha * eye[:, :, c] + alpha_inv * im[minY:maxY, minX:maxX, c]
        )


def draw(face, im):
    left_eye_outer = face[33]
    left_eye_inner = face[133]
    left_eye_upper = face[159]
    left_eye_lower = face[145]
    right_eye_outer = face[263]
    right_eye_inner = face[362]
    right_eye_upper = face[386]
    right_eye_lower = face[374]

    lox, _ = get_coords(left_eye_outer, im)
    lix, _ = get_coords(left_eye_inner, im)
    rox, _ = get_coords(right_eye_outer, im)
    rix, _ = get_coords(right_eye_inner, im)
    _, luy = get_coords(left_eye_upper, im)
    _, lly = get_coords(left_eye_lower, im)
    _, ruy = get_coords(right_eye_upper, im)
    _, rly = get_coords(right_eye_lower, im)

    eye = cv2.imread("eye.png", cv2.IMREAD_UNCHANGED)
    add_eye(min(lox, lix), max(lox, lix), min(luy, lly) - 5, max(luy, lly) + 5, eye, im)

    eye = cv2.flip(eye, 1)
    add_eye(min(rox, rix), max(rox, rix), min(ruy, rly) - 5, max(ruy, rly) + 5, eye, im)


def create_virtual_cam():
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True
    ) as face_mesh:
        _, frame1 = cap.read()
        with pyvirtualcam.Camera(
            width=frame1.shape[1], height=frame1.shape[0], fps=20
        ) as cam:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Error reading camera frame")
                    break

                # To improve performance, optionally mark the image as not writeable
                # to pass by reference
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face = face_landmarks.landmark
                        draw(face, image)

                        # mp_drawing.draw_landmarks(
                        #     image=image,
                        #     landmark_list=face_landmarks,
                        #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        #     landmark_drawing_spec=None,
                        #     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
                        # )

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cam.send(image)
                        cam.sleep_until_next_frame()

                cv2.imshow("Face", cv2.flip(image, 1))  # selfie flip
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()


if __name__ == "__main__":
    create_virtual_cam()
