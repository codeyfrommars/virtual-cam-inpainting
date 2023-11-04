import cv2
import mediapipe as mp
import pyvirtualcam

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # depends on input device, usually 0


def get_coords(point, im):
    x = int(im.shape[1] * point.x)
    y = int(im.shape[0] * point.y)
    return (x, y)


# def draw(face, image):
#     lip_mid = face[164]
#     lip_left = face[322]
#     lip_right = face[92]
#     lip_upper = face[2]
#     lip_lower = face[0]
#
#     x_mid, _ = get_coords(lip_mid, image)
#     x_left, _ = get_coords(lip_left, image)
#     x_right, _ = get_coords(lip_right, image)
#
#     _, y_upper = get_coords(lip_upper, image)
#     _, y_lower = get_coords(lip_lower, image)
#
#     stash = cv2.imread('halfstash.png', cv2.IMREAD_UNCHANGED)
#     add_halfstash(x_mid, x_left, y_upper, y_lower, stash, image)
#
#     stash = cv2.flip(stash, 1)
#     add_halfstash(x_right, x_mid, y_upper, y_lower, stash, image)

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
                    # face = face_landmarks.landmark
                    # draw(face, image)

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cam.send(image)
                    cam.sleep_until_next_frame()

            cv2.imshow("Face", cv2.flip(image, 1))  # selfie flip
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
