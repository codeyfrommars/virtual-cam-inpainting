import cv2
import mediapipe as mp
import pyvirtualcam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

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


def add_outline(face, im):
    img_height, img_width = im.shape[:2]
    left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    right_eye_indices = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
    left_eye_landmarks = [face[i] for i in left_eye_indices]
    right_eye_landmarks = [face[i] for i in right_eye_indices]
    left_eye_landmarks = [(int(landmark.x * img_width), int(landmark.y * img_height)) for landmark in left_eye_landmarks]
    right_eye_landmarks = [(int(landmark.x * img_width), int(landmark.y * img_height)) for landmark in right_eye_landmarks]
    left_eye_np = np.array(left_eye_landmarks)
    right_eye_np = np.array(right_eye_landmarks)
    
    eye_fill_mask = np.zeros_like(im, dtype=np.uint8)
    eye_outline_mask = np.zeros_like(im, dtype=np.uint8)
    cv2.fillPoly(eye_fill_mask, [left_eye_np], (0, 255, 0))
    cv2.fillPoly(eye_fill_mask, [right_eye_np], (0, 255, 0))
    cv2.polylines(eye_outline_mask, [left_eye_np], True, (0, 0, 255), 2)
    cv2.polylines(eye_outline_mask, [right_eye_np], True, (0, 0, 255), 2)

    alpha = 0.4
    fill_mask = eye_fill_mask.astype(bool)
    outline_mask = eye_outline_mask.astype(bool)
    im[fill_mask] = cv2.addWeighted(im, alpha, eye_fill_mask, 1 - alpha, 0)[fill_mask]
    im[outline_mask] = eye_outline_mask[outline_mask]

    return im


def get_dist(pts):
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_ratio(face, im):
    img_height, img_width = im.shape[:2]
    left_hor_indices = [33, 133]
    left_vert_indices = [145, 159]
    right_hor_indices = [263, 362]
    right_vert_indices = [374, 386]
    left_hor_landmarks = [face[i] for i in left_hor_indices]
    left_vert_landmarks = [face[i] for i in left_vert_indices]
    right_hor_landmarks = [face[i] for i in right_hor_indices]
    right_vert_landmarks = [face[i] for i in right_vert_indices]
    left_hor_landmarks = [(int(landmark.x * img_width), int(landmark.y * img_height)) for landmark in left_hor_landmarks]
    left_vert_landmarks = [(int(landmark.x * img_width), int(landmark.y * img_height)) for landmark in left_vert_landmarks]
    right_hor_landmarks = [(int(landmark.x * img_width), int(landmark.y * img_height)) for landmark in right_hor_landmarks]
    right_vert_landmarks = [(int(landmark.x * img_width), int(landmark.y * img_height)) for landmark in right_vert_landmarks]
    left_hor_dist = get_dist(left_hor_landmarks)
    left_vert_dist = get_dist(left_vert_landmarks)
    right_hor_dist = get_dist(right_hor_landmarks)
    right_vert_dist = get_dist(right_vert_landmarks)
    return ((left_hor_dist / left_vert_dist) + (right_hor_dist / right_vert_dist)) / 2


def draw(face, im):
    # Face mesh indices:
    # https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
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
            ratios = []
            frame_num = 0
            frames = []

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Error reading camera frame")
                    break
                frame_num += 1
                frames.append(frame_num)

                # To improve performance, optionally mark the image as not writeable
                # to pass by reference
                image.flags.writeable = False
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image
                image.flags.writeable = True
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face = face_landmarks.landmark
                        ratio = get_ratio(face, image)
                        ratios.append(get_ratio(face, image))
                        if ratio > 5: # This number may need to be tuned for each user or intelligently set and not just hard-coded
                            image = cv2.flip(image, 1)
                            cv2.putText(image, 'Blink', (image.shape[0]//2, 100), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 255), 2)
                            image = cv2.flip(image, 1)
                        else:
                            draw(face, image)
                            # image = add_outline(face, image)

                        # mp_drawing.draw_landmarks(
                        #     image=image,
                        #     landmark_list=face_landmarks,
                        #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        #     landmark_drawing_spec=None,
                        #     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
                        # )

                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cam.send(image)
                        cam.sleep_until_next_frame()

                cv2.imshow("Face", cv2.flip(image, 1))  # selfie flip
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("Face", cv2.WND_PROP_VISIBLE) < 1:
                    break

    cap.release()
    cv2.destroyAllWindows()

    # blink_inds = []
    # n_stds = 2
    # for i in range(30, len(ratios)):
    #     if np.mean(ratios[i-30:i]) + (n_stds * np.std(ratios[i-30:i])) < ratios[i]:
    #         blink_inds.append(i)

    # plt.figure()
    # plt.gca().set(title='Eye Ratio over Time', xlabel='Frame Number', ylabel='Ratio of Eye Height to Width', xlim=[0, len(ratios)-1])
    # plt.plot(ratios)
    # plt.vlines(blink_inds, 0, np.max(ratios), colors='r', linestyles='dashed')
    # plt.show()


if __name__ == "__main__":
    create_virtual_cam()
