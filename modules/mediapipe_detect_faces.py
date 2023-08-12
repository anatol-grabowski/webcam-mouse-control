from modules.mp_landmarks_to_points import mp_landmarks_to_points


def mediapipe_detect_faces(face_mesh, rgb, num_warmup=3, num_avg=3):
    for i in range(num_warmup):
        output = face_mesh.process(rgb)
    output = face_mesh.process(rgb)
    if output is None or output.multi_face_landmarks is None:
        return None
    faces = mp_landmarks_to_points(output.multi_face_landmarks)
    for i in range(num_avg - 1):
        output = face_mesh.process(rgb)
        if output is None or output.multi_face_landmarks is None:
            return None
        faces += mp_landmarks_to_points(output.multi_face_landmarks)
    faces /= num_avg
    return faces
