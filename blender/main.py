import bpy
import mathutils
import math
import sys
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import time
import os
from keentools.utils import coords, materials


def get_eyes(head):
    left_eye_center_vertindices = [
        10022, 10123, 10290, 10375, 17860, 17864, 17866, 17873, 17875, 17879, 17884, 17888]
    leye_center = mathutils.Vector([0, 0, 0])
    for i in left_eye_center_vertindices:
        leye_center += head.data.vertices[i].co @ head.matrix_world
    leye_center /= len(left_eye_center_vertindices)

    right_eye_center_vertindices = [
        10142, 10155, 10177, 10181, 17951, 17952, 17953, 17959, 17972, 17975, 17977, 17983]
    reye_center = mathutils.Vector([0, 0, 0])
    for i in right_eye_center_vertindices:
        reye_center += head.data.vertices[i].co @ head.matrix_world
    reye_center /= len(right_eye_center_vertindices)

    eyes_center = leye_center + (reye_center - leye_center) / 2
    return eyes_center, leye_center, reye_center


def get_cam_axes(cam):
    camx = mathutils.Vector([1, 0, 0])  # cam left/right
    camx.rotate(cam.rotation_euler)
    camy = mathutils.Vector([0, 1, 0])  # cam down/up
    camy.rotate(cam.rotation_euler)  # Vector((0.02784164994955063, 0.10910069942474365, 0.9936408400535583))
    camz = mathutils.Vector([0, 0, -1])  # cam to look point
    camz.rotate(cam.rotation_euler)  # Vector((0.1062391996383667, 0.9880731105804443, -0.1114661917090416))
    return camx, camy, camz


def draw_mon(monpos, monx, mony, monrect):
    verts = [
        cursor_to_mon(np.array([-1, -1]), monpos, monx, mony, monrect),
        cursor_to_mon(np.array([-1, 1]), monpos, monx, mony, monrect),
        cursor_to_mon(np.array([1, 1]), monpos, monx, mony, monrect),
        cursor_to_mon(np.array([1, -1]), monpos, monx, mony, monrect),
        cursor_to_mon(np.array([-1, -1]), monpos, monx, mony, monrect),
    ]
    line(verts)


def draw_eyeline(monpos, eye_center):
    conelen = (eye_center - monpos).magnitude
    bpy.ops.mesh.primitive_cone_add(radius1=10/1000, depth=conelen)
    cone = bpy.context.object
    trans_mat = mathutils.Matrix.Translation(eye_center + Vector([0, 0, 1]) * conelen / 2)
    conetip = eye_center + Vector([0, 0, 1]) * conelen / 2 + Vector([0, 0, 1]) * conelen / 2
    rot_mat = get_rotation_to_target(conetip, monpos, eye_center)
    cone.matrix_world = rot_mat @ trans_mat @ cone.matrix_world
    cone.hide_render = True
    return cone


def fill_empty_pixels(image1, image2):
    # Ensure the images have the same size
    if image1.size[0] != image2.size[0] or image1.size[1] != image2.size[1]:
        print("Error: Images must have the same size.")
        return

    pixels1 = np.array(image1.pixels).reshape(image1.size[0], image1.size[1], 4)
    pixels2 = np.array(image2.pixels).reshape(image2.size[0], image2.size[1], 4)
    empty1 = pixels1[:, :, 3] == 0
    pixels1[empty1] = pixels2[empty1]
    image1.pixels.foreach_set([float(v) for v in pixels1.ravel()])


def scale(obj, scale_factor, center=Vector([0, 0, 0])):
    translation_to_origin = mathutils.Matrix.Translation(-center)
    scale_matrix = mathutils.Matrix.Scale(scale_factor, 4)
    translation_back = mathutils.Matrix.Translation(center)
    final_matrix = translation_back @ scale_matrix @ translation_to_origin
    obj.matrix_world = final_matrix @ obj.matrix_world


def scale_scene(head, cam, deyes_target):
    _, leye_center, reye_center = get_eyes(head)
    deyes_model = (reye_center - leye_center).magnitude
    scale_factor = deyes_target / deyes_model

    scale(head, scale_factor)
    scale(cam, scale_factor)


def get_rotation(angle_rad, axis, center=Vector([0, 0, 0])):
    rotation_matrix = mathutils.Matrix.Rotation(angle_rad, 4, axis)
    translation_to_origin = mathutils.Matrix.Translation(-center)
    translation_back = mathutils.Matrix.Translation(center)
    final_matrix = translation_back @ rotation_matrix @ translation_to_origin
    return final_matrix


def cursor_to_mon(cur, monpos, monx, mony, monrect):
    '''
    Expect cursor to be a 2d np.array.
    Expect cursor x and y to be in [-1..1] range.
    Return point on monitor that corresponds to cursor xy.
    #              __________. monrect[1]
    #             |     y    |
    #             |     o x  |
    #  monrect[0] .__________|
    '''
    mon_w, mon_h = monrect[1] - monrect[0]
    mon_topl = monpos + monx * monrect[0, 0] + mony * monrect[0, 1]
    cur = (np.array(cur) + 1) / 2  # convert to [0..1] range
    p = mon_topl + monx * mon_w * cur[0] + mony * mon_h * cur[1]
    return p


def get_rotation_to_target(point, target, center=Vector([0, 0, 0])):
    '''
    Return how much rotation is needed to turn the point around center to be aligned with the target.
    '''
    line_direction = point - center
    target_direction = target - center
    line_direction.normalize()
    target_direction.normalize()

    rotation_axis = line_direction.cross(target_direction)
    dot_product = line_direction.dot(target_direction)
    rotation_angle = math.acos(min(dot_product, 1))
    rotation_quat = mathutils.Quaternion(rotation_axis, rotation_angle)

    rotation_matrix = mathutils.Matrix.Rotation(rotation_quat.angle, 4, rotation_quat.axis)
    translation_to_origin = mathutils.Matrix.Translation(-center)
    translation_back = mathutils.Matrix.Translation(center)
    final_matrix = translation_back @ rotation_matrix @ translation_to_origin
    return final_matrix


def get_look_at_cursor_mat(eyes_center0, mon0, monx, mony, monz, monrect, offset_monxyz, tilt_rad, cur):
    '''
    Expect desired eyes offset to be provided in monitor coordinate system.
    Usage of returned mat: obj.matrix_world = mat @ obj.matrix_world.
    '''
    tilt_mat = get_rotation(tilt_rad, eyes_center0-mon0, eyes_center0)

    offset = monx * offset_monxyz[0] + mony * offset_monxyz[1] + monz * offset_monxyz[2]
    translation_mat = mathutils.Matrix.Translation(offset)

    target = cursor_to_mon(cur, mon0, monx, mony, monrect)
    rotation_mat = get_rotation_to_target(mon0+offset, target, eyes_center0+offset)

    final_mat = rotation_mat @ translation_mat @ tilt_mat
    return final_mat


def gen_orientations_ordered(eyes_center, monpos, monx, mony, monz, monrect):
    span = 50 / 1000  # m
    tilt_span = math.radians(10)
    for x in np.linspace(-1, 1, 2):
        for y in np.linspace(-1, 1, 2):
            for k in np.linspace(-span, span, 2):
                for i in np.linspace(-span, span, 2):
                    for j in np.linspace(-span, span, 2):
                        for t in np.linspace(-tilt_span, tilt_span, 2):
                            mat = get_look_at_cursor_mat(eyes_center,
                                                         monpos, monx, mony, monz, monrect, [i, j, k], t, [x, y])
                            yield mat, x, y, i, j, k, t


def gen_orientations(eyes_center, monpos, monx, mony, monz, monrect):
    span = 50 / 1000  # m
    tilt_span = math.radians(10)
    while True:
        i = np.random.uniform(-span, span)
        j = np.random.uniform(-span, span)
        k = np.random.uniform(-span, span)
        t = np.random.uniform(-tilt_span, tilt_span)
        x = np.random.uniform(-1, 1)
        y = np.random.unifomr(-1, 1)
        mat = get_look_at_cursor_mat(eyes_center, monpos, monx, mony, monz, monrect, [i, j, k], t, [x, y])
        yield mat, x, y, i, j, k, t


def bpyimage_to_rgb(rendered):
    def linear_to_srgb(value):
        if value <= 0.0031308:
            return value * 12.92
        return 1.055 * (value ** (1.0 / 2.4)) - 0.055

    img = np.array(rendered.pixels).reshape(rendered.size[1], rendered.size[0], rendered.channels)
    img = np.vectorize(linear_to_srgb)(img)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = img[::-1, :, [0, 1, 2]]  # RGB, cv2 expects BGR
    return img


def render(cam, filename):
    bpy.data.scenes[0].camera = cam
    bpy.ops.render.render()
    rendered = bpy.data.images['Viewer Node']
    datadirpath = '/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data'
    # rendered.save_render(f'{datadirpath}/renders/render.png')
    rgb = bpyimage_to_rgb(rendered)
    bgr = rgb[:, :, ::-1]
    cv2.imwrite(f'{datadirpath}/renders/{filename}', bgr)


def line(verts):
    import bpy
    import bmesh
    from bpy_extras.object_utils import AddObjectHelper, object_data_add

    edges = [*zip(range(len(verts)-1), range(1, len(verts)))]
    mesh = bpy.data.meshes.new(name="line")
    mesh.from_pydata(verts, edges, faces=[])
    object_data_add(bpy.context, mesh)


def clear():
    while len(bpy.data.objects) > 0:
        bpy.data.objects.remove(bpy.data.objects[0])
    while len(bpy.data.collections) > 0:
        bpy.data.collections.remove(bpy.data.collections[0])
    while len(bpy.data.scenes) > 1:
        bpy.data.scenes.remove(bpy.data.scenes[0])
    bpy.data.scenes[0].name = 'Scene'
    while len(bpy.data.cameras) > 0:
        bpy.data.cameras.remove(bpy.data.cameras[0])
    while len(bpy.data.meshes) > 0:
        bpy.data.meshes.remove(bpy.data.meshes[0])
    for im in bpy.data.images:
        if im.users == 0:
            bpy.data.images.remove(im)


def create_head_from_image(image_filepath):
    clear()
    bpy.ops.keentools_fb.add_head()
    bpy.data.scenes['Scene'].keentools_fb_settings.heads[0].auto_focal_estimation = False
    bpy.data.scenes['Scene'].keentools_fb_settings.heads[0].focal = 17.5  # mm

    bpy.ops.keentools_fb.open_multiple_filebrowser(files=({'name': image_filepath},), headnum=0)
    bpy.data.scenes['Scene'].keentools_fb_settings.heads[0].cameras[0].auto_focal_estimation = False
    bpy.data.scenes['Scene'].keentools_fb_settings.heads[0].cameras[0].focal = 17.5  # mm

    override = bpy.context.copy()
    override['area'] = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
    override['region'] = next(region for region in override['area'].regions if region.type == 'WINDOW')
    with bpy.context.temp_override(**override):
        #        bpy.ops.keentools_fb.select_camera(0, 0)
        res = bpy.ops.keentools_fb.pickmode_starter()

    bpy.ops.keentools_fb.tex_selector(0)
    bpy.ops.keentools_fb.show_tex()

    deyes = 65 / 1000  # m
    head = bpy.data.objects['FBHead']
    cam = bpy.data.objects['fbCamera']
    scale_scene(head, cam, deyes)
    return head, cam


def draw_aux(monpos, monx, mony, monrect, reye_center, leye_center):
    draw_mon(monpos, monx, mony, monrect)
    coner = draw_eyeline(monpos, reye_center)
    conel = draw_eyeline(monpos, leye_center)
    return conel, coner


def augment_image(image_filepath):
    fname = os.path.basename(image_filepath).rsplit('.', maxsplit=1)[0]
    head, cam = create_head_from_image(image_filepath)
    camx, camy, camz = get_cam_axes(cam)
    monpos, monx, mony, monz = cam.location, camx, camy, camz
    eyes_center, reye_center, leye_center = get_eyes(head)
    monrect = np.array([[-300, -250], [300, 350]]) / 1000  # m
    eye1, eye2 = draw_aux(monpos, monx, mony, monrect, reye_center, leye_center)

    mat0s = [head.matrix_world.copy(), eye1.matrix_world.copy(), eye2.matrix_world.copy()]
    orientations = gen_orientations_ordered(eyes_center, monpos, monx, mony, monz, monrect)
    for iter in range(64):
        mat, x, y, i, j, k, t = next(orientations)
        for mat0, obj in zip(mat0s, [head, eye1, eye2]):
            obj.matrix_world = mat @ mat0
        filename = f'{fname} {iter} [{x:.4f} {y:.4f}] {i*1000:.0f} {j*1000:.0f} {k*1000:.0f} {math.degrees(t):.1f}.jpg'
        render(cam, filename)
        iter += 1


def main(image_filepaths):
    for imfp in image_filepaths:
        augment_image(imfp)


image_filepaths = [
    '/home/anatoly/my_photo-13.jpg',
    '/home/anatoly/my_photo-14.jpg',
]

images_sides = [
    '/home/anatoly/my_photo-24.jpg',
    '/home/anatoly/my_photo-23.jpg',
    '/home/anatoly/my_photo-22.jpg',
    '/home/anatoly/my_photo-21.jpg',
]


head = bpy.data.objects['FBHead']
cam = bpy.data.objects['fbCamera']
