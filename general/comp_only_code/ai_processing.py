import math
import numpy as np
import cv2
import sys

valid_area_threshold = 500
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_green = (0, 255, 0)
color_yellow = (0, 255, 255)
color_purple = (255, 0, 255)
color_bg = (255, 255, 0)
color_black = (0, 0, 0)
color_gray = (150, 150, 150)
colors = [color_red, color_blue, color_green, color_purple, color_bg]

radius = 5
thickness = 5

FRACTURE_THRESHOLD = 20
border_np_array = [np.array([-10, -10], dtype=np.float32), np.array([10, -10], dtype=np.float32),
                   np.array([-10, 10], dtype=np.float32), np.array([10, 10], dtype=np.float32)]

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    return clahe.apply(image)


def resizing_image(image, height_and_width, color='white'):
    """ Applies image resizing with the size of the given length on an image and returns
    the re-sized image with fixed size.

    Args:
        image (image): numpy array of the image
        height_and_width (int): length of both height and width to be re-sized

    Returns:
        (numpy.ndarray): numpy representation of re-sized image in size of (height_and_width, height_and_width)

    Example:
        > > > image = cv2.imread('1035642_.svs')
        > > > resizing_image(image, 1024)
        (numpy.ndarray)
    """
    try:
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        if image.shape == (height_and_width, height_and_width):
            return image, (0, 0, height_and_width, height_and_width)
        elif (image.shape[0] < height_and_width) and (image.shape[1] < height_and_width):
            size = image.shape[:2]
            delta_w = height_and_width - size[1]
            delta_h = height_and_width - size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0), (left, top, height_and_width-right, height_and_width-bottom)
        else:
            old_size = image.shape[:2]
            ratio = float(height_and_width) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            im = cv2.resize(image, (new_size[1], new_size[0]))
            delta_w = height_and_width - new_size[1]
            delta_h = height_and_width - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0), (left, top, height_and_width-right, height_and_width-bottom)
    except Exception as error:
        raise Exception("Exception occurred while resizing the image: " + str(error))


def resizing_image_wh(image, width, height):
    """ Applies image resizing with the size of the given length on an image and returns
    the re-sized image with fixed size.
    Args:
        image (numpy.ndarray): numpy array of the image
        height_and_width (int): length of both height and width to be re-sized
    Returns:
        (numpy.ndarray): numpy representation of re-sized image in size of (height_and_width, height_and_width)
    Example:
        > > > image = cv2.imread('1035642_.svs')
        > > > resizing_image(image, 1024)
        (numpy.ndarray)
    """
    try:
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        im = cv2.resize(image, (width, height))
        return im
    except Exception as error:
        raise Exception("Exception occurred while resizing the image: " + str(error))


def get_x_point(coord_tuple):
    return coord_tuple[0]


def get_y_point(coord_tuple):
    return coord_tuple[1]


def shape(image):
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    return image.shape


def give_border(coords, border, shp, logging=False):
    coords = coords[0] - border, coords[1] - border, coords[2] + border, coords[3] + border

    if coords[0] < 0:
        coords = (0, coords[1], coords[2], coords[3])
    if coords[1] < 0:
        coords = (coords[0], 0, coords[2], coords[3])
    if coords[2] > shp[1]:
        coords = (coords[0], coords[1], shp[1], coords[3])
    if coords[3] > shp[0]:
        coords = (coords[0], coords[1], coords[2], shp[0])

    return coords


def crop(image, coords, border=0, logging=False):
    # coords should be a tuple with (x1, y1, x2, y2)
    left, top, right, bottom = give_border(coords, border, shape(image), logging=logging)
    if isinstance(image, np.ndarray):
        return image[top: bottom, left: right]
    else:
        return image.crop((left, top, right, bottom))


def morphology_operation(binary_dataset_mask):
    kernel_size = 5
    img_dilate = cv2.dilate(binary_dataset_mask, np.array([1] * int(math.pow(kernel_size, 2)))
                            .reshape(kernel_size, kernel_size), iterations=3)
    img_erode = cv2.erode(img_dilate, np.array([1] * int(math.pow(kernel_size, 2)))
                          .reshape(kernel_size, kernel_size), iterations=3)

    kernel_size = 25
    img_dilate = cv2.dilate(img_erode, np.array([1] * int(math.pow(kernel_size, 2)))
                            .reshape(kernel_size, kernel_size), iterations=3)

    return img_dilate


def average(lst):
    return sum(lst) / len(lst)


def calc_slope(start_point, end_point):
    return (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])


def overlay_mask(image, mask):
    redImg = np.zeros(image.shape, image.dtype)
    redImg[:, :] = (0, 0, 122)
    redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
    cv2.addWeighted(redMask, 1, image, 1, 0, image)
    return image


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return math.degrees(angle_radians)


def angle_of_line(point1, point2):
    angle = np.rad2deg(np.arctan2(point1[1] - point2[1], point1[0] - point2[0]))
    return angle


def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    #     return ang + 360 if ang < 0 else ang
    #     return ang
    return ang if ang <= 180 else 360 - ang


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def cal_cobb_angle(up_apical_lu, up_apical_ru, down_apical_ld, down_apical_rd):
    intersect_point = line_intersection(
        (up_apical_lu, up_apical_ru), (down_apical_ld, down_apical_rd)
    )
    cobb_angle = np.abs(get_angle(up_apical_lu, intersect_point, down_apical_ld))
    return cobb_angle


def extract_apical_points(angles):
    angles = np.array(angles)
    apical_idx = [np.argmax(angles[:, -1]), np.argmin(angles[:, -1])]
    up_apical_idx = min(apical_idx)
    down_apical_idx = max(apical_idx)
    up_apical = angles[up_apical_idx]
    down_apical = angles[down_apical_idx]

    up_apical_lu, up_apical_ru = up_apical[:2]
    down_apical_ld, down_apical_rd = down_apical[2:4]

    return up_apical_lu, up_apical_ru, down_apical_ld, down_apical_rd


def get_avg_angle(element):
    return element[5]


def cal_extended_points(img_shape, pt1, pt2, extend_rate=1.5):
    # pt1 -> left side , pt2 -> right side
    # y = ax + b
    pt1, pt2 = np.array(pt1), np.array(pt2)
    distance = np.sqrt((pt1[1] - pt2[1]) ** 2 + (pt1[0] - pt2[0]) ** 2).astype(np.int32)
    each_extend = (extend_rate - 1) / 2
    each_distance = (distance * each_extend).astype(np.int32)
    max_x = img_shape[1]
    min_x = 0

    a = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0] + sys.float_info.epsilon)
    b = pt1[1] - a * pt1[0]

    # extended_pt1
    pt1[0] -= each_distance
    pt1[0] = max(min_x, pt1[0])
    pt1[1] = a * pt1[0] + b
    pt1 = pt1.astype(np.int32)

    # extended_pt2
    pt2[0] += each_distance
    pt2[0] = min(max_x, pt2[0])
    pt2[1] = a * pt2[0] + b
    pt2 = pt2.astype(np.int32)

    return tuple(pt1), tuple(pt2)


def calculate_angle(left, right):
    xDiff = right[0] - left[0]
    yDiff = right[1] - left[1]
    return math.degrees(math.atan2(yDiff, xDiff))

def calculate_distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def calculate_height(upper, lower):
    dist = math.sqrt((upper[0] - lower[0]) ** 2 + (upper[1] - lower[1]) ** 2)
    return dist


def all_points_to_tuple_points(point_list):
    tuple_point_list = []
    for i in range(int(len(point_list) / 2)):
        tuple_point_list.append((point_list[2 * i], point_list[2 * i + 1]))
    return tuple_point_list


def height_ordering(point_list):
    avg_height_and_points_list = []
    for pl in point_list:
        avg_height = sum([get_y_point(point) for point in pl]) / 4
        avg_height_and_points_list.append((pl, avg_height))
    avg_height_and_points_list.sort(key=get_y_point)
    return avg_height_and_points_list


def give_border_to_corner_points(corner_points_list):
    lu, ru, ld, rd = corner_points_list
    lu, ru, ld, rd = lu + border_np_array[0], ru + border_np_array[1], ld + border_np_array[2], rd + border_np_array[3]
    return lu, ru, ld, rd


def add_midpoints(spine_properties, QUAD_DIVISER):
    new_return_list = []
    for pl in spine_properties:
        lu, ru, ld, rd = tuple(pl)
        lm, um, rm, dm = (int(math.ceil(lu[0] + ld[0]) / 2), int(math.ceil((lu[1] + ld[1]) / 2))), \
                         (int(math.ceil(lu[0] + ru[0]) / 2), int(math.ceil(lu[1] + ru[1]) / 2)), \
                         (int(math.ceil(ru[0] + rd[0]) / 2), int(math.ceil(ru[1] + rd[1]) / 2)), \
                         (int(math.ceil(ld[0] + rd[0]) / 2), int(math.ceil(ld[1] + rd[1]) / 2))
        mm = ((int(math.ceil((lm[0] + rm[0]) / 2))), (int(math.ceil((um[1] + dm[1]) / 2))))

        if lu[1] > ru[1]:
            lum, rum = (int(math.ceil(lu[0] + (ru[0] - lu[0]) / QUAD_DIVISER)),
                        int(math.ceil(lu[1] - (lu[1] - ru[1]) / QUAD_DIVISER))), \
                       (int(math.ceil(ru[0] - (ru[0] - lu[0]) / QUAD_DIVISER)),
                        int(math.ceil(ru[1] + (lu[1] - ru[1]) / QUAD_DIVISER)))
            lmm, rmm = (int(math.ceil(lm[0] + (rm[0] - lm[0]) / QUAD_DIVISER)),
                        int(math.ceil(lm[1] - (lm[1] - rm[1]) / QUAD_DIVISER))), \
                       (int(math.ceil(rm[0] - (rm[0] - lm[0]) / QUAD_DIVISER)),
                        int(math.ceil(rm[1] + (lm[1] - rm[1]) / QUAD_DIVISER)))
            ldm, rdm = (int(math.ceil(ld[0] + (rd[0] - ld[0]) / QUAD_DIVISER)),
                        int(math.ceil(ld[1] - (ld[1] - rd[1]) / QUAD_DIVISER))), \
                       (int(math.ceil(rd[0] - (rd[0] - ld[0]) / QUAD_DIVISER)),
                        int(math.ceil(rd[1] + (ld[1] - rd[1]) / QUAD_DIVISER)))
        else:
            lum, rum = (int(math.ceil(lu[0] + (ru[0] - lu[0]) / QUAD_DIVISER)),
                        int(math.ceil(lu[1] + (ru[1] - lu[1]) / QUAD_DIVISER))), \
                       (int(math.ceil(ru[0] - (ru[0] - lu[0]) / QUAD_DIVISER)),
                        int(math.ceil(ru[1] - (ru[1] - lu[1]) / QUAD_DIVISER)))
            lmm, rmm = (int(math.ceil(lm[0] + (rm[0] - lm[0]) / QUAD_DIVISER)),
                        int(math.ceil(lm[1] + (rm[1] - lm[1]) / QUAD_DIVISER))), \
                       (int(math.ceil(rm[0] - (rm[0] - lm[0]) / QUAD_DIVISER)),
                        int(math.ceil(rm[1] - (rm[1] - lm[1]) / QUAD_DIVISER)))
            ldm, rdm = (int(math.ceil(ld[0] + (rd[0] - ld[0]) / QUAD_DIVISER)),
                        int(math.ceil(ld[1] + (rd[1] - ld[1]) / QUAD_DIVISER))), \
                       (int(math.ceil(rd[0] - (rd[0] - ld[0]) / QUAD_DIVISER)),
                        int(math.ceil(rd[1] - (rd[1] - ld[1]) / QUAD_DIVISER)))

        new_return_list.append((pl, (lm, um, rm, dm, lum, rum, lmm, rmm, ldm, rdm)))
    return new_return_list


def draw_quad_box(corner_points, mid_points, drawing_image='', drawing_mask=''):
    draw_box(drawing_image, corner_points)
    draw_mid_lines(drawing_image, mid_points)
    draw_box(drawing_mask, corner_points)
    draw_mid_lines(drawing_mask, mid_points)


def draw_box(drawing_image, corner_points, color=color_green):
    cv2.line(drawing_image, tuple(corner_points[0]), tuple(corner_points[1]), color, thickness)
    cv2.line(drawing_image, tuple(corner_points[1]), tuple(corner_points[3]), color, thickness)
    cv2.line(drawing_image, tuple(corner_points[2]), tuple(corner_points[0]), color, thickness)
    cv2.line(drawing_image, tuple(corner_points[3]), tuple(corner_points[2]), color, thickness)


def draw_mid_lines(drawing_image, mid_points, color=color_green):
    cv2.line(drawing_image, tuple(mid_points[4]), tuple(mid_points[8]), color, thickness)
    cv2.line(drawing_image, tuple(mid_points[5]), tuple(mid_points[9]), color, thickness)
    cv2.line(drawing_image, tuple(mid_points[0]), tuple(mid_points[6]), color, thickness)
    cv2.line(drawing_image, tuple(mid_points[2]), tuple(mid_points[7]), color, thickness)


def find_closest_point(candid_pnts, corner_pnt):
    closest_pnt, min_dist = 0, 100000
    for cd_pnt in candid_pnts:
        dist = calculate_distance(cd_pnt[0], cd_pnt[1], corner_pnt[0], corner_pnt[1])
        if dist < min_dist:
            closest_pnt, min_dist = tuple(cd_pnt), dist
    return closest_pnt


def ultimate_corner_point_by_distance(corner_points, quad_points, ravel_points, pixel_spacing,
                                      drawing=False, drawing_image='', drawing_mask=''):
    lu_quad_points, ru_quad_points, rd_quad_points, ld_quad_points = quad_points
    lu, ru, ld, rd = corner_points

    lu_final = find_closest_point(lu_quad_points, lu)
    lu_final = lu_final if lu_final != 0 else find_closest_point(ravel_points, lu)
    ld_final = find_closest_point(ld_quad_points, ld)
    ld_final = ld_final if ld_final != 0 else find_closest_point(ravel_points, ld)
    ru_final = find_closest_point(ru_quad_points, ru)
    ru_final = ru_final if ru_final != 0 else find_closest_point(ravel_points, ru)
    rd_final = find_closest_point(rd_quad_points, rd)
    rd_final = rd_final if rd_final != 0 else find_closest_point(ravel_points, rd)

    left_height, right_height = 0, 0
    if (lu_final != 0) and (ld_final != 0):
        left_height_pxl = calculate_height(lu_final, ld_final)
        left_height = round(left_height_pxl * float(pixel_spacing), 2)

        if drawing:
            cv2.line(drawing_image, lu_final, ld_final, color_red, thickness)
            cv2.line(drawing_mask, lu_final, ld_final, color_red, thickness)
            cv2.putText(drawing_image, str(left_height) + "mm", (lu_final[0] - 350, ld_final[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2, color_red, 3)
            cv2.putText(drawing_mask, str(left_height) + "mm", (lu_final[0] - 350, ld_final[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2, color_red, 3)

    if (ru_final != 0) and (rd_final != 0):
        right_height_pxl = calculate_height(ru_final, rd_final)
        right_height = round(right_height_pxl * float(pixel_spacing), 2)
        if drawing:
            cv2.line(drawing_image, ru_final, rd_final, color_red, thickness)
            cv2.line(drawing_mask, ru_final, rd_final, color_red, thickness)

    return lu_final, ld_final, ru_final, rd_final, left_height, right_height



def ultimate_corner_point_by_distance_v2(corner_points, pixel_spacing,
                                      drawing=False, drawing_image='', drawing_mask=''):
    lu, ru, ld, rd = corner_points

    left_height, right_height = 0, 0
    if (tuple(lu) != 0) and (tuple(ld) != 0):
        left_height_pxl = calculate_height(lu, ld)
        left_height = round(left_height_pxl * float(pixel_spacing), 2)

        if drawing:
            cv2.line(drawing_image, lu, ld, color_red, thickness)
            cv2.line(drawing_mask, lu, ld, color_red, thickness)
            cv2.putText(drawing_image, str(left_height) + "mm", (lu[0] - 350, ld[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2, color_red, 3)
            cv2.putText(drawing_mask, str(left_height) + "mm", (lu[0] - 350, ld[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2, color_red, 3)

    if (tuple(ru) != 0) and (tuple(rd) != 0):
        right_height_pxl = calculate_height(ru, rd)
        right_height = round(right_height_pxl * float(pixel_spacing), 2)
        if drawing:
            cv2.line(drawing_image, ru, rd, color_red, thickness)
            cv2.line(drawing_mask, ru, rd, color_red, thickness)

    return lu, ld, ru, rd, left_height, right_height



def split_points_to_quads(corner_points, mid_points, ravel_points, drawing=False, drawing_image='', drawing_mask=''):
    lu, ru, ld, rd = corner_points
    lm, um, rm, dm, lum, rum, lmm, rmm, ldm, rdm = mid_points

    lu_quadrant_cnt = np.array([lu, list(lum), list(lmm), list(lm)]).reshape((-1, 1, 2)).astype(np.int32)
    ru_quadrant_cnt = np.array([list(rum), ru, list(rm), list(rmm)]).reshape((-1, 1, 2)).astype(np.int32)
    rd_quadrant_cnt = np.array([list(rmm), list(rm), rd, list(rdm)]).reshape((-1, 1, 2)).astype(np.int32)
    ld_quadrant_cnt = np.array([list(lm), list(lmm), list(ldm), ld]).reshape((-1, 1, 2)).astype(np.int32)

    lu_quad_points, ru_quad_points, rd_quad_points, ld_quad_points = [], [], [], []
    for rcp in ravel_points:
        if cv2.pointPolygonTest(lu_quadrant_cnt, rcp, True) >= 0:
            lu_quad_points.append(list(rcp))
        elif cv2.pointPolygonTest(ru_quadrant_cnt, rcp, True) >= 0:
            ru_quad_points.append(list(rcp))
        elif cv2.pointPolygonTest(rd_quadrant_cnt, rcp, True) >= 0:
            rd_quad_points.append(list(rcp))
        elif cv2.pointPolygonTest(ld_quadrant_cnt, rcp, True) >= 0:
            ld_quad_points.append(list(rcp))

        if drawing:
            cv2.circle(drawing_image, rcp, radius, color_gray, thickness)
            cv2.circle(drawing_mask, rcp, radius, color_gray, thickness)
    return lu_quad_points, ru_quad_points, rd_quad_points, ld_quad_points


def rearange_corner_points(corner_points):
    order = np.argsort(corner_points, axis=0)

    lu_candi_idx = np.intersect1d(order[:, 0][:2], order[:, 1][:3])
    rd_candi_idx = np.intersect1d(order[:, 0][2:], order[:, 1][1:])
    ld_candi_idx = np.intersect1d(order[:, 0][:2], order[:, 1][1:])
    ru_candi_idx = np.intersect1d(order[:, 0][2:], order[:, 1][:3])

    lu_candi = np.array([corner_points[idx] for idx in lu_candi_idx])
    lu = lu_candi[lu_candi[:, 1].argsort()][0]

    rd_candi = np.array([corner_points[idx] for idx in rd_candi_idx])
    rd = rd_candi[rd_candi[:, 1].argsort()][-1]

    ld_candi = np.array([corner_points[idx] for idx in ld_candi_idx])
    ld = ld_candi[ld_candi[:, 1].argsort()][-1]

    ru_candi = np.array([corner_points[idx] for idx in ru_candi_idx])
    ru = ru_candi[ru_candi[:, 1].argsort()][0]

    return [lu, ru, ld, rd]


def extract_patch_corner_points(cnts):
    temp = []
    for candi in cnts[:]:
        temp.append(candi)

    cnts = temp
    box_points = []
    for cnt in cnts:
        rc = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rc)
        box_points.append(box)

    return box_points
