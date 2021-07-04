from comp_only_code.ai_processing import *

# PIXEL_SPACING = 0.160
QUAD_DIVISER_VAL = 3
KERNEL_SIZE = 40


def locate_fracture_spine(corner_points, drawing=False, drawing_image='', drawing_mask=''):
    if drawing:
        draw_box(drawing_image, corner_points, color=color_red)
        draw_box(drawing_mask, corner_points, color=color_red)


def front_back_fracture_classifier(front_height, back_height):
    if (1 - (front_height/back_height))*20 > FRACTURE_THRESHOLD:
        return True
    else:
        return False


def get_spine_heights(corner_pnts, ravel_pnts, org_img_shape, pixel_spacing, drawing=False, drawing_image='', drawing_mask=''):
    '''  From the quadrant points and ravel (polygon points),
    1. get heights from each ends,
    2. draw boxes and height lines,
    3. write text for the height measurement,
    4. and conduct a list containing important properties for each of the spines
    '''
    spine_properties = []
    index_num = 0
    first_spine_exclude, last_spine_exclude = False, False
    last_spine_index = len(corner_pnts)-1
    print("GSH 1")
    for i, ((corner_points, mid_points), ravel_corner_points) in enumerate(zip(corner_pnts, ravel_pnts)):
        if (corner_points[0][1] <= -20) or (corner_points[1][1] <= -20) or \
                        (corner_points[3][1] >= org_img_shape[0]) or (corner_points[2][1] >= org_img_shape[0]):
            if i == 0:
                first_spine_exclude = True
            if i == last_spine_index:
                last_spine_exclude = True
            continue
        print("GSH 2")
        # Allocate each ravel points to one of the quadrants
        quad_points = split_points_to_quads(corner_points, mid_points, ravel_corner_points,
                                            drawing=drawing,
                                            drawing_image=drawing_image,
                                            drawing_mask=drawing_mask)
        print("GSH 3")
        # Find the closest ravel point in each of the quadrant from the box corner points.
        lu_point, ld_point, ru_point, rd_point, left_height, right_height = \
            ultimate_corner_point_by_distance(corner_points, quad_points, ravel_corner_points, pixel_spacing,
                                              drawing=drawing,
                                              drawing_image=drawing_image, drawing_mask=drawing_mask)
        print("GSH 4")
        # draw green box for the quadurant around the spine
        # draw_quad_box(corner_points, mid_points, drawing_image=drawing_image, drawing_mask=drawing_mask)
        # spine_properties.append([index_num, corner_points, front_back_fracture_classifier(left_height, right_height),
        spine_properties.append([index_num, corner_points, front_back_fracture_classifier(left_height, right_height),
                                 left_height, right_height, 0, 0])
        print("GSH 5")
        index_num += 1
    return spine_properties, first_spine_exclude, last_spine_exclude


def mask_opening(mask_image, size, pixel_spacing):
    kernel = np.ones((size, size), np.uint8)
    opening = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)
    return opening


def find_closest_index(spine_properties, index):
    max_index = len(spine_properties)-1
    closest_prev_idx, closest_next_idx = -1, -1
    for i in range(max_index+1):
        if index-i-1 <= 0:
            break
        if not(spine_properties[index-i-1][2]):
            closest_prev_idx = index-i-1
            break

    for i in range(max_index+1):
        if index+i+1 >= max_index:
            break
        if not(spine_properties[index+i+1][2]):
            closest_next_idx = index+i+1
            break

    if (closest_prev_idx == -1) and (closest_next_idx == -1):
        return -1
    elif closest_next_idx == -1:
        return closest_prev_idx
    elif closest_prev_idx == -1:
        return closest_next_idx
    elif (closest_next_idx-index) < (index-closest_prev_idx):
        return closest_next_idx
    elif (closest_next_idx-index) > (index-closest_prev_idx):
        return closest_prev_idx
    else:
        return closest_prev_idx


def get_spine_comp_ratio(spine_properties, fst_spine_exclude, lst_spine_exclude, drawing=False, drawing_image='', drawing_mask=''):
    '''    from the spine properties,
    1. test if it has a fracture by calculating the compression rate,
    2. if spine seems to have a fracture draw line on the result image.
    (side note, this is only tested for the left hand side of the spine.)
    '''

    # TODO: IF UPPER AND LOWER SPINES ARE ALL IDENTIFIED AS CF, find the closest that is not CF and recalculate.
    max_index, num_of_spine = len(spine_properties)-1, len(spine_properties)
    new_fracture_found, fracture_idx_history, current_fracture_index = True, [], []
    while new_fracture_found:
        print("====================norm_order=======================")
        _ = [print(sp) for sp in spine_properties]
        for (i, _, cf_bool, lh, rh, l_rate, r_rate) in spine_properties:
            print("When comparing " + str(i))
            new_fracture_found = False

            # Decide whether to exclude the current spine from analysis
            if (not fst_spine_exclude) and (i == 0):
                continue
            if (not lst_spine_exclude) and (i == max_index):
                continue
            if (lh == 0) or (rh == 0):
                continue

            no_prev = True if (i == 0) or (spine_properties[i - 1][3] == 0) else False
            no_next = True if (i == num_of_spine - 1) or (spine_properties[i + 1][3] == 0) else False

            if no_prev and no_next:
                print("Case 0")
                pass
            elif (i == 0) or (i == num_of_spine-1) or no_prev or no_next:
                print("Case 1")
                closest_index = find_closest_index(spine_properties, i)
                spine_properties[i][5] = 100 - (lh / (spine_properties[closest_index][3])) * 100
            else:
                if (not spine_properties[i-1][2]) and (not spine_properties[i+1][2]):
                    print("Case 2")
                    spine_properties[i][5] = 100 - (lh / ((spine_properties[i - 1][3] + spine_properties[i + 1][3]) / 2)) * 100
                else:
                    print("Case 3")
                    closest_index = find_closest_index(spine_properties, i)
                    print("Closest index: " + str(closest_index))
                    spine_properties[i][5] = 100 - (lh / (spine_properties[closest_index][3])) * 100

            if fracture_idx_history.count(i) > 10:
                spine_properties[i][2] = True
                if i not in current_fracture_index:
                    current_fracture_index.append(i)
            elif spine_properties[i][5] > FRACTURE_THRESHOLD:
                spine_properties[i][2] = True
                print(spine_properties[i])
                if i not in current_fracture_index:
                    current_fracture_index.append(i)
                    fracture_idx_history.append(i)
                    new_fracture_found = True
                    break
            if fracture_idx_history.count(i) > 10:
                pass
            elif spine_properties[i][5] < FRACTURE_THRESHOLD:
                spine_properties[i][2] = False
                print(spine_properties[i])
                if i in current_fracture_index:
                    current_fracture_index.remove(i)
                    new_fracture_found = True
                    break

            print(spine_properties[i])


    return spine_properties


def comp_1st_prep(image):
    resized_image, crop_coords = resizing_image(image, 512)
    normalized_image = cv2.normalize(resized_image, 0, 255, norm_type=cv2.NORM_MINMAX)
    normalized_image = normalized_image * (1./255.)
    return normalized_image, crop_coords


def get_spine_properties(original_image, mask_image, pixel_spacing, debug_image_dir, file_name):
    mask_image = mask_image.astype(np.uint8)
    print("GSP 1")

    # Find contour and filter them by comparing the size of the contours.
    # If the size of the contour is 4 times less than the average size of the contour, filter-out. (For small spines)
    # (For combined spines)
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    avg_contours_area = average([cv2.contourArea(cnt) for cnt in contours])
    exception_abs_area_bool = False
    print("GSP 2")
    if avg_contours_area > 3000:
        exception_abs_area_bool = True
    contours_idx_area = [(i, cv2.contourArea(cnt)) for i, cnt in enumerate(contours)]

    print("GSP 3")
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > avg_contours_area / 4]
    exception_area = [cv2.contourArea(cnt) > (avg_contours_area*2) for cnt in contours]
    exception_rel_area_bool = any(exception_area)
    
    print("GSP 4")
    pms_patch_extract_corners = extract_patch_corner_points(contours)
    # [lu, ru, ld, rd]
    arranged_corner_points = [rearange_corner_points(ppec) for ppec in pms_patch_extract_corners]
    arranged_corner_points = [give_border_to_corner_points(acp) for acp in arranged_corner_points]

    print("GSP 5")
    arranged_corner_mid_points = add_midpoints(arranged_corner_points, QUAD_DIVISER=QUAD_DIVISER_VAL)
    arranged_ravel_points = [all_points_to_tuple_points(cv2.approxPolyDP(cnt, 0.009*cv2.arcLength(cnt, True), True).ravel())
                             for cnt in contours]

    # Find ultimate corner points, heights and create spine properties
    print("GSP 6")
    spine_properties, fst_spine_exclude, lst_spine_exclude\
        = get_spine_heights(arranged_corner_mid_points, arranged_ravel_points, original_image.shape, pixel_spacing)
    spine_properties = get_spine_comp_ratio(spine_properties, fst_spine_exclude, lst_spine_exclude)

    """
    print("GSP 7")
    for (i, corner_points, cf_bool, lh, rh, l_rate, r_rate) in spine_properties:
        if cf_bool:
            draw_box(mid_point_exp_img, corner_points, color=color_red)
            draw_box(mid_point_exp_msk, corner_points, color=color_red)
    """
    exception_area_bool = exception_abs_area_bool and exception_rel_area_bool
    print("GSP 8")

    return spine_properties, exception_area_bool

def comp_1st_post(original_image, pred_image, crop_coords, original_shape, pixel_spacing, debug_image_dir, file_name):
    exception = ''

    # RESTORE PRED_IMAGE to ORIGINAL IMAGE SIZE
    pred_image = pred_image * 255
    restored_pred = cv2.resize(crop(pred_image, crop_coords), (original_shape[:2][::-1]))

    # GET CROPPING POINT as the coordinate of the resized image.
    normalized_image = cv2.normalize(restored_pred, 0, 255, norm_type=cv2.NORM_MINMAX)
    _, pred_mask_binary = cv2.threshold(normalized_image, 128, 255, cv2.THRESH_BINARY)

    # pred_mask_binary = mask_opening(pred_mask_binary, KERNEL_SIZE, pixel_spacing)     # For coarse-ing the dilated spine area
    cropped_spine_properties, exception_bool = get_spine_properties(original_image, pred_mask_binary, pixel_spacing, debug_image_dir, file_name)

    result_img = cv2.cvtColor(original_image.copy(), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(pred_mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    highest_l_rate = 0
    fracture_exists = False
    for (i, (lup, rup, ldp, rdp), cf_bool, lh, rh, l_rate, r_rate) in cropped_spine_properties:
        print("Spine properties: " + str(cropped_spine_properties[i]))
        if cf_bool:
            fracture_exists = True
            draw_box(result_img, (lup, rup, ldp, rdp), color=color_blue)
            cv2.putText(result_img, str(round(l_rate, 2)), (int(math.floor(ldp[0] - 180)), int(math.floor(ldp[1]-50))),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color_red, 3)
        if l_rate > highest_l_rate:
            highest_l_rate = l_rate
    if not fracture_exists:
        highest_l_rate = 0

    if exception_bool:
        exception = 'AREA'
        cv2.putText(result_img, 'Area Exception Case', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_red, 3)
    return highest_l_rate, result_img, exception


def ai_analyze_compression(image, pixel_spacing, file_name, model_1):
    prep_img_1, crop_coords = comp_1st_prep(image)
    cf_1_pred = model_1.predict(prep_img_1.reshape((1, 512, 512, 1))).reshape((512, 512, 1))
    highest_comp_rate, result_image, exception = comp_1st_post(image, cf_1_pred, crop_coords, image.shape, pixel_spacing,
                                                               '', file_name)
    return highest_comp_rate, result_image, exception
