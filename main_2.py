import cv2
import numpy as np
import copy
import os
import glob
from datetime import datetime
from tracker import Tracker


class AreaOfInterest(object):
    def __init__(self):
        self.top_left = 240
        self.top_right = 360
        self.bottom_left = 250
        self.bottom_right = 480


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


if __name__ == '__main__':
    # The one I first used for testing; after staring at it so much, I've grown attached to this road :3
    FPS = 24
    ROAD_DIST_METERS = 20
    HIGHWAY_SPEED_LIMIT = 65
    area = AreaOfInterest()
    # Initial background subtractor and text font
    fgbg = cv2.createBackgroundSubtractorMOG2()
    font = cv2.FONT_HERSHEY_PLAIN

    centers = []

    # y-cooridinate for speed detection line
    Y_THRESH_TOP = 200
    Y_THRESH_BOTTOM = 350

    object_area = 1600
    blob_min_width_near = 10
    blob_min_height_near = 10

    frame_start_time = None

    # Create object tracker
    tracker = Tracker(40, 3, 2, 1)

    cap = cv2.VideoCapture("movie.mp4")
    current_frame = 0
    temp_average_speeds = 0
    total_speed = 0
    total_vehicle = 0

    while True:
        current_frame += 1

        average_speed = temp_average_speeds

        centers = []
        frame_start_time = datetime.utcnow()
        ret, frame = cap.read()

        pts1 = np.float32(
            [[area.top_left, Y_THRESH_TOP], [area.top_right, Y_THRESH_TOP], [area.bottom_left, Y_THRESH_BOTTOM],
             [area.bottom_right, Y_THRESH_BOTTOM]])

        pts2 = np.float32(
            [(area.top_left, Y_THRESH_TOP), (area.top_right, Y_THRESH_TOP), (area.bottom_right, Y_THRESH_BOTTOM),
             (area.bottom_left, Y_THRESH_BOTTOM)])
        #
        # rect = order_points(pts1)
        # (tl, tr, br, bl) = rect

        # widthA = np.sqrt(((area.bottom_right - area.bottom_left) ** 2) + ((Y_THRESH_BOTTOM - Y_THRESH_BOTTOM) ** 2))
        # widthB = np.sqrt(((area.top_right - area.top_left) ** 2) + ((Y_THRESH_TOP - Y_THRESH_TOP) ** 2))
        # maxWidth = max(int(widthA), int(widthB))

        # heightA = np.sqrt(((area.top_right - area.bottom_right) ** 2) + ((Y_THRESH_TOP - Y_THRESH_BOTTOM) ** 2))
        # heightB = np.sqrt(((area.top_left - area.bottom_left) ** 2) + ((Y_THRESH_TOP - Y_THRESH_BOTTOM) ** 2))
        # maxHeight = max(int(heightA), int(heightB))

        # dst = np.array([
        #     [0, 0],
        #     [maxWidth - 1, 0],
        #     [maxWidth - 1, maxHeight - 1],
        #     [0, maxHeight - 1]], dtype="float32")

        # M = cv2.getPerspectiveTransform(pts, dst)
        # bev = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        bev = cv2.warpPerspective(frame, matrix, (430, 860))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        # Perform some Morphological operations to remove noise
        # kernel = np.ones((4, 4), np.uint8)
        # kernel_dilate = np.ones((5, 5), np.uint8)
        # opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # dilation = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_dilate)

        _, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find centers of all detected objects
        found_y = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            found_y = y
            if Y_THRESH_TOP - 30 < y < Y_THRESH_BOTTOM + 35 and area.top_left < x < area.top_right:
                countour_area = w * h
                # if w >= blob_min_width_near and h >= blob_min_height_near:
                #     center = np.array([[x + w / 2], [y + h / 2]])
                #     centers.append(np.round(center))
                #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if countour_area > object_area:
                    center = np.array([[x + w / 2], [y + h / 2]])
                    centers.append(np.round(center))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if centers:
            tracker.update(centers, current_frame, found_y)

            for vehicle in tracker.tracks:
                if len(vehicle.trace) > 1:
                    for j in range(len(vehicle.trace) - 1):
                        # Draw trace line

                        x1 = vehicle.trace[j][0][0]
                        y1 = vehicle.trace[j][1][0]
                        x2 = vehicle.trace[j + 1][0][0]
                        y2 = vehicle.trace[j + 1][1][0]

                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

                    try:

                        trace_i = len(vehicle.trace) - 1

                        trace_x = vehicle.trace[trace_i][0][0]
                        trace_y = vehicle.trace[trace_i][1][0]

                        # Check if tracked object has reached the speed detection line
                        if trace_y < Y_THRESH_TOP + 25:
                            duration = float(current_frame - vehicle.first_detected_frame) / float(FPS)

                            if 10 < (current_frame - vehicle.first_detected_frame):
                                # print duration
                                # print float(current_frame - vehicle.first_detected_frame), ' ', vehicle.track_id
                                speed = (float(ROAD_DIST_METERS) / duration) * (3600 / 1000)
                                # print speed
                                total_speed += speed
                                total_vehicle += 1
                                tracker.tracks.pop(vehicle)

                        cv2.putText(frame, 'ID: ' + str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                    except:
                        pass

        # Draw line used for speed detection
        cv2.line(frame, (area.top_left, Y_THRESH_TOP + 10), (area.top_right, Y_THRESH_TOP), (255, 0, 0), 2)
        cv2.line(frame, (area.bottom_left, Y_THRESH_BOTTOM + 27), (area.bottom_right, Y_THRESH_BOTTOM), (255, 0, 0), 2)

        # Draw Circle
        cv2.circle(frame, (area.top_left, Y_THRESH_TOP + 10), 2, (0, 0, 255), 1)
        cv2.circle(frame, (area.top_right, Y_THRESH_TOP), 2, (0, 0, 255), 1)
        cv2.circle(frame, (area.bottom_left, Y_THRESH_BOTTOM + 27), 2, (0, 0, 255), 1)
        cv2.circle(frame, (area.bottom_right, Y_THRESH_BOTTOM), 2, (0, 0, 255), 1)

        if total_vehicle != 0:
            average_speed = total_speed / total_vehicle
            temp_average_speeds = average_speed

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, ("Kecepatan : {0:.2f} KM/Jam".format(average_speed)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1)
        cv2.putText(frame, ("Jumlah Kendaraan : {} ".format(0)), (400, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1)

        # Display all images
        cv2.imshow('original', frame)
        # cv2.imshow('opening/dilation', bev)
        # cv2.imshow('background subtraction', fgmask)

        # Quit when escape key pressed
        if cv2.waitKey(5) == 27:
            break

        # Sleep to keep video speed consistent
        # time.sleep(1.0 / FPS)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # remove all speeding_*.png images created in runtime
    for file in glob.glob('speeding_*.png'):
        os.remove(file)
