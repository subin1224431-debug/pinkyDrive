import cv2
import numpy as np
import requests
import time


# =========================================================
# Pinky Pro
# =========================================================
HOST = "http://192.168.4.1:8000"


# =========================================================
# 기본 주행 / P 제어
# =========================================================
BASE_SPEED = 25
KP = 0.12
MAX_SPEED = 40
SEARCH_SPEED = 18

# 회전 후 일반 곡선은 기존 KP 사용
# 황토색 박스 구간에서만 더 강하게 우회전
KP_BROWN = 0.22

# 황토색 구간에서 복원된 중심선 자체를
# 화면 오른쪽으로 조금 이동
BROWN_CENTER_SHIFT_PX = 28


# =========================================================
# 영상 처리
# =========================================================
ROI_START_RATIO = 0.58

LOWER_WHITE = np.array([0, 0, 175])
UPPER_WHITE = np.array([180, 75, 255])

LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 80])

# 황토색 시작값
LOWER_BROWN = np.array([5, 30, 70])
UPPER_BROWN = np.array([35, 220, 240])

MIN_AREA = 500
STEP = 15
MIN_WIDTH = 25
MAX_JUMP = 60
MAX_POINTS = 8

TARGET_INDEX = 2


# =========================================================
# 화살표 메우기
# =========================================================
ARROW_FILL_GAP_RATIO = 0.38


# =========================================================
# 화살표 검출 / 카운팅
# =========================================================
ARROW_MIN_AREA = 700
ARROW_CONFIRM_FRAMES = 3
ARROW_GONE_FRAMES = 4


# =========================================================
# 2번, 3번 화살표 이후 공통 동작
#
# 화살표가 화면에서 사라짐
# -> 2.8초 동안 중심선 추종
# -> 정지
# -> 오른쪽 약 50도 회전
# =========================================================
AFTER_ARROW_FOLLOW_TIME = 2.8

STOP_BEFORE_TURN_TIME = 0.30

TURN_SPEED = 22

# 기존 45도 약 0.62초 기준으로 50도 시작값
TURN_50_TIME = 0.69


# =========================================================
# 회전 후 새 길 탐색
# =========================================================
SEARCH_TURN_SPEED = 18
SEARCH_MAX_TIME = 3.0

RECOVERY_CENTER_RANGE = 100
NEW_ROAD_CONFIRM_FRAMES = 5
NEW_ROAD_MIN_POINTS = 5


# =========================================================
# 회전 후 황토색 박스 중심점 복원
# =========================================================
BROWN_RATIO_THRESHOLD = 0.06

ROAD_WIDTH_ALPHA = 0.18

# 보이는 도로 바깥 경계 + 기억한 반쪽 폭을 강하게 사용
EDGE_WEIGHT = 0.90

# 회전 후 현재 곡선의 이전 중심은 약하게 사용
HISTORY_WEIGHT = 0.10

# 곡선 중심선 프레임 간 smoothing
CURVE_SMOOTH_ALPHA = 0.70

# 황토색을 실제로 한 번 본 뒤,
# 이 프레임 수만큼 연속으로 사라지면 정상 FOLLOW 복귀
BROWN_CLEAR_FRAMES = 10


# =========================================================
# 상태
#
# FOLLOW
# AFTER_ARROW
# STOP_BEFORE_TURN
# TURN_RIGHT
# SEARCH_NEW_ROAD
# POST_TURN_CURVE_FOLLOW
# =========================================================
drive_state = "FOLLOW"

auto_mode = False

last_error = 0.0

arrow_count = 0
arrow_locked = False
arrow_seen_counter = 0
arrow_gone_counter = 0

# 현재 회전 동작을 발생시킨 화살표 번호
# 2 또는 3
active_turn_arrow = None

# 이미 회전 동작을 끝낸 가장 최근 화살표 번호
last_completed_turn_arrow = 0

state_start_time = 0.0

new_road_counter = 0

brown_seen_after_turn = False
brown_clear_counter = 0


# =========================================================
# 회전 후 새 곡선 전용 기억값
#
# 회전 전 직선 중심선/도로폭은 사용하지 않음
# =========================================================
post_turn_widths = [None] * MAX_POINTS
post_turn_prev_centers = [None] * MAX_POINTS


# =========================================================
# 모터 제어
# =========================================================
def drive(left, right):

    left = int(
        np.clip(
            left,
            -100,
            100
        )
    )

    right = int(
        np.clip(
            right,
            -100,
            100
        )
    )

    try:

        requests.post(
            HOST + "/drive",
            json={
                "l": left,
                "r": right
            },
            timeout=0.3
        )

    except requests.RequestException:
        pass


def stop():

    try:

        requests.post(
            HOST + "/stop",
            timeout=0.3
        )

    except requests.RequestException:
        pass


# =========================================================
# 작은 흰색 노이즈 제거
# =========================================================
def remove_small_components(mask, min_area):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# =========================================================
# 검은 화살표 부분을 흰색 도로로 메우기
# =========================================================
def fill_arrow_gaps(white_mask):

    filled = white_mask.copy()

    h, w = white_mask.shape

    max_gap = int(
        w * ARROW_FILL_GAP_RATIO
    )

    for y in range(h):

        xs = np.where(
            white_mask[y] == 255
        )[0]

        if len(xs) == 0:
            continue

        groups = np.split(
            xs,
            np.where(
                np.diff(xs) > 1
            )[0] + 1
        )

        groups = [
            g
            for g in groups
            if len(g) >= MIN_WIDTH
        ]

        if len(groups) < 2:
            continue

        for i in range(
            len(groups) - 1
        ):

            left_end = int(
                groups[i][-1]
            )

            right_start = int(
                groups[i + 1][0]
            )

            gap = (
                right_start
                - left_end
                - 1
            )

            if (
                gap > 0
                and
                gap <= max_gap
            ):

                filled[
                    y,
                    left_end:right_start + 1
                ] = 255

    kernel = np.ones(
        (3, 5),
        np.uint8
    )

    filled = cv2.morphologyEx(
        filled,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return filled


# =========================================================
# 검은 화살표 검출
# =========================================================
def detect_arrow(hsv, road_mask):

    black_mask = cv2.inRange(
        hsv,
        LOWER_BLACK,
        UPPER_BLACK
    )

    arrow_mask = cv2.bitwise_and(
        black_mask,
        road_mask
    )

    arrow_mask = cv2.morphologyEx(
        arrow_mask,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    contours, _ = cv2.findContours(
        arrow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    arrow_found = False
    best_area = 0

    _, roi_w = arrow_mask.shape

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < ARROW_MIN_AREA:
            continue

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        center_x = x + cw // 2

        # 화면 좌우 끝 검은 영역 제외
        if (
            center_x < int(roi_w * 0.15)
            or
            center_x > int(roi_w * 0.85)
        ):
            continue

        if area > best_area:

            best_area = area
            arrow_found = True

    return (
        arrow_found,
        arrow_mask,
        best_area
    )


# =========================================================
# 한 scan line의 흰색 그룹
# =========================================================
def get_groups(row):

    xs = np.where(
        row == 255
    )[0]

    if len(xs) == 0:
        return []

    groups = np.split(
        xs,
        np.where(
            np.diff(xs) > 1
        )[0] + 1
    )

    return [
        g
        for g in groups
        if len(g) >= MIN_WIDTH
    ]


# =========================================================
# 기본 중심선 추출
# =========================================================
def get_basic_center_points(
    road_mask,
    roi_start
):

    roi_h, roi_w = road_mask.shape

    points = []

    prev_center = None

    for scan_i, y in enumerate(
        range(
            roi_h - 1,
            0,
            -STEP
        )
    ):

        if scan_i >= MAX_POINTS:
            break

        groups = get_groups(
            road_mask[y]
        )

        if len(groups) == 0:
            continue

        # 첫 점은 화면 중심에 가장 가까운 흰 영역
        if prev_center is None:

            chosen = min(
                groups,
                key=lambda g:
                abs(
                    (
                        int(g[0])
                        + int(g[-1])
                    ) / 2.0
                    - roi_w / 2.0
                )
            )

        # 이후 점은 직전 중심에 가까운 흰 영역
        else:

            chosen = min(
                groups,
                key=lambda g:
                abs(
                    (
                        int(g[0])
                        + int(g[-1])
                    ) / 2.0
                    - prev_center
                )
            )

        x_left = int(
            chosen[0]
        )

        x_right = int(
            chosen[-1]
        )

        x_center = (
            x_left + x_right
        ) / 2.0

        if prev_center is not None:

            if abs(
                x_center
                - prev_center
            ) > MAX_JUMP:

                continue

        points.append(
            {
                "scan_i": scan_i,
                "x": x_center,
                "left": x_left,
                "right": x_right,
                "y_roi": y,
                "y_frame": y + roi_start
            }
        )

        prev_center = x_center

    return points


# =========================================================
# 회전 후 곡선 + 황토색 중심점 복원
#
# 핵심:
# 황토색 때문에 한쪽 흰색만 남아도
# 그 흰색 덩어리의 중심을 그대로 따라가지 않는다.
#
# 왼쪽 흰 영역이면:
#   왼쪽 도로 외곽 경계 + 기억한 도로폭/2
#
# 오른쪽 흰 영역이면:
#   오른쪽 도로 외곽 경계 - 기억한 도로폭/2
#
# 이렇게 실제 도로 중심점을 복원한다.
# =========================================================
def get_post_turn_corrected_points(
    road_mask,
    roi_start,
    brown_zone,
    width_memory,
    prev_centers
):

    roi_h, roi_w = road_mask.shape

    corrected_points = []

    new_prev_centers = (
        [None] * MAX_POINTS
    )

    frame_prev_center = None

    for scan_i, y in enumerate(
        range(
            roi_h - 1,
            0,
            -STEP
        )
    ):

        if scan_i >= MAX_POINTS:
            break

        groups = get_groups(
            road_mask[y]
        )

        if len(groups) == 0:
            continue

        # 회전 후 새 곡선의 이전 프레임 중심만 사용
        if (
            prev_centers[scan_i]
            is not None
        ):

            predicted_center = float(
                prev_centers[scan_i]
            )

        elif frame_prev_center is not None:

            predicted_center = float(
                frame_prev_center
            )

        else:

            predicted_center = (
                roi_w / 2.0
            )

        chosen = min(
            groups,
            key=lambda g:
            abs(
                (
                    int(g[0])
                    + int(g[-1])
                ) / 2.0
                - predicted_center
            )
        )

        left = int(
            chosen[0]
        )

        right = int(
            chosen[-1]
        )

        raw_center = (
            left + right
        ) / 2.0

        current_width = (
            right - left
        )

        # -------------------------------------------------
        # 정상 흰 도로일 때:
        # 회전 후 새 곡선의 도로폭을 높이별로 기억
        # -------------------------------------------------
        if not brown_zone:

            if current_width >= MIN_WIDTH:

                if (
                    width_memory[scan_i]
                    is None
                ):

                    width_memory[scan_i] = (
                        float(current_width)
                    )

                else:

                    width_memory[scan_i] = (
                        (
                            1.0
                            - ROAD_WIDTH_ALPHA
                        )
                        * width_memory[scan_i]

                        + ROAD_WIDTH_ALPHA
                        * float(current_width)
                    )

            corrected_x = float(
                raw_center
            )

        # -------------------------------------------------
        # 황토색 박스 구간:
        # 중심점 복원
        # -------------------------------------------------
        else:

            remembered_width = (
                width_memory[scan_i]
            )

            if remembered_width is not None:

                visible_center = (
                    left + right
                ) / 2.0

                # 왼쪽 흰 부분만 보이는 경우
                if (
                    visible_center
                    < predicted_center
                ):

                    edge_based_center = (
                        left
                        + remembered_width / 2.0
                    )

                # 오른쪽 흰 부분만 보이는 경우
                else:

                    edge_based_center = (
                        right
                        - remembered_width / 2.0
                    )

                corrected_x = (
                    EDGE_WEIGHT
                    * edge_based_center

                    + HISTORY_WEIGHT
                    * predicted_center
                )

            elif (
                prev_centers[scan_i]
                is not None
            ):

                corrected_x = float(
                    prev_centers[scan_i]
                )

            else:

                corrected_x = float(
                    raw_center
                )

        # 황토색 구간에서는 복원된 중심선 자체를
        # 화면 오른쪽으로 조금 이동
        if brown_zone:

            corrected_x += (
                BROWN_CENTER_SHIFT_PX
            )

        corrected_x = float(
            np.clip(
                corrected_x,
                0,
                roi_w - 1
            )
        )

        # 곡선 모양을 유지하면서 약간 부드럽게
        if (
            prev_centers[scan_i]
            is not None
        ):

            previous_x = float(
                prev_centers[scan_i]
            )

            corrected_x = (
                (
                    1.0
                    - CURVE_SMOOTH_ALPHA
                )
                * previous_x

                + CURVE_SMOOTH_ALPHA
                * corrected_x
            )

        new_prev_centers[
            scan_i
        ] = corrected_x

        corrected_points.append(
            {
                "scan_i": scan_i,
                "x": corrected_x,
                "raw_x": raw_center,
                "left": left,
                "right": right,
                "y_roi": y,
                "y_frame": y + roi_start
            }
        )

        frame_prev_center = (
            corrected_x
        )

    # -----------------------------------------------------
    # 복원 중심점들을 2차 곡선으로 연결
    # -----------------------------------------------------
    fitted_points = (
        corrected_points.copy()
    )

    if len(corrected_points) >= 4:

        ys = np.array(
            [
                p["y_roi"]
                for p in corrected_points
            ],
            dtype=np.float32
        )

        xs = np.array(
            [
                p["x"]
                for p in corrected_points
            ],
            dtype=np.float32
        )

        try:

            coeff = np.polyfit(
                ys,
                xs,
                2
            )

            fitted_points = []

            for p in corrected_points:

                fitted_x = float(
                    np.polyval(
                        coeff,
                        p["y_roi"]
                    )
                )

                fitted_x = float(
                    np.clip(
                        fitted_x,
                        0,
                        roi_w - 1
                    )
                )

                q = p.copy()

                q["x"] = fitted_x

                fitted_points.append(
                    q
                )

        except Exception:

            fitted_points = (
                corrected_points.copy()
            )

    return (
        fitted_points,
        width_memory,
        new_prev_centers
    )


# =========================================================
# P 제어
# =========================================================
def run_p_control(
    error,
    kp=KP
):

    global last_error

    if error is not None:

        last_error = error

        correction = (
            kp * error
        )

        left_speed = (
            BASE_SPEED
            + correction
        )

        right_speed = (
            BASE_SPEED
            - correction
        )

        left_speed = int(
            np.clip(
                left_speed,
                0,
                MAX_SPEED
            )
        )

        right_speed = int(
            np.clip(
                right_speed,
                0,
                MAX_SPEED
            )
        )

        drive(
            left_speed,
            right_speed
        )

    else:

        if last_error < 0:

            drive(
                0,
                SEARCH_SPEED
            )

        elif last_error > 0:

            drive(
                SEARCH_SPEED,
                0
            )

        else:

            stop()


# =========================================================
# 카메라
# =========================================================
cap = cv2.VideoCapture(
    HOST + "/video"
)

if not cap.isOpened():

    print("카메라 연결 실패")
    raise SystemExit


print("""
=================================================
Pinky Pro
=================================================

P       : AUTO ON / OFF
W/S/A/D : 수동
SPACE   : 정지
R       : 초기화
ESC     : 종료

[동작]

1번 화살표
-> count 1
-> 계속 주행

2번 화살표
-> count 2
-> 화면에서 사라짐
-> 2.8초 중심선 추종
-> 정지
-> 오른쪽 약 50도 회전
-> 새 길 탐색
-> 황토색 중심점 복원 + 강한 우회전

3번 화살표
-> count 3
-> 화면에서 사라짐
-> 똑같이 2.8초 중심선 추종
-> 정지
-> 오른쪽 약 50도 회전
-> 새 길 탐색
-> 다시 주행

황토색 구간:
-> 한쪽 흰색 중심 그대로 사용 X
-> 도로 경계 + 기억한 도로폭으로 중심 복원
-> 복원 중심선을 오른쪽으로 28px 추가 보정
-> 황토색에서만 KP_BROWN = 0.22로 더 강하게 우회전
=================================================
""")


# =========================================================
# 메인 루프
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:

        print("영상 수신 실패")
        break

    h, w = frame.shape[:2]

    # =====================================================
    # ROI
    # =====================================================
    roi_start = int(
        h * ROI_START_RATIO
    )

    roi = frame[
        roi_start:h,
        :
    ]

    roi_h, roi_w = roi.shape[:2]

    # =====================================================
    # HSV
    # =====================================================
    roi_blur = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv2.cvtColor(
        roi_blur,
        cv2.COLOR_BGR2HSV
    )

    # =====================================================
    # 흰색 도로
    # =====================================================
    white_mask = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    white_mask = remove_small_components(
        white_mask,
        MIN_AREA
    )

    # 검은 화살표 부분 메우기
    road_mask = fill_arrow_gaps(
        white_mask
    )

    # =====================================================
    # 황토색 검출
    # 화면 중앙 60%만 봄
    # =====================================================
    brown_mask = cv2.inRange(
        hsv,
        LOWER_BROWN,
        UPPER_BROWN
    )

    focus_x1 = int(
        roi_w * 0.20
    )

    focus_x2 = int(
        roi_w * 0.80
    )

    brown_focus = brown_mask[
        :,
        focus_x1:focus_x2
    ]

    if brown_focus.size > 0:

        brown_ratio = (
            cv2.countNonZero(
                brown_focus
            )
            / brown_focus.size
        )

    else:

        brown_ratio = 0.0

    brown_zone = (
        brown_ratio
        >= BROWN_RATIO_THRESHOLD
    )

    # =====================================================
    # 화살표 검출
    # =====================================================
    (
        arrow_found,
        arrow_mask,
        arrow_area
    ) = detect_arrow(
        hsv,
        road_mask
    )

    # =====================================================
    # 화살표 카운팅
    #
    # 정상 FOLLOW 상태일 때
    # 1, 2, 3번 화살표 모두 계속 카운트
    # =====================================================
    if drive_state in ("FOLLOW", "POST_TURN_CURVE_FOLLOW"):

        if arrow_found:

            arrow_gone_counter = 0

            if not arrow_locked:

                arrow_seen_counter += 1

                if (
                    arrow_seen_counter
                    >= ARROW_CONFIRM_FRAMES
                ):

                    arrow_count += 1

                    arrow_locked = True

                    arrow_seen_counter = 0

                    print(
                        f"ARROW COUNT = {arrow_count}"
                    )

        else:

            arrow_seen_counter = 0

            if arrow_locked:

                arrow_gone_counter += 1

                if (
                    arrow_gone_counter
                    >= ARROW_GONE_FRAMES
                ):

                    arrow_locked = False

                    arrow_gone_counter = 0

                    print(
                        f"ARROW {arrow_count} DISAPPEARED"
                    )

                    # =====================================
                    # 2번 또는 3번 화살표가 사라지면
                    # 동일한 회전 시퀀스 시작
                    # =====================================
                    if (
                        arrow_count in (2, 3)
                        and
                        arrow_count
                        > last_completed_turn_arrow
                    ):

                        active_turn_arrow = (
                            arrow_count
                        )

                        drive_state = (
                            "AFTER_ARROW"
                        )

                        state_start_time = (
                            time.time()
                        )

                        print(
                            f"ARROW {arrow_count} GONE"
                            " -> FOLLOW FOR 2.8 SEC"
                        )

    # =====================================================
    # 기본 중심선
    # =====================================================
    basic_points = (
        get_basic_center_points(
            road_mask,
            roi_start
        )
    )

    basic_target_x = None
    basic_target_y = None
    basic_error = None

    if (
        len(basic_points)
        > TARGET_INDEX
    ):

        basic_target_x = (
            basic_points[
                TARGET_INDEX
            ]["x"]
        )

        basic_target_y = (
            basic_points[
                TARGET_INDEX
            ]["y_frame"]
        )

        basic_error = (
            basic_target_x
            - roi_w / 2.0
        )

    # =====================================================
    # 회전 후 보정 중심선
    # =====================================================
    corrected_points = []

    corrected_target_x = None
    corrected_target_y = None
    corrected_error = None

    if (
        drive_state
        == "POST_TURN_CURVE_FOLLOW"
    ):

        (
            corrected_points,
            post_turn_widths,
            post_turn_prev_centers
        ) = get_post_turn_corrected_points(
            road_mask,
            roi_start,
            brown_zone,
            post_turn_widths,
            post_turn_prev_centers
        )

        if (
            len(corrected_points)
            > TARGET_INDEX
        ):

            corrected_target_x = (
                corrected_points[
                    TARGET_INDEX
                ]["x"]
            )

            corrected_target_y = (
                corrected_points[
                    TARGET_INDEX
                ]["y_frame"]
            )

            corrected_error = (
                corrected_target_x
                - roi_w / 2.0
            )


    # =====================================================
    # 자동 주행
    # =====================================================
    if auto_mode:

        # -------------------------------------------------
        # 정상 FOLLOW
        # -------------------------------------------------
        if drive_state == "FOLLOW":

            run_p_control(
                basic_error,
                KP
            )

        # -------------------------------------------------
        # 2번/3번 화살표가 사라진 뒤 2.8초
        # -------------------------------------------------
        elif (
            drive_state
            == "AFTER_ARROW"
        ):

            elapsed = (
                time.time()
                - state_start_time
            )

            if (
                elapsed
                < AFTER_ARROW_FOLLOW_TIME
            ):

                run_p_control(
                    basic_error,
                    KP
                )

            else:

                stop()

                drive_state = (
                    "STOP_BEFORE_TURN"
                )

                state_start_time = (
                    time.time()
                )

                print(
                    f"ARROW {active_turn_arrow}: "
                    "2.8 SEC COMPLETE -> STOP"
                )

        # -------------------------------------------------
        # 회전 전 잠깐 정지
        # -------------------------------------------------
        elif (
            drive_state
            == "STOP_BEFORE_TURN"
        ):

            stop()

            elapsed = (
                time.time()
                - state_start_time
            )

            if (
                elapsed
                >= STOP_BEFORE_TURN_TIME
            ):

                drive_state = (
                    "TURN_RIGHT"
                )

                state_start_time = (
                    time.time()
                )

                print(
                    f"ARROW {active_turn_arrow}: "
                    "TURN RIGHT 50 DEG"
                )

        # -------------------------------------------------
        # 오른쪽 약 50도 회전
        # -------------------------------------------------
        elif (
            drive_state
            == "TURN_RIGHT"
        ):

            elapsed = (
                time.time()
                - state_start_time
            )

            if (
                elapsed
                < TURN_50_TIME
            ):

                drive(
                    TURN_SPEED,
                    -TURN_SPEED
                )

            else:

                stop()

                # =========================================
                # 회전 전 중심선 정보는 폐기
                # 새 길 기준으로 다시 시작
                # =========================================
                post_turn_widths = (
                    [None] * MAX_POINTS
                )

                post_turn_prev_centers = (
                    [None] * MAX_POINTS
                )

                last_error = 0.0

                new_road_counter = 0

                brown_seen_after_turn = False
                brown_clear_counter = 0

                drive_state = (
                    "SEARCH_NEW_ROAD"
                )

                state_start_time = (
                    time.time()
                )

                # 이번 회전 동작을 완료한 화살표 번호 저장
                # 2번 회전 후에도 3번 화살표를 새로 인식할 수 있게 한다.
                if active_turn_arrow is not None:

                    last_completed_turn_arrow = (
                        active_turn_arrow
                    )

                print(
                    "50 DEG TURN COMPLETE"
                    " -> SEARCH NEW ROAD"
                )

        # -------------------------------------------------
        # 회전 후 새 길 찾기
        # -------------------------------------------------
        elif (
            drive_state
            == "SEARCH_NEW_ROAD"
        ):

            elapsed = (
                time.time()
                - state_start_time
            )

            road_ok = False

            if (
                basic_target_x is not None
                and
                len(basic_points)
                >= NEW_ROAD_MIN_POINTS
            ):

                search_error = (
                    basic_target_x
                    - roi_w / 2.0
                )

                if (
                    abs(search_error)
                    < RECOVERY_CENTER_RANGE
                ):

                    road_ok = True

            if road_ok:

                new_road_counter += 1

            else:

                new_road_counter = 0

            # 새 길을 여러 프레임 안정적으로 확인
            if (
                new_road_counter
                >= NEW_ROAD_CONFIRM_FRAMES
            ):

                stop()

                # 새 길의 초기 도로폭/중심 기억
                for p in basic_points:

                    i = p["scan_i"]

                    width = (
                        p["right"]
                        - p["left"]
                    )

                    if (
                        i < MAX_POINTS
                        and
                        width >= MIN_WIDTH
                    ):

                        post_turn_widths[i] = (
                            float(width)
                        )

                        post_turn_prev_centers[i] = (
                            float(p["x"])
                        )

                drive_state = (
                    "POST_TURN_CURVE_FOLLOW"
                )

                new_road_counter = 0

                last_error = 0.0

                print(
                    "NEW ROAD FOUND"
                    " -> CORRECTED CURVE FOLLOW"
                )

            elif (
                elapsed
                < SEARCH_MAX_TIME
            ):

                drive(
                    SEARCH_TURN_SPEED,
                    -SEARCH_TURN_SPEED
                )

            else:

                stop()

                auto_mode = False

                print(
                    "SEARCH TIMEOUT"
                    " -> AUTO OFF"
                )

        # -------------------------------------------------
        # 회전 후 곡선 + 황토색 박스
        # -------------------------------------------------
        elif (
            drive_state
            == "POST_TURN_CURVE_FOLLOW"
        ):

            # 황토색 박스에서만 더 강한 P제어
            if brown_zone:

                run_p_control(
                    corrected_error,
                    KP_BROWN
                )

            else:

                run_p_control(
                    corrected_error,
                    KP
                )

            # =============================================
            # 황토색 박스가 여러 개 연속으로 나와도
            # 이 상태를 계속 유지한다.
            #
            # 황토색이 나오면:
            #   중심점 복원 + 오른쪽 중심선 보정
            #   + 황토색 전용 강한 P제어
            #
            # 황토색이 잠깐 사라져도 FOLLOW로 복귀하지 않는다.
            # 따라서 다음 황토색 박스에서도 같은 보정이 다시 적용된다.
            # =============================================
            if brown_zone:

                brown_seen_after_turn = True
                brown_clear_counter = 0

            else:

                brown_clear_counter += 1

    # =====================================================
    # 화면 표시
    # =====================================================
    result = frame.copy()

    # ROI 시작선
    cv2.line(
        result,
        (0, roi_start),
        (w, roi_start),
        (0, 255, 255),
        2
    )

    # 화면 중심
    cv2.line(
        result,
        (
            roi_w // 2,
            roi_start
        ),
        (
            roi_w // 2,
            h
        ),
        (0, 255, 0),
        2
    )

    # -----------------------------------------------------
    # 일반 중심선 표시
    # -----------------------------------------------------
    if (
        drive_state
        != "POST_TURN_CURVE_FOLLOW"
    ):

        for p in basic_points:

            cv2.circle(
                result,
                (
                    int(p["x"]),
                    int(p["y_frame"])
                ),
                4,
                (0, 0, 255),
                -1
            )

        for i in range(
            len(basic_points) - 1
        ):

            p1 = (
                int(
                    basic_points[i]["x"]
                ),
                int(
                    basic_points[i]["y_frame"]
                )
            )

            p2 = (
                int(
                    basic_points[i + 1]["x"]
                ),
                int(
                    basic_points[i + 1]["y_frame"]
                )
            )

            cv2.line(
                result,
                p1,
                p2,
                (255, 0, 0),
                3
            )

        if (
            basic_target_x is not None
            and
            basic_target_y is not None
        ):

            cv2.circle(
                result,
                (
                    int(basic_target_x),
                    int(basic_target_y)
                ),
                9,
                (0, 255, 255),
                -1
            )

    # -----------------------------------------------------
    # 회전 후 황토색 보정 중심선 표시
    # -----------------------------------------------------
    else:

        # 빨간 점 = 보정 전 흰 영역 중심
        for p in corrected_points:

            cv2.circle(
                result,
                (
                    int(p["raw_x"]),
                    int(p["y_frame"])
                ),
                3,
                (0, 0, 255),
                -1
            )

        # 파란 선 = 복원된 곡선 중심선
        for i in range(
            len(corrected_points) - 1
        ):

            p1 = (
                int(
                    corrected_points[i]["x"]
                ),
                int(
                    corrected_points[i]["y_frame"]
                )
            )

            p2 = (
                int(
                    corrected_points[i + 1]["x"]
                ),
                int(
                    corrected_points[i + 1]["y_frame"]
                )
            )

            cv2.line(
                result,
                p1,
                p2,
                (255, 0, 0),
                3
            )

        if (
            corrected_target_x is not None
            and
            corrected_target_y is not None
        ):

            cv2.circle(
                result,
                (
                    int(corrected_target_x),
                    int(corrected_target_y)
                ),
                9,
                (0, 255, 255),
                -1
            )

    # =====================================================
    # 상태 텍스트
    # =====================================================
    mode_text = (
        "AUTO"
        if auto_mode
        else "MANUAL"
    )

    cv2.putText(
        result,
        f"MODE: {mode_text}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"STATE: {drive_state}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"ARROW COUNT: {arrow_count}",
        (20, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"BROWN: {brown_ratio:.2f}",
        (20, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"KP: {KP:.2f} / BROWN: {KP_BROWN:.2f}",
        (20, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 0, 0),
        2
    )

    if (
        drive_state
        == "AFTER_ARROW"
    ):

        remaining = max(
            0.0,
            AFTER_ARROW_FOLLOW_TIME
            - (
                time.time()
                - state_start_time
            )
        )

        cv2.putText(
            result,
            f"ARROW {active_turn_arrow} TURN IN: {remaining:.1f}s",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 255),
            2
        )

    if (
        drive_state
        == "POST_TURN_CURVE_FOLLOW"
    ):

        if brown_zone:

            status_text = (
                "BROWN: RIGHT SHIFT + STRONG TURN"
            )

        else:

            status_text = (
                "POST TURN CURVE FOLLOW"
            )

        cv2.putText(
            result,
            status_text,
            (20, 198),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 255),
            2
        )

    # =====================================================
    # 창 출력
    # =====================================================
    cv2.imshow(
        "Pinky Main",
        result
    )

    cv2.imshow(
        "White Mask",
        white_mask
    )

    cv2.imshow(
        "Arrow Filled Road Mask",
        road_mask
    )

    cv2.imshow(
        "Brown Mask",
        brown_mask
    )

    cv2.imshow(
        "Arrow Mask",
        arrow_mask
    )

    # =====================================================
    # 키보드
    # =====================================================
    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    # AUTO ON/OFF
    if key == ord("p"):

        auto_mode = (
            not auto_mode
        )

        stop()

        print(
            "AUTO ON"
            if auto_mode
            else "AUTO OFF"
        )

    # 수동 전진
    elif key == ord("w"):

        auto_mode = False

        drive(
            30,
            30
        )

    # 수동 후진
    elif key == ord("s"):

        auto_mode = False

        drive(
            -30,
            -30
        )

    # 수동 좌회전
    elif key == ord("a"):

        auto_mode = False

        drive(
            -25,
            25
        )

    # 수동 우회전
    elif key == ord("d"):

        auto_mode = False

        drive(
            25,
            -25
        )

    # 정지
    elif key == 32:

        auto_mode = False

        stop()

        print("STOP")

    # 전체 초기화
    elif key == ord("r"):

        auto_mode = False

        drive_state = "FOLLOW"

        last_error = 0.0

        arrow_count = 0
        arrow_locked = False
        arrow_seen_counter = 0
        arrow_gone_counter = 0

        active_turn_arrow = None
        last_completed_turn_arrow = 0

        state_start_time = 0.0

        new_road_counter = 0

        brown_seen_after_turn = False
        brown_clear_counter = 0

        post_turn_widths = (
            [None] * MAX_POINTS
        )

        post_turn_prev_centers = (
            [None] * MAX_POINTS
        )

        stop()

        print("RESET")

    # 종료
    elif key == 27:

        stop()
        break


# =========================================================
# 종료
# =========================================================
stop()

cap.release()

cv2.destroyAllWindows()

print("종료")
