import cv2
import numpy as np
import requests
import time


# =========================================================
# Pinky Pro 주소
# =========================================================
HOST = "http://192.168.4.1:8000"


# =========================================================
# 주행 설정
# =========================================================
BASE_SPEED = 25
KP = 0.12
MAX_SPEED = 40
SEARCH_SPEED = 18


# =========================================================
# ROI
# 0.68 = 화면 아래쪽 32%만 사용
# =========================================================
ROI_START_RATIO = 0.68


# =========================================================
# 흰색 도로 HSV
# =========================================================
LOWER_WHITE = np.array([0, 0, 175])
UPPER_WHITE = np.array([180, 75, 255])


# =========================================================
# 중심선 검출 설정
# =========================================================
MIN_AREA = 500
STEP = 15
MIN_WIDTH = 25
MAX_JUMP = 60
MAX_POINTS = 8


# =========================================================
# 도로 내부 검은 영역 연결 설정
#
# 0.30 = ROI 폭의 30% 이하 간격이면
# 같은 도로 내부의 화살표/박스로 판단
# =========================================================
INTERNAL_GAP_RATIO = 0.30


# =========================================================
# 교차로 감지
# =========================================================
INTERSECTION_WIDTH_ENTER = 0.92
INTERSECTION_WIDTH_EXIT = 0.80

INTERSECTION_CONFIRM_FRAMES = 3
INTERSECTION_COOLDOWN = 3.0


# =========================================================
# 우회전
# =========================================================
TURN_SPEED = 30
TURN_MIN_TIME = 0.55
TURN_MAX_TIME = 1.50


# =========================================================
# 우회전 후 중심선 재획득
# =========================================================
RECOVERY_CENTER_RANGE = 100
RECOVERY_FRAMES = 3


# =========================================================
# 상태
# =========================================================
drive_state = "FOLLOW"
auto_mode = False

last_error = 0

intersection_locked = False
intersection_counter = 0
last_intersection_time = -999.0

turn_start_time = 0.0
recovery_counter = 0


# =========================================================
# 모터 제어
# =========================================================
def drive(left, right):

    left = int(np.clip(left, -100, 100))
    right = int(np.clip(right, -100, 100))

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
def remove_small_components(binary_mask, min_area):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(binary_mask)

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            cleaned[
                labels == i
            ] = 255

    return cleaned


# =========================================================
# 흰색 도로 내부의 검은 화살표 / 황토색 박스 무시
#
# 예:
#
# 흰색도로 | 검은 화살표 | 흰색도로
#
#            ↓
#
# 흰색도로 |   흰색      | 흰색도로
#
# 단,
# 흰색 영역 사이 간격이 너무 크면
# 서로 다른 도로라고 보고 연결하지 않음
# =========================================================
def fill_road_internal_gaps(white_mask):

    filled = white_mask.copy()

    h, w = white_mask.shape

    max_internal_gap = int(
        w * INTERNAL_GAP_RATIO
    )

    for y in range(h):

        xs = np.where(
            white_mask[y] == 255
        )[0]

        if len(xs) == 0:
            continue


        # 연속된 흰색 영역 분리
        groups = np.split(
            xs,
            np.where(
                np.diff(xs) > 1
            )[0] + 1
        )


        # 너무 작은 흰색 조각 제거
        groups = [
            g
            for g in groups
            if len(g) >= MIN_WIDTH
        ]


        if len(groups) < 2:
            continue


        # 인접한 흰색 영역끼리 검사
        for i in range(
            len(groups) - 1
        ):

            left_group = groups[i]
            right_group = groups[i + 1]

            left_end = int(
                left_group[-1]
            )

            right_start = int(
                right_group[0]
            )


            gap = (
                right_start
                - left_end
                - 1
            )


            # 작은 내부 간격이면
            # 화살표 / STOP 박스로 보고 연결
            if (
                gap > 0
                and
                gap <= max_internal_gap
            ):

                filled[
                    y,
                    left_end:right_start + 1
                ] = 255


    return filled


# =========================================================
# 카메라 연결
# =========================================================
cap = cv2.VideoCapture(
    HOST + "/video"
)

if not cap.isOpened():

    print("카메라 연결 실패")
    raise SystemExit


print("""
================================
Pinky Pro Centerline Follower
================================

P       : AUTO ON / OFF
W       : 수동 전진
S       : 수동 후진
A       : 수동 좌회전
D       : 수동 우회전
SPACE   : 정지
R       : 초기화
ESC     : 종료
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
    # 1. ROI
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
    # 2. Blur + HSV
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
    # 3. 흰색 도로 검출
    # =====================================================
    white_mask = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )


    # =====================================================
    # 4. 작은 노이즈 제거
    # =====================================================
    kernel_open = np.ones(
        (3, 3),
        np.uint8
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1
    )


    # =====================================================
    # 5. 아주 작은 틈만 메움
    # =====================================================
    kernel_close = np.ones(
        (7, 7),
        np.uint8
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=1
    )


    # =====================================================
    # 6. 작은 흰색 잡영 제거
    # =====================================================
    white_mask = remove_small_components(
        white_mask,
        MIN_AREA
    )


    # =====================================================
    # 7. ★ 도로 내부 검은 화살표/황토색 박스 메우기
    # =====================================================
    road_mask = fill_road_internal_gaps(
        white_mask
    )


    # =====================================================
    # 8. 도로폭 계산
    #
    # 메워진 road_mask 사용
    # =====================================================
    width_ratios = []


    for y in range(
        roi_h - 1,
        0,
        -10
    ):

        xs = np.where(
            road_mask[y] == 255
        )[0]


        if len(xs) < MIN_WIDTH:
            continue


        left_edge = int(
            xs[0]
        )

        right_edge = int(
            xs[-1]
        )


        road_width = (
            right_edge
            - left_edge
        )


        ratio = (
            road_width
            / roi_w
        )


        width_ratios.append(
            ratio
        )


    if len(width_ratios) > 0:

        width_ratio = float(
            np.percentile(
                width_ratios,
                80
            )
        )

    else:

        width_ratio = 0.0


    # =====================================================
    # 9. 교차로 후보
    # =====================================================
    if intersection_locked:

        intersection_candidate = (
            width_ratio
            > INTERSECTION_WIDTH_EXIT
        )

    else:

        intersection_candidate = (
            width_ratio
            > INTERSECTION_WIDTH_ENTER
        )


    # =====================================================
    # 10. 교차로 연속 확인
    # =====================================================
    if (
        drive_state == "FOLLOW"
        and
        intersection_candidate
    ):

        intersection_counter += 1

    else:

        intersection_counter = 0


    intersection_detected = (
        intersection_counter
        >= INTERSECTION_CONFIRM_FRAMES
    )


    # =====================================================
    # 11. 중심선 계산
    #
    # ★ white_mask가 아니라
    # road_mask 사용
    # =====================================================
    center_points = []

    prev_center = None


    for y in range(
        roi_h - 1,
        0,
        -STEP
    ):

        xs = np.where(
            road_mask[y] == 255
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


        if len(groups) == 0:
            continue


        # =================================================
        # 첫 번째 중심점
        # =================================================
        if prev_center is None:

            chosen = min(
                groups,
                key=lambda g:
                abs(
                    (
                        int(g[0])
                        + int(g[-1])
                    ) // 2
                    - roi_w // 2
                )
            )


        # =================================================
        # 이후 중심점
        # =================================================
        else:

            predicted_x = (
                prev_center
            )


            if len(center_points) >= 2:

                x1 = (
                    center_points[-2][0]
                )

                x2 = (
                    center_points[-1][0]
                )

                dx = (
                    x2 - x1
                )

                dx = int(
                    np.clip(
                        dx,
                        -35,
                        35
                    )
                )

                predicted_x = (
                    x2 + dx
                )


            chosen = min(
                groups,
                key=lambda g:
                abs(
                    (
                        int(g[0])
                        + int(g[-1])
                    ) // 2
                    - predicted_x
                )
            )


        x_left = int(
            chosen[0]
        )

        x_right = int(
            chosen[-1]
        )

        x_center = (
            x_left
            + x_right
        ) // 2


        # =================================================
        # 갑자기 다른 영역으로 튀는 것 방지
        # =================================================
        if prev_center is not None:

            if abs(
                x_center
                - prev_center
            ) > MAX_JUMP:

                continue


        original_y = (
            y
            + roi_start
        )


        center_points.append(
            (
                x_center,
                original_y
            )
        )


        prev_center = (
            x_center
        )


        if len(center_points) >= MAX_POINTS:
            break


    # =====================================================
    # 12. 목표 중심점
    # =====================================================
    target_x = None
    target_y = None


    if len(center_points) >= 3:

        target_x, target_y = (
            center_points[2]
        )


    # =====================================================
    # 13. Error
    # =====================================================
    error = None


    if target_x is not None:

        error = (
            target_x
            - w // 2
        )

        last_error = (
            error
        )


    # =====================================================
    # 14. FOLLOW -> TURN_RIGHT
    # =====================================================
    now = time.time()


    if (
        drive_state == "FOLLOW"
        and
        intersection_detected
        and
        not intersection_locked
        and
        now - last_intersection_time
        >= INTERSECTION_COOLDOWN
    ):

        intersection_locked = True

        intersection_counter = 0

        drive_state = (
            "TURN_RIGHT"
        )

        turn_start_time = (
            time.time()
        )

        recovery_counter = 0

        stop()

        print(
            "FOLLOW -> TURN_RIGHT"
        )


    # =====================================================
    # 15. 자동주행
    # =====================================================
    if auto_mode:


        # =================================================
        # FOLLOW
        # =================================================
        if drive_state == "FOLLOW":

            if error is not None:

                correction = (
                    KP * error
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

                # 중심선을 잃었을 때
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


        # =================================================
        # TURN_RIGHT
        # =================================================
        elif drive_state == "TURN_RIGHT":

            elapsed = (
                time.time()
                - turn_start_time
            )


            # 최소 회전 시간
            if elapsed < TURN_MIN_TIME:

                drive(
                    TURN_SPEED,
                    -TURN_SPEED
                )


            else:

                # =========================================
                # 새 중심선 확인
                # =========================================
                if (
                    target_x is not None
                    and
                    abs(
                        target_x
                        - w // 2
                    )
                    < RECOVERY_CENTER_RANGE
                ):

                    recovery_counter += 1

                else:

                    recovery_counter = 0


                # =========================================
                # 새 중심선 확인 완료
                # =========================================
                if (
                    recovery_counter
                    >= RECOVERY_FRAMES
                ):

                    stop()

                    drive_state = (
                        "FOLLOW"
                    )

                    recovery_counter = 0

                    last_intersection_time = (
                        time.time()
                    )

                    print(
                        "TURN_RIGHT -> FOLLOW"
                    )


                # 계속 우회전
                elif elapsed < TURN_MAX_TIME:

                    drive(
                        TURN_SPEED,
                        -TURN_SPEED
                    )


                # 너무 오래 돌면 정지
                else:

                    stop()

                    auto_mode = False

                    print(
                        "TURN TIMEOUT -> AUTO OFF"
                    )


    # =====================================================
    # 16. 교차로 잠금 해제
    # =====================================================
    if (
        drive_state == "FOLLOW"
        and
        width_ratio
        < INTERSECTION_WIDTH_EXIT
    ):

        intersection_locked = False


    # =====================================================
    # 17. 결과 화면
    # =====================================================
    result = frame.copy()


    # ROI 시작
    cv2.line(
        result,
        (0, roi_start),
        (w, roi_start),
        (0, 255, 255),
        2
    )


    # 화면 중앙
    cv2.line(
        result,
        (w // 2, roi_start),
        (w // 2, h),
        (0, 255, 0),
        2
    )


    # 중심점
    for x, y in center_points:

        cv2.circle(
            result,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


    # 중심선
    for i in range(
        len(center_points) - 1
    ):

        cv2.line(
            result,
            center_points[i],
            center_points[i + 1],
            (255, 0, 0),
            3
        )


    # 목표점
    if target_x is not None:

        cv2.circle(
            result,
            (
                int(target_x),
                int(target_y)
            ),
            9,
            (0, 255, 255),
            -1
        )


    # =====================================================
    # 화면 정보
    # =====================================================
    mode_text = (
        "AUTO"
        if auto_mode
        else "MANUAL"
    )


    cv2.putText(
        result,
        f"MODE: {mode_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"STATE: {drive_state}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"Width: {width_ratio:.2f}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )


    if error is not None:

        cv2.putText(
            result,
            f"Error: {error}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )


    if drive_state == "TURN_RIGHT":

        cv2.putText(
            result,
            "TURN RIGHT",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


    # =====================================================
    # 영상 출력
    # =====================================================
    cv2.imshow(
        "Pinky Centerline",
        result
    )


    # 원래 흰색 검출
    cv2.imshow(
        "White Mask",
        white_mask
    )


    # 내부 화살표/박스를 메운 결과
    cv2.imshow(
        "Road Mask Filled",
        road_mask
    )


    # =====================================================
    # 18. 키보드
    # =====================================================
    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # P = AUTO
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


    # W
    elif key == ord("w"):

        auto_mode = False

        drive(
            30,
            30
        )


    # S
    elif key == ord("s"):

        auto_mode = False

        drive(
            -30,
            -30
        )


    # A
    elif key == ord("a"):

        auto_mode = False

        drive(
            -25,
            25
        )


    # D
    elif key == ord("d"):

        auto_mode = False

        drive(
            25,
            -25
        )


    # SPACE
    elif key == 32:

        auto_mode = False

        stop()

        print("STOP")


    # R
    elif key == ord("r"):

        auto_mode = False

        drive_state = "FOLLOW"

        last_error = 0

        intersection_locked = False
        intersection_counter = 0
        last_intersection_time = -999.0

        turn_start_time = 0.0
        recovery_counter = 0

        stop()

        print("RESET")


    # ESC
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