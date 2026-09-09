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

# 기존에 잘 됐던 P제어 유지
KP = 0.12

MAX_SPEED = 40

# 중심선을 잃었을 때 탐색 속도
SEARCH_SPEED = 18


# =========================================================
# 영상 처리 설정
# =========================================================
ROI_START_RATIO = 0.58

LOWER_WHITE = np.array([0, 0, 175])
UPPER_WHITE = np.array([180, 75, 255])

MIN_AREA = 500

STEP = 15
MIN_WIDTH = 25
MAX_JUMP = 60
MAX_POINTS = 8


# =========================================================
# 교차로 설정
#
# 네 실제 측정값
#
# 직선 White ratio ≈ 0.87
# 교차로 White ratio ≈ 0.95
# =========================================================

# 흰색 비율
INTERSECTION_WHITE_ENTER = 0.92
INTERSECTION_WHITE_EXIT = 0.89

# 도로 폭 비율
#
# 1.0 = 화면 ROI 폭 전체
# 처음에는 0.88로 테스트
INTERSECTION_WIDTH_ENTER = 0.88
INTERSECTION_WIDTH_EXIT = 0.82


# 교차로 한 번 인식 후
# 3초 동안 다음 교차로 카운트 금지
INTERSECTION_COOLDOWN = 3.0


intersection_count = 0

intersection_locked = False

last_intersection_time = -999.0

clear_counter = 0

CLEAR_FRAMES = 8


# =========================================================
# 자동주행
# =========================================================
auto_mode = False

last_error = 0


# =========================================================
# 2번 / 3번 교차로 우회전
# =========================================================
drive_state = "FOLLOW"

TURN_SPEED = 30

# 약 40도 제자리 회전
# 실제 로봇에서 조정
TURN_40_TIME = 0.65

turn_start_time = 0.0


# =========================================================
# Pinky 모터 명령
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
# 카메라 연결
# =========================================================
cap = cv2.VideoCapture(
    HOST + "/video"
)

if not cap.isOpened():

    print("카메라 연결 실패")
    exit()


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
R       : 교차로 카운트 초기화
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


    # =====================================================
    # 2. Blur
    # =====================================================
    roi_blur = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )


    # =====================================================
    # 3. HSV
    # =====================================================
    hsv = cv2.cvtColor(
        roi_blur,
        cv2.COLOR_BGR2HSV
    )


    # =====================================================
    # 4. 흰색 도로 이진화
    # =====================================================
    mask = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )


    # =====================================================
    # 5. 작은 노이즈 제거
    # =====================================================
    kernel_open = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1
    )


    # =====================================================
    # 6. 도로 내부 틈 메우기
    # =====================================================
    kernel_close = np.ones(
        (17, 17),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2
    )


    # =====================================================
    # 7. 작은 흰색 덩어리 제거
    # =====================================================
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned_mask = np.zeros_like(
        mask
    )


    for i in range(
        1,
        num_labels
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_AREA:

            cleaned_mask[
                labels == i
            ] = 255


    mask = cleaned_mask


    roi_h, roi_w = mask.shape


    # =====================================================
    # 8. White Ratio
    # =====================================================
    white_ratio = (
        cv2.countNonZero(mask)
        / mask.size
    )


    # =====================================================
    # 9. 도로 폭 계산
    #
    # 여러 가로줄에서 흰색 도로 폭 계산
    # =====================================================
    road_widths = []


    for y in range(
        roi_h - 1,
        0,
        -15
    ):

        xs = np.where(
            mask[y] == 255
        )[0]


        if len(xs) < MIN_WIDTH:

            continue


        # 흰색이 여러 덩어리라면 분리
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


        # 가장 넓은 흰색 도로 사용
        largest_group = max(
            groups,
            key=len
        )


        road_width = (
            largest_group[-1]
            - largest_group[0]
        )


        road_widths.append(
            road_width
        )


    # =====================================================
    # 평균 도로 폭
    # =====================================================
    if len(road_widths) > 0:

        avg_road_width = float(
            np.mean(
                road_widths
            )
        )

    else:

        avg_road_width = 0.0


    # =====================================================
    # 픽셀 대신 화면 폭 대비 비율로 사용
    #
    # 예:
    # 0.70 = ROI 폭의 70%
    # 0.95 = ROI 폭의 95%
    # =====================================================
    width_ratio = (
        avg_road_width
        / roi_w
    )


    # =====================================================
    # 10. 교차로 감지
    #
    # White ratio + Width ratio
    # 둘 다 만족해야 교차로
    # =====================================================
    if intersection_locked:

        is_intersection = (
            white_ratio
            > INTERSECTION_WHITE_EXIT
            and
            width_ratio
            > INTERSECTION_WIDTH_EXIT
        )

    else:

        is_intersection = (
            white_ratio
            > INTERSECTION_WHITE_ENTER
            and
            width_ratio
            > INTERSECTION_WIDTH_ENTER
        )


    # =====================================================
    # 11. 교차로 카운트
    # =====================================================
    now = time.time()


    if (
        is_intersection
        and
        not intersection_locked
        and
        now - last_intersection_time
        >= INTERSECTION_COOLDOWN
        and
        drive_state == "FOLLOW"
    ):

        intersection_count += 1

        intersection_locked = True

        last_intersection_time = now

        clear_counter = 0


        print(
            f"INTERSECTION {intersection_count}"
        )


        # =================================================
        # 1번 교차로
        # → 직진
        # =================================================
        if intersection_count == 1:

            drive_state = "INTERSECTION_1"

            print(
                "INTERSECTION 1 -> STRAIGHT"
            )


        # =================================================
        # 2번 / 3번
        # → 정지 후 제자리 40도 우회전
        # =================================================
        elif intersection_count in [2, 3]:

            stop()

            drive_state = "TURN_RIGHT"

            turn_start_time = (
                time.time()
            )

            print(
                f"INTERSECTION {intersection_count}"
                " -> RIGHT TURN"
            )


    # =====================================================
    # 12. 교차로 탈출
    # =====================================================
    if not is_intersection:

        clear_counter += 1


        if clear_counter >= CLEAR_FRAMES:

            intersection_locked = False

            clear_counter = 0


            # 1번 교차로를 완전히 빠져나옴
            if drive_state == "INTERSECTION_1":

                drive_state = "FOLLOW"

                print(
                    "INTERSECTION 1 CLEAR"
                )

    else:

        clear_counter = 0


    # =====================================================
    # 13. 중심선용 mask
    # =====================================================
    tracking_mask = mask.copy()


    # =====================================================
    # 14. 1번 교차로 직진 보정
    #
    # 오른쪽 갈림길 무시
    # =====================================================
    if drive_state == "INTERSECTION_1":

        cut_x = int(
            roi_w * 0.72
        )

        tracking_mask[
            :,
            cut_x:
        ] = 0


    # =====================================================
    # 15. 중심점 계산
    # =====================================================
    center_points = []

    prev_center = None


    for y in range(
        roi_h - 1,
        0,
        -STEP
    ):

        xs = np.where(
            tracking_mask[y]
            == 255
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
        # 첫 중심점
        # =================================================
        if prev_center is None:

            chosen = min(
                groups,
                key=lambda g:
                abs(
                    (
                        (
                            g[0]
                            + g[-1]
                        )
                        // 2
                    )
                    - roi_w // 2
                )
            )


        # =================================================
        # 곡선 예측
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
                        (
                            g[0]
                            + g[-1]
                        )
                        // 2
                    )
                    - predicted_x
                )
            )


        x_left = chosen[0]

        x_right = chosen[-1]


        x_center = (
            x_left
            + x_right
        ) // 2


        # =================================================
        # 튀는 중심점 제거
        # =================================================
        if prev_center is not None:

            if abs(
                x_center
                - prev_center
            ) > MAX_JUMP:

                continue


        original_y = (
            y + roi_start
        )


        center_points.append(
            (
                int(x_center),
                int(original_y)
            )
        )


        prev_center = (
            x_center
        )


        if (
            len(center_points)
            >= MAX_POINTS
        ):

            break


    # =====================================================
    # 16. 기존 2차 곡선 fitting
    #
    # 네가 이 버전이 가장 잘 된다고 했으므로 유지
    # =====================================================
    if len(center_points) >= 5:

        try:

            ys = np.array(
                [
                    p[1]
                    for p in center_points
                ],
                dtype=np.float32
            )


            xs = np.array(
                [
                    p[0]
                    for p in center_points
                ],
                dtype=np.float32
            )


            coeff = np.polyfit(
                ys,
                xs,
                2
            )


            smooth_points = []


            for y_value in ys:

                x_value = (
                    coeff[0]
                    * y_value
                    * y_value

                    + coeff[1]
                    * y_value

                    + coeff[2]
                )


                x_value = np.clip(
                    x_value,
                    0,
                    w - 1
                )


                smooth_points.append(
                    (
                        int(x_value),
                        int(y_value)
                    )
                )


            center_points = (
                smooth_points
            )


        except Exception:

            pass


    # =====================================================
    # 17. Center Error
    # =====================================================
    error = None


    allow_tracking = (
        drive_state == "FOLLOW"
        or
        drive_state == "INTERSECTION_1"
    )


    if (
        len(center_points) >= 3
        and
        allow_tracking
    ):

        target_index = 2


        target_x, target_y = (
            center_points[
                target_index
            ]
        )


        error = (
            target_x
            - w // 2
        )


        last_error = (
            error
        )


    # =====================================================
    # 18. 자동주행
    # =====================================================
    if auto_mode:


        # =================================================
        # 2 / 3번
        # 제자리 우회전
        # =================================================
        if drive_state == "TURN_RIGHT":

            elapsed = (
                time.time()
                - turn_start_time
            )


            if elapsed < TURN_40_TIME:

                drive(
                    TURN_SPEED,
                    -TURN_SPEED
                )


            else:

                stop()

                drive_state = "FOLLOW"

                intersection_locked = True

                # 회전 직후 다시 같은 교차로 세는 것 방지
                last_intersection_time = (
                    time.time()
                )


                print(
                    "RIGHT TURN COMPLETE"
                )


        # =================================================
        # 일반 주행 / 1번 교차로
        # =================================================
        elif error is not None:

            # 기존 P 제어 유지
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


            left_speed = np.clip(
                left_speed,
                0,
                MAX_SPEED
            )


            right_speed = np.clip(
                right_speed,
                0,
                MAX_SPEED
            )


            drive(
                left_speed,
                right_speed
            )


        # =================================================
        # 중심선 잃음
        # =================================================
        else:

            if drive_state == "INTERSECTION_1":

                drive(
                    18,
                    18
                )


            elif last_error < 0:

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


    # =====================================================
    # 19. 결과 화면
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


    # 화면 중앙선
    cv2.line(
        result,
        (
            w // 2,
            roi_start
        ),
        (
            w // 2,
            h
        ),
        (0, 255, 0),
        2
    )


    # =====================================================
    # 1번 교차로 오른쪽 무시선
    # =====================================================
    if drive_state == "INTERSECTION_1":

        cut_x_display = int(
            w * 0.72
        )


        cv2.line(
            result,
            (
                cut_x_display,
                roi_start
            ),
            (
                cut_x_display,
                h
            ),
            (0, 165, 255),
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
    if (
        error is not None
        and
        len(center_points) >= 3
    ):

        target_x, target_y = (
            center_points[2]
        )


        cv2.circle(
            result,
            (
                target_x,
                target_y
            ),
            9,
            (0, 255, 255),
            -1
        )


        cv2.putText(
            result,
            f"Center Error: {error}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


    # =====================================================
    # 20. 상태 표시
    # =====================================================
    mode_text = (
        "AUTO"
        if auto_mode
        else "MANUAL"
    )


    cv2.putText(
        result,
        f"MODE: {mode_text}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"STATE: {drive_state}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"Intersection: {intersection_count}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"White ratio: {white_ratio:.2f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"Width ratio: {width_ratio:.2f}",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 0, 0),
        2
    )


    if is_intersection:

        cv2.putText(
            result,
            "INTERSECTION DETECTED",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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


    cv2.imshow(
        "Road Mask",
        tracking_mask
    )


    # =====================================================
    # 21. 키보드
    # =====================================================
    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # P = 자동 중심선 추종
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


    elif key == ord("w"):

        auto_mode = False

        drive(
            30,
            30
        )


    elif key == ord("s"):

        auto_mode = False

        drive(
            -30,
            -30
        )


    elif key == ord("a"):

        auto_mode = False

        drive(
            -25,
            25
        )


    elif key == ord("d"):

        auto_mode = False

        drive(
            25,
            -25
        )


    elif key == 32:

        auto_mode = False

        stop()

        print("STOP")


    elif key == ord("r"):

        intersection_count = 0

        intersection_locked = False

        clear_counter = 0

        last_intersection_time = -999.0

        drive_state = "FOLLOW"

        stop()


        print(
            "INTERSECTION RESET"
        )


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