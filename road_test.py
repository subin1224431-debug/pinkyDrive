import cv2
import numpy as np
import requests


# =========================================================
# Pinky Pro 주소
# =========================================================
HOST = "http://192.168.4.1:8000"


# =========================================================
# 주행 설정
# =========================================================
BASE_SPEED = 25

# GitHub 공개코드의 P제어 개념 참고
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
# 직선 약 0.87
# 교차로 약 0.95
# =========================================================

# 교차로 진입 기준
INTERSECTION_ENTER = 0.93

# 교차로 탈출 기준
INTERSECTION_EXIT = 0.90

intersection_count = 0
intersection_locked = False

clear_counter = 0
CLEAR_FRAMES = 10


# =========================================================
# 자동주행 상태
# =========================================================
auto_mode = False

last_error = 0


# =========================================================
# Pinky 모터 명령
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
    # 2. 노이즈 완화
    # =====================================================
    roi_blur = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )


    # =====================================================
    # 3. HSV 변환
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
    # 6. 도로 내부 글씨/틈 메우기
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

    cleaned_mask = np.zeros_like(mask)


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


    # =====================================================
    # 8. 교차로 감지
    #
    # 흰색 영역 비율
    # =====================================================
    white_ratio = (
        cv2.countNonZero(mask)
        / mask.size
    )


    # =====================================================
    # 교차로 진입 / 탈출 기준을 다르게 설정
    #
    # 일반도로:
    # 0.87 정도
    #
    # 교차로:
    # 0.95 정도
    # =====================================================
    if intersection_locked:

        # 이미 교차로라면
        # 0.90 아래로 내려갈 때까지 유지
        is_intersection = (
            white_ratio
            > INTERSECTION_EXIT
        )

    else:

        # 일반도로에서는
        # 0.93 이상이 되어야 교차로 진입
        is_intersection = (
            white_ratio
            > INTERSECTION_ENTER
        )


    # =====================================================
    # 9. 교차로 카운트
    # =====================================================
    if (
        is_intersection
        and not intersection_locked
    ):

        intersection_count += 1

        intersection_locked = True

        clear_counter = 0

        print(
            f"INTERSECTION {intersection_count}"
        )


    # =====================================================
    # 교차로 탈출 확인
    # =====================================================
    if not is_intersection:

        clear_counter += 1

        if (
            clear_counter
            >= CLEAR_FRAMES
        ):

            intersection_locked = False

            clear_counter = 0

    else:

        clear_counter = 0


    # =====================================================
    # 10. 중심선 계산용 mask
    # =====================================================
    tracking_mask = mask.copy()


    # =====================================================
    # 11. 1번 교차로 직진 보정
    #
    # 오른쪽 갈림길 무시
    # =====================================================
    if (
        intersection_count == 1
        and intersection_locked
    ):

        cut_x = int(
            w * 0.72
        )

        tracking_mask[
            :,
            cut_x:
        ] = 0


    # =====================================================
    # 12. 중심점 계산
    # =====================================================
    roi_h, roi_w = (
        tracking_mask.shape
    )

    center_points = []

    prev_center = None


    for y in range(
        roi_h - 1,
        0,
        -STEP
    ):

        xs = np.where(
            tracking_mask[y] == 255
        )[0]


        if len(xs) == 0:
            continue


        # 흰색 영역 분리
        groups = np.split(
            xs,
            np.where(
                np.diff(xs) > 1
            )[0] + 1
        )


        # 너무 작은 영역 제거
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
                        (g[0] + g[-1])
                        // 2
                    )
                    - roi_w // 2
                )
            )


        # =================================================
        # 곡선 예측
        # =================================================
        else:

            predicted_x = prev_center


            if len(center_points) >= 2:

                x1 = center_points[-2][0]
                x2 = center_points[-1][0]

                dx = x2 - x1

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
                        (g[0] + g[-1])
                        // 2
                    )
                    - predicted_x
                )
            )


        x_left = chosen[0]
        x_right = chosen[-1]

        x_center = (
            x_left + x_right
        ) // 2


        # 갑자기 다른 영역으로 튀는 것 방지
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


        prev_center = x_center


        if (
            len(center_points)
            >= MAX_POINTS
        ):

            break


    # =====================================================
    # 13. 곡선 도로 2차 fitting
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


            center_points = smooth_points


        except Exception:

            pass


    # =====================================================
    # 14. 결과 화면
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


    # 카메라 중앙선
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


    # =====================================================
    # 15. Center Error 계산
    # =====================================================
    error = None


    allow_tracking = (
        not is_intersection
        or intersection_count == 1
    )


    if (
        len(center_points) >= 3
        and allow_tracking
    ):

        target_index = min(
            2,
            len(center_points) - 1
        )

        target_x, target_y = (
            center_points[
                target_index
            ]
        )


        error = (
            target_x
            - w // 2
        )


        last_error = error


        # 목표점
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
            0.7,
            (0, 0, 255),
            2
        )


    # =====================================================
    # 16. 자동 중심선 추종
    # =====================================================
    if auto_mode:

        # =================================================
        # 2번 / 3번 교차로
        #
        # 현재 코드에서는 우선 정지
        # =================================================
        if (
            is_intersection
            and intersection_count in [2, 3]
        ):

            stop()


        # =================================================
        # 정상 중심선 발견
        # =================================================
        elif error is not None:

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
        # 중심선을 잃었을 경우
        # =================================================
        else:

            if last_error < 0:

                # 왼쪽에서 마지막으로 봄
                drive(
                    0,
                    SEARCH_SPEED
                )


            elif last_error > 0:

                # 오른쪽에서 마지막으로 봄
                drive(
                    SEARCH_SPEED,
                    0
                )


            else:

                stop()


    # =====================================================
    # 17. 교차로 메시지
    # =====================================================
    if is_intersection:

        if intersection_count == 1:

            message = (
                "INTERSECTION 1 - STRAIGHT"
            )

        elif intersection_count == 2:

            message = (
                "INTERSECTION 2 - RIGHT"
            )

        elif intersection_count == 3:

            message = (
                "INTERSECTION 3 - RIGHT"
            )

        else:

            message = (
                f"INTERSECTION {intersection_count}"
            )


        cv2.putText(
            result,
            message,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2
        )


    # =====================================================
    # 18. 상태 표시
    # =====================================================
    mode_text = (
        "AUTO"
        if auto_mode
        else "MANUAL"
    )


    cv2.putText(
        result,
        f"MODE: {mode_text}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"Intersection: {intersection_count}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"White ratio: {white_ratio:.2f}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )


    # =====================================================
    # 19. 영상 출력
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
    # 20. 키보드
    # =====================================================
    key = cv2.waitKey(1) & 0xFF


    # AUTO ON/OFF
    if key == ord("p"):

        auto_mode = not auto_mode

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

        print("FORWARD")


    # 수동 후진
    elif key == ord("s"):

        auto_mode = False

        drive(
            -30,
            -30
        )

        print("BACKWARD")


    # 수동 좌회전
    elif key == ord("a"):

        auto_mode = False

        drive(
            -25,
            25
        )

        print("LEFT")


    # 수동 우회전
    elif key == ord("d"):

        auto_mode = False

        drive(
            25,
            -25
        )

        print("RIGHT")


    # 긴급 정지
    elif key == 32:

        auto_mode = False

        stop()

        print("STOP")


    # 교차로 번호 초기화
    elif key == ord("r"):

        intersection_count = 0
        intersection_locked = False
        clear_counter = 0

        print(
            "INTERSECTION RESET"
        )


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

