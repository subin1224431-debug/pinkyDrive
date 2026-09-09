import cv2
import numpy as np
import requests


# =========================================================
# Pinky Pro 주소
# =========================================================
HOST = "http://192.168.4.1:8000"


# =========================================================
# 모터 명령
# =========================================================
def drive(left, right):
    try:
        requests.post(
            HOST + "/drive",
            json={
                "l": int(left),
                "r": int(right)
            },
            timeout=0.3
        )
    except requests.RequestException:
        pass


def stop():
    try:
        requests.post(
            HOST + "/stop",
            json={},
            timeout=0.3
        )
    except requests.RequestException:
        pass


# =========================================================
# 카메라 연결
# =========================================================
cap = cv2.VideoCapture(HOST + "/video")

if not cap.isOpened():
    print("로봇 카메라 연결 실패")
    exit()


# =========================================================
# 교차로 관련 변수
# =========================================================
intersection_count = 0

# 같은 교차로를 여러 번 세지 않도록 잠금
intersection_locked = False

# 교차로에서 완전히 빠져나왔는지 확인
clear_counter = 0

# 연속 일반도로 프레임 수
CLEAR_FRAMES = 10


print("""
==================================
Pinky Pro Road Test
==================================

W : 전진
S : 후진
A : 좌회전
D : 우회전

SPACE : 정지
ESC : 종료
""")


# =========================================================
# 메인 루프
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("카메라 영상 수신 실패")
        break


    h, w = frame.shape[:2]


    # =====================================================
    # 1. ROI 설정
    #
    # 카메라가 위쪽을 많이 보기 때문에
    # 아래쪽 42%만 사용
    # =====================================================
    roi_start = int(h * 0.58)

    roi = frame[
        roi_start:h,
        :
    ]


    # =====================================================
    # 2. Gaussian Blur
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
    # 4. 흰색 도로 검출
    # =====================================================
    lower_white = np.array(
        [0, 0, 175]
    )

    upper_white = np.array(
        [180, 75, 255]
    )

    mask = cv2.inRange(
        hsv,
        lower_white,
        upper_white
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
    # 6. 도로 내부 글씨 / 검은 틈 메우기
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

    MIN_AREA = 500

    for i in range(1, num_labels):

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
    # 8. 교차로 감지용 White Ratio
    #
    # 중요:
    # 여기서는 아직 1번 교차로 오른쪽을 자르기 전 mask를 사용
    # =====================================================
    white_pixels = cv2.countNonZero(mask)

    total_pixels = mask.size

    white_ratio = (
        white_pixels
        / total_pixels
    )


    # 실제 경기장에서 나중에 조정
    INTERSECTION_RATIO = 0.68

    is_intersection = (
        white_ratio > INTERSECTION_RATIO
    )


    # =====================================================
    # 9. 교차로 카운트
    # =====================================================

    # 새로운 교차로 진입
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
    # 교차로에서 빠져나왔는지 확인
    # =====================================================
    if not is_intersection:

        clear_counter += 1

        # 일정 프레임 동안 일반도로가 나오면
        # 다음 교차로를 인식할 수 있게 잠금 해제
        if clear_counter >= CLEAR_FRAMES:

            intersection_locked = False

            clear_counter = 0

    else:

        clear_counter = 0


    # =====================================================
    # 10. 중심선 계산용 Mask 따로 복사
    # =====================================================
    tracking_mask = mask.copy()


    # =====================================================
    # 11. 1번 교차로 직진 보정
    #
    # 1번 교차로에서는 오른쪽으로 갈라지는 흰색 도로를
    # 중심선 계산에서 제외
    #
    # 교차로 감지 자체는 원래 mask로 이미 끝났기 때문에
    # tracking_mask만 수정함
    # =====================================================
    if (
        intersection_count == 1
        and intersection_locked
    ):

        cut_x = int(
            tracking_mask.shape[1] * 0.72
        )

        tracking_mask[
            :,
            cut_x:
        ] = 0


    # =====================================================
    # 12. 중심선 계산
    # =====================================================
    roi_h, roi_w = tracking_mask.shape

    center_points = []

    STEP = 15

    MIN_WIDTH = 25

    # 중심점이 갑자기 튀는 것 제한
    MAX_JUMP = 60

    # 가까운 도로만 사용
    MAX_POINTS = 8

    prev_center = None


    # 아래쪽부터 위쪽으로 검색
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


        # =================================================
        # 한 줄에 흰색 영역이 여러 개 있을 수 있으므로
        # 연속된 그룹으로 분리
        # =================================================
        groups = np.split(
            xs,
            np.where(
                np.diff(xs) > 1
            )[0] + 1
        )


        # 너무 작은 흰색 영역 제거
        groups = [
            g
            for g in groups
            if len(g) >= MIN_WIDTH
        ]

        if len(groups) == 0:
            continue


        # =================================================
        # 첫 번째 중심점
        #
        # 화면 중앙에 가장 가까운 도로 선택
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
        # 두 번째 이후
        #
        # 곡선일 경우 이전 중심점 이동 방향을 보고
        # 다음 중심 위치를 예상
        # =================================================
        else:

            predicted_x = prev_center


            # 중심점이 2개 이상 있으면
            # 최근 방향으로 다음 위치 예상
            if len(center_points) >= 2:

                x1 = center_points[-2][0]

                x2 = center_points[-1][0]

                dx = x2 - x1

                # 너무 크게 예측하지 않도록 제한
                dx = int(
                    np.clip(
                        dx,
                        -35,
                        35
                    )
                )

                predicted_x = x2 + dx


            # 예상 위치와 가장 가까운 흰색 영역 선택
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


        # =================================================
        # 갑자기 다른 도로로 튀는 것 방지
        # =================================================
        if prev_center is not None:

            if abs(
                x_center - prev_center
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


        # 가까운 도로만 사용
        if len(center_points) >= MAX_POINTS:
            break


    # =====================================================
    # 13. 곡선 도로 보정
    #
    # 중심점들을 2차 곡선으로 fitting
    # 직선이면 거의 직선으로,
    # 원형/곡선이면 부드러운 곡선으로 만들어줌
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


            # x = a*y^2 + b*y + c
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


                # 화면 밖으로 나가지 않도록 제한
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
            # fitting에 문제가 있으면
            # 원래 중심점 사용
            pass


    # =====================================================
    # 14. 결과 영상
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
        (w // 2, h),
        (w // 2, roi_start),
        (0, 255, 0),
        2
    )


    # =====================================================
    # 1번 교차로에서 오른쪽 무시 영역 표시
    # =====================================================
    if (
        intersection_count == 1
        and intersection_locked
    ):

        cut_x_display = int(
            w * 0.72
        )

        cv2.line(
            result,
            (cut_x_display, roi_start),
            (cut_x_display, h),
            (0, 165, 255),
            2
        )


        cv2.putText(
            result,
            "IGNORE RIGHT AREA",
            (max(10, cut_x_display - 190),
             roi_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            2
        )


    # =====================================================
    # 중심점 표시
    # =====================================================
    for x, y in center_points:

        cv2.circle(
            result,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


    # =====================================================
    # 중심선 표시
    # =====================================================
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
    # 15. 일반 도로 또는 1번 교차로
    #
    # 1번 교차로는 직진해야 하므로
    # 보정된 중심선을 계속 사용
    # =====================================================
    allow_center_tracking = (
        not is_intersection
        or intersection_count == 1
    )


    if (
        len(center_points) >= 3
        and allow_center_tracking
    ):

        # 아래에서 세 번째 점을 목표점으로 사용
        target_index = min(
            2,
            len(center_points) - 1
        )


        target_x, target_y = (
            center_points[target_index]
        )


        error = (
            target_x
            - w // 2
        )


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
            0.75,
            (0, 0, 255),
            2
        )


    # =====================================================
    # 16. 교차로별 안내
    # =====================================================
    if is_intersection:

        # 1번 교차로
        if intersection_count == 1:

            message = (
                "INTERSECTION 1 - GO STRAIGHT"
            )


        # 2번 교차로
        elif intersection_count == 2:

            message = (
                "INTERSECTION 2 - TURN RIGHT"
            )


        # 3번 교차로
        elif intersection_count == 3:

            message = (
                "INTERSECTION 3 - TURN RIGHT"
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
            0.60,
            (0, 0, 255),
            2
        )


    # =====================================================
    # 17. 상태 표시
    # =====================================================
    cv2.putText(
        result,
        f"Intersection count: {intersection_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 0, 0),
        2
    )


    cv2.putText(
        result,
        f"White ratio: {white_ratio:.2f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 0, 0),
        2
    )


    # =====================================================
    # 18. 화면 출력
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
    # 19. 키보드 수동 주행
    # =====================================================
    key = cv2.waitKey(1) & 0xFF


    # W = 전진
    if key == ord("w"):

        drive(30, 30)

        print("FORWARD")


    # S = 후진
    elif key == ord("s"):

        drive(-30, -30)

        print("BACKWARD")


    # A = 좌회전
    elif key == ord("a"):

        drive(-25, 25)

        print("LEFT")


    # D = 우회전
    elif key == ord("d"):

        drive(25, -25)

        print("RIGHT")


    # SPACE = 정지
    elif key == 32:

        stop()

        print("STOP")


    # ESC = 종료
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