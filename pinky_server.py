#!/usr/bin/env python3

import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from pinkylib import Camera, Motor


app = Flask(__name__)

# pinkylib 동시 접근 방지
bus = threading.Lock()

# 모터
motor = Motor()
motor.enable_motor()

# 카메라
cam = Camera()
cam.start()

# 현재 주행 상태
drive_state = {
    "l": 0,
    "r": 0,
    "last": 0.0
}


# =====================================
# 데드맨
# 1초 동안 명령이 없으면 자동 정지
# =====================================
def watchdog():

    while True:

        time.sleep(0.1)

        if (
            drive_state["l"] == 0
            and drive_state["r"] == 0
        ):
            continue

        if time.time() - drive_state["last"] > 1.0:

            with bus:
                motor.stop()

            drive_state["l"] = 0
            drive_state["r"] = 0

            print("데드맨 정지", flush=True)


threading.Thread(
    target=watchdog,
    daemon=True
).start()


# =====================================
# 카메라 영상 스트림
# =====================================
def mjpeg():

    while True:

        with bus:

            frame = cam.get_frame()

            if frame is None:
                continue

            frame = cv2.resize(
                frame,
                (640, 480)
            )

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

        if ok:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

        time.sleep(0.06)


# =====================================
# /video
# =====================================
@app.get("/video")
def video():

    return Response(
        mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =====================================
# /drive
# =====================================
@app.post("/drive")
def drive():

    data = request.get_json(force=True)

    left = max(
        -100,
        min(100, int(data.get("l", 0)))
    )

    right = max(
        -100,
        min(100, int(data.get("r", 0)))
    )

    with bus:

        if left == 0 and right == 0:
            motor.stop()

        else:
            motor.move(left, right)

    drive_state["l"] = left
    drive_state["r"] = right
    drive_state["last"] = time.time()

    return jsonify(
        {
            "l": left,
            "r": right
        }
    )


# =====================================
# /stop
# =====================================
@app.post("/stop")
def stop():

    with bus:
        motor.stop()

    drive_state["l"] = 0
    drive_state["r"] = 0

    return jsonify(
        {
            "stopped": True
        }
    )


# =====================================
# /health
# =====================================
@app.get("/health")
def health():

    return jsonify(
        {
            "ok": True
        }
    )


# =====================================
# 서버 시작
# =====================================
if __name__ == "__main__":

    try:

        app.run(
            host="0.0.0.0",
            port=8000,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    finally:

        motor.stop()
        motor.disable_motor()
        motor.close()
        cam.close()