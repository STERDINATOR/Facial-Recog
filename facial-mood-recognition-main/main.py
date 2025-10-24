
"""Real-time face detection + emotion/age/gender overlay.

This cleaned version fixes duplicated code, avoids automatic large-model downloads by
default, and makes guarded downloads non-fatal (they won't raise exceptions). It uses
MediaPipe (if installed) for detection and FaceMesh for alignment, falling back to
OpenCV Haar cascades when necessary. Heavy inference (DeepFace) runs in a background
thread to keep the UI responsive.

Run under the project's virtualenv, e.g.:
  C:/path/to/.venv/Scripts/python.exe facial-mood-recognition-main/main.py

Press 'q' to quit.
"""

import time
import cv2
import numpy as np
import threading
import queue
import platform
import os
from deepface import DeepFace

try:
    import mediapipe as mp  # type: ignore
    USE_MEDIAPIPE = True
except Exception:
    mp = None
    USE_MEDIAPIPE = False

# --- Config (tweak these) ---
DOWN_SCALE = 0.6           # run detector on downscaled frame (0.4-0.8)
FRAME_SKIP_MIN = 2         # min frames to skip between analyses
FRAME_SKIP_MAX = 12        # max frames to skip between analyses
ANALYZE_SIZE = (160, 160)  # DeepFace input crop size (width, height)
TARGET_FPS = 18            # target UI FPS
FAST_MODE = False          # force fast mode if True
ANALYSIS_QUEUE_MAX = 2     # max pending analysis jobs

# By default we avoid downloading large external model files automatically.
# Set these to True only if you want the script to attempt downloads.
USE_DNN = False            # enable OpenCV DNN face detector (will attempt downloads when True)
USE_AGE_GENDER = False    # disable OpenCV age/gender models due to broken download links; rely on DeepFace for accurate gender detection

# Haar cascade fallback (available in OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# MediaPipe FaceMesh instance used for alignment (created lazily)
MP_FACE_MESH = None
if USE_MEDIAPIPE and mp is not None:
    try:
        MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5)
    except Exception:
        MP_FACE_MESH = None

# --- DNN model paths ---
DNN_PROTO = os.path.join(os.path.dirname(__file__), 'deploy.prototxt')
DNN_MODEL = os.path.join(os.path.dirname(__file__), 'res10_300x300_ssd_iter_140000.caffemodel')
_dnn_net = None
_dnn_attempted = False
_dnn_failed = False

AGE_PROTO = os.path.join(os.path.dirname(__file__), 'age_deploy.prototxt')
AGE_MODEL = os.path.join(os.path.dirname(__file__), 'age_net.caffemodel')
GENDER_PROTO = os.path.join(os.path.dirname(__file__), 'gender_deploy.prototxt')
GENDER_MODEL = os.path.join(os.path.dirname(__file__), 'gender_net.caffemodel')
_age_net = None
_gender_net = None
_age_gender_attempted = False
_age_gender_failed = False


def _download_file(url, dst):
    import urllib.request
    try:
        urllib.request.urlretrieve(url, dst)
        return True
    except Exception as e:
        print(f'Failed to download {url}:', e)
        return False


def ensure_dnn_model():
    """Try to download DNN model files but don't raise on failure; return True on success."""
    proto_candidates = [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
    ]
    model_candidates = [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel',
        'https://github.com/opencv/opencv_3rdparty/raw/master/res10_300x300_ssd_iter_140000.caffemodel',
        'https://github.com/opencv/opencv_3rdparty/raw/master/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    ]
    ok = True
    if not os.path.exists(DNN_PROTO):
        print('Downloading DNN prototxt...')
        success = False
        for url in proto_candidates:
            if _download_file(url, DNN_PROTO):
                success = True
                break
        ok = success and ok
    if not os.path.exists(DNN_MODEL):
        print('Downloading DNN caffemodel (may be large)...')
        success = False
        for url in model_candidates:
            if _download_file(url, DNN_MODEL):
                success = True
                break
        ok = success and ok
    return ok


def get_dnn_net():
    global _dnn_net, _dnn_attempted, _dnn_failed
    if _dnn_net is not None:
        return _dnn_net
    if not USE_DNN:
        return None
    if _dnn_attempted and _dnn_failed:
        return None
    _dnn_attempted = True
    try:
        ok = ensure_dnn_model()
        if not ok:
            _dnn_failed = True
            return None
        net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        _dnn_net = net
        return net
    except Exception as e:
        print('Failed to load DNN face detector:', e)
        _dnn_failed = True
        return None


def ensure_age_gender_models():
    age_proto_candidates = [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/age_deploy.prototxt',
        'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt'
    ]
    age_model_candidates = [
        'https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/age_net.caffemodel',
        'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/age_net.caffemodel'
    ]
    gender_proto_candidates = [
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/gender_deploy.prototxt',
        'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt'
    ]
    gender_model_candidates = [
        'https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/gender_net.caffemodel',
        'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/gender_net.caffemodel'
    ]
    ok = True
    if not os.path.exists(AGE_PROTO):
        print('Downloading age prototxt...')
        success = False
        for url in age_proto_candidates:
            if _download_file(url, AGE_PROTO):
                success = True
                break
        ok = success and ok
    if not os.path.exists(AGE_MODEL):
        print('Downloading age caffemodel (may be large)...')
        success = False
        for url in age_model_candidates:
            if _download_file(url, AGE_MODEL):
                success = True
                break
        ok = success and ok
    if not os.path.exists(GENDER_PROTO):
        print('Downloading gender prototxt...')
        success = False
        for url in gender_proto_candidates:
            if _download_file(url, GENDER_PROTO):
                success = True
                break
        ok = success and ok
    if not os.path.exists(GENDER_MODEL):
        print('Downloading gender caffemodel (may be large)...')
        success = False
        for url in gender_model_candidates:
            if _download_file(url, GENDER_MODEL):
                success = True
                break
        ok = success and ok
    return ok


def get_age_gender_nets():
    global _age_net, _gender_net, _age_gender_attempted, _age_gender_failed
    if _age_net is not None and _gender_net is not None:
        return _age_net, _gender_net
    if not USE_AGE_GENDER:
        return None, None
    if _age_gender_attempted and _age_gender_failed:
        return None, None
    _age_gender_attempted = True
    try:
        ok = ensure_age_gender_models()
        if not ok:
            _age_gender_failed = True
            return None, None
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
        _age_net = age_net
        _gender_net = gender_net
        return age_net, gender_net
    except Exception as e:
        print('Failed to load age/gender models:', e)
        _age_gender_failed = True
        return None, None


def predict_age_gender(age_net, gender_net, face_img):
    try:
        if age_net is None or gender_net is None:
            return None, None
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_list = ['Male', 'Female']
        gender = gender_list[int(np.argmax(gender_preds))]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        idx = int(np.argmax(age_preds))
        age_range = age_list[idx]
        if '-' in age_range:
            parts = age_range.strip('()').split('-')
            low = int(parts[0])
            high = int(parts[1])
            approx_age = int((low + high) / 2)
        else:
            approx_age = int(age_range.strip('()'))
        return approx_age, gender
    except Exception as e:
        print('Age/gender prediction failed:', e)
        return None, None


# Background analysis structures
analysis_queue = queue.Queue(maxsize=ANALYSIS_QUEUE_MAX)
analysis_lock = threading.Lock()
analysis_job = {'results': {}, 'centers': []}


def analyze_image(img_bgr):
    """Run DeepFace analyze on a single BGR crop, suppressing console output."""
    try:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            res = DeepFace.analyze(img_bgr, actions=['emotion', 'age', 'gender'], enforce_detection=False, detector_backend='opencv')
        if isinstance(res, list) and len(res) > 0:
            return res[0]
        return res
    except Exception as e:
        # short message
        print(f"DeepFace analyze error: {e}")
        return None


def analysis_worker():
    global MP_FACE_MESH
    # create face mesh lazily in worker if available to avoid blocking main thread
    if USE_MEDIAPIPE and mp is not None and MP_FACE_MESH is None:
        try:
            MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5)
        except Exception:
            MP_FACE_MESH = None

    while True:
        try:
            frame, faces = analysis_queue.get(timeout=0.2)
        except Exception:
            continue
        local_results = {}
        centers = []
        for (x1, y1, x2, y2) in faces:
            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            def align_face(img):
                # Prefer MediaPipe face mesh landmarks for robust alignment
                try:
                    if MP_FACE_MESH is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = MP_FACE_MESH.process(rgb)
                        if results and getattr(results, 'multi_face_landmarks', None):
                            lm = results.multi_face_landmarks[0]
                            h, w = img.shape[:2]
                            x_l = int(lm.landmark[33].x * w)
                            y_l = int(lm.landmark[33].y * h)
                            x_r = int(lm.landmark[263].x * w)
                            y_r = int(lm.landmark[263].y * h)
                            dx = x_r - x_l
                            dy = y_r - y_l
                            angle = np.degrees(np.arctan2(dy, dx))
                            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
                            return rotated
                except Exception:
                    pass
                # fallback to eye cascade
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    if len(eyes) >= 2:
                        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                        (ex1, ey1, ew1, eh1) = eyes[0]
                        (ex2, ey2, ew2, eh2) = eyes[1]
                        c1 = (int(ex1 + ew1/2), int(ey1 + eh1/2))
                        c2 = (int(ex2 + ew2/2), int(ey2 + eh2/2))
                        if c2[0] < c1[0]:
                            c1, c2 = c2, c1
                        dx = c2[0] - c1[0]
                        dy = c2[1] - c1[1]
                        angle = np.degrees(np.arctan2(dy, dx))
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
                        return rotated
                except Exception:
                    pass
                return img

            aligned = align_face(crop)
            try:
                small = cv2.resize(aligned, ANALYZE_SIZE)
            except Exception:
                small = aligned

            res = analyze_image(small)

            # optionally use OpenCV age/gender nets if available
            age_net, gender_net = get_age_gender_nets()
            if (age_net is not None and gender_net is not None) and (aligned is not None and aligned.size != 0):
                try:
                    face_for_ag = cv2.resize(aligned, (227, 227))
                    pred_age, pred_gender = predict_age_gender(age_net, gender_net, face_for_ag)
                    if res is None:
                        res = {}
                    if pred_age is not None:
                        res['age'] = pred_age
                    if pred_gender is not None:
                        res['gender'] = pred_gender
                except Exception as e:
                    print(f"Age/gender prediction error: {e}")
                    pass
            else:
                # Fallback to DeepFace age/gender if OpenCV models not available
                if res is None:
                    res = {}
                if 'age' not in res or res['age'] is None:
                    try:
                        age_res = DeepFace.analyze(small, actions=['age'], enforce_detection=False, detector_backend='opencv')
                        if isinstance(age_res, list) and len(age_res) > 0:
                            res['age'] = age_res[0].get('age')
                        elif isinstance(age_res, dict):
                            res['age'] = age_res.get('age')
                    except Exception as e:
                        print(f"DeepFace age fallback error: {e}")
                if 'gender' not in res or res['gender'] is None:
                    try:
                        gender_res = DeepFace.analyze(small, actions=['gender'], enforce_detection=False, detector_backend='opencv')
                        if isinstance(gender_res, list) and len(gender_res) > 0:
                            res['gender'] = gender_res[0].get('dominant_gender')
                        elif isinstance(gender_res, dict):
                            res['gender'] = gender_res.get('dominant_gender')
                        else:
                            res['gender'] = None
                    except Exception as e:
                        print(f"DeepFace gender fallback error: {e}")

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy))
            local_results[(cx, cy)] = res

        with analysis_lock:
            analysis_job['results'] = local_results
            analysis_job['centers'] = centers


def start_background_worker():
    t = threading.Thread(target=analysis_worker, daemon=True)
    t.start()


def detect_faces(frame, down_scale=DOWN_SCALE, mp_detector=None):
    boxes = []
    # Prefer MediaPipe if available
    if USE_MEDIAPIPE and mp is not None and mp_detector is not None:
        try:
            small = cv2.resize(frame, (0, 0), fx=down_scale, fy=down_scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = mp_detector.process(rgb_small)
            if getattr(results, 'detections', None):
                h, w = small.shape[:2]
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    # scale to original coords
                    x1 = max(0, int(x1 / down_scale))
                    y1 = max(0, int(y1 / down_scale))
                    x2 = min(frame.shape[1], int(x2 / down_scale))
                    y2 = min(frame.shape[0], int(y2 / down_scale))
                    pad = int(0.12 * max(x2 - x1, y2 - y1))
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(frame.shape[1], x2 + pad)
                    y2 = min(frame.shape[0], y2 + pad)
                    boxes.append((x1, y1, x2, y2))
        except Exception:
            boxes = []

    # try OpenCV DNN face detector if enabled and MediaPipe didn't find anything
    if len(boxes) == 0:
        net = get_dnn_net()
        if net is not None:
            try:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                for i in range(0, detections.shape[2]):
                    conf = float(detections[0, 0, i, 2])
                    if conf > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype('int')
                        pad = int(0.12 * max(x2 - x1, y2 - y1))
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        boxes.append((x1, y1, x2, y2))
            except Exception:
                boxes = []

    # Haar cascade fallback
    if len(boxes) == 0:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                pad = int(0.12 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                boxes.append((x1, y1, x2, y2))
        except Exception:
            pass

    return boxes


def main_loop():
    frame_count = 0
    last_time = time.time()
    fps_ema = TARGET_FPS
    frame_skip = FRAME_SKIP_MIN
    recovery_attempts = 0

    def open_video_capture(idx=0):
        if platform.system() == 'Windows':
            for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                try:
                    cap_try = cv2.VideoCapture(idx, api)
                    if cap_try.isOpened():
                        print(f'Opened camera with API {api}')
                        return cap_try
                except Exception:
                    continue
        return cv2.VideoCapture(idx)

    cap = open_video_capture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check your camera and permissions.")

    mp_detector = None
    if USE_MEDIAPIPE and mp is not None:
        try:
            mp_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception:
            mp_detector = None

    start_background_worker()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                recovery_attempts += 1
                print(f"Failed to grab frame from webcam (attempt {recovery_attempts})")
                time.sleep(0.2)
                if recovery_attempts > 6:
                    break
                cap.release()
                time.sleep(0.1)
                cap.open(0)
                continue
            recovery_attempts = 0

            frame_count += 1
            frame = cv2.flip(frame, 1)

            boxes = detect_faces(frame, down_scale=DOWN_SCALE, mp_detector=mp_detector)

            now = time.time()
            fps = 1.0 / (now - last_time) if now != last_time else TARGET_FPS
            fps_ema = 0.85 * fps_ema + 0.15 * fps
            last_time = now

            if not FAST_MODE:
                if fps_ema < TARGET_FPS - 2 and frame_skip < FRAME_SKIP_MAX:
                    frame_skip += 1
                elif fps_ema > TARGET_FPS + 2 and frame_skip > FRAME_SKIP_MIN:
                    frame_skip -= 1

            if len(boxes) > 0 and frame_count % frame_skip == 0:
                try:
                    analysis_queue.put_nowait((frame.copy(), boxes))
                except queue.Full:
                    try:
                        _ = analysis_queue.get_nowait()
                        analysis_queue.put_nowait((frame.copy(), boxes))
                    except Exception:
                        pass

            with analysis_lock:
                results_snapshot = dict(analysis_job.get('results', {}))
                centers_snapshot = list(analysis_job.get('centers', []))

            for (x1, y1, x2, y2) in boxes:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                label = ""
                best_key = None
                best_dist = None
                for (rcx, rcy) in centers_snapshot:
                    d = (rcx - cx) ** 2 + (rcy - cy) ** 2
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_key = (rcx, rcy)

                if best_key and best_dist is not None and best_dist < ((x2 - x1) ** 2) * 4:
                    res = results_snapshot.get(best_key)
                    if res:
                        emotion = res.get('dominant_emotion') or res.get('emotion') or ''
                        age = res.get('age')
                        gender = res.get('dominant_gender')
                        if isinstance(age, (int, float)) and age > 0:
                            age_str = f", {int(age)}"
                        else:
                            age_str = ""
                        if gender and isinstance(gender, str) and gender in ['Man', 'Woman']:
                            gender_str = f", {gender}"
                        else:
                            gender_str = ""
                        label = f"{emotion}{age_str}{gender_str}"

                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"FPS: {fps_ema:.1f} (skip {frame_skip})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('Webcam with DeepFace', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if USE_MEDIAPIPE and mp is not None and 'mp_detector' in locals() and mp_detector is not None:
            try:
                mp_detector.close()
            except Exception:
                pass


if __name__ == '__main__':
    main_loop()
