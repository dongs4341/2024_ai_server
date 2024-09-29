from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
from werkzeug.utils import secure_filename
import mimetypes

app = Flask(__name__)

# YOLOv5 모델 로드 (.pt 파일 경로를 지정)
model_path = './pill_detection_model.pt'  # 알약 검출 모델 경로
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
except Exception as e:
    app.logger.error(f'Model loading error: {e}')
    model = None

@app.route('/')
def index():
    return "Pill Detection Server is running!"

@app.route('/detect_pill', methods=['POST'])
def detect_pill():
    if 'file' not in request.files:
        return jsonify({'error': "파일 부분이 없습니다"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': "선택된 파일이 없습니다"}), 400

    # 파일 MIME 타입 검사하여 이미지가 아니면 에러 반환
    mime_type = mimetypes.guess_type(file.filename)[0]
    if not mime_type or not mime_type.startswith('image'):
        return jsonify({'error': "업로드된 파일이 이미지가 아닙니다"}), 400

    # 이미지 파일 읽기
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({'error': f'이미지 파일을 처리하는 중 오류 발생: {e}'}), 400

    # 모델이 로드되지 않았다면, 에러 메시지 반환
    if model is None:
        return jsonify({'error': '모델 로딩 실패, 알약 탐지를 수행할 수 없습니다.'}), 500

    # 객체 탐지 수행
    try:
        results = model(img, size=640)
        results.render()  # 탐지된 객체를 이미지에 그리기
        pills_info = process_detected_pills(results)
    except Exception as e:
        return jsonify({'error': f'알약 탐지 중 오류 발생: {e}'}), 500

    if not pills_info:
        return jsonify({'result': 0, 'message': '알약을 검출하지 못했습니다.'}), 200
    else:
        return jsonify({'result': 1, 'pills': pills_info}), 200


# 알약 탐지 결과 처리 함수
def process_detected_pills(results):
    pill_details = []
    for i, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
        x1, y1, x2, y2 = map(int, map(lambda x: x.item(), xyxy))
        conf = conf.item()
        cls_id = int(cls.item())

        # 검출된 알약의 정보를 저장 (좌표, 정확도, 클래스 ID 등)
        pill_details.append({
            'class_id': cls_id,
            'confidence': conf,
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        })

    return pill_details


if __name__ == '__main__':
    app.run(debug=True, port=5001)