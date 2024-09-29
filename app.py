from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
from pyzbar.pyzbar import decode
import mimetypes
from werkzeug.utils import secure_filename

app = Flask(__name__)

# YOLOv5 모델 로드
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolo5.pt', force_reload=True)
except Exception as e:
    app.logger.error(f'Model loading error: {e}')
    model = None

@app.route('/')
def index():
    return "AI Server is running!"

@app.route('/detect_fixed', methods=['POST'])
def detect_fixed():
    print("접속 성공")
    if 'file' not in request.files:
        return jsonify({'error': "파일 부분이 없습니다"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': "선택된 파일이 없습니다"}), 400

    # 파일의 MIME 타입을 확인합니다.
    if not file.content_type.startswith('image/'):
        return jsonify({'error': '허용되지 않는 파일 형식입니다'}), 400

    # 항상 같은 바코드 번호 반환
    return jsonify({'barcode': "880123456893"}), 200

@app.route('/detect_barcode', methods=['POST'])
def detect_barcode():
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
    except UnidentifiedImageError as e:
        return jsonify({'error': f'이미지 파일을 인식할 수 없습니다: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'파일 처리 중 오류 발생: {e}'}), 400

    # 바코드 숫자 추출
    try:
        barcodes = decode(img)
    except PyZbarError as e:
        return jsonify({'error': f'바코드 추출 중 오류 발생: {e}'}), 400

    barcode_numbers = [barcode.data.decode('utf-8') for barcode in barcodes]

    if not barcode_numbers:
        return jsonify({'result': 0, 'message': 'No barcodes detected'}), 200
    else:
        return jsonify({'result': 1, 'barcodes': barcode_numbers}), 200

@app.route('/detect_ai', methods=['POST'])
def detect_ai():
    if 'file' not in request.files:
        return jsonify({'error': "파일 부분이 없습니다"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': "선택된 파일이 없습니다"}), 400

    # 이미지 파일 읽기
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as e:
        return jsonify({'error': f'이미지 파일을 인식할 수 없습니다: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'파일 처리 중 오류 발생: {e}'}), 400

    # 파일 MIME 타입 검사하여 이미지가 아니면 에러 반환
    mime_type = mimetypes.guess_type(file.filename)[0]
    if not mime_type or not mime_type.startswith('image'):
        return jsonify({'error': "업로드된 파일이 이미지가 아닙니다"}), 400


    # 모델이 로드되지 않았다면, 에러 메시지 반환
    if model is None:
        return jsonify({'error': '모델 로딩 실패, 객체 탐지를 수행할 수 없습니다.'}), 500


    # 객체 탐지 수행
    try:
        results = model(img, size=640)
        results.render()  # 탐지된 객체를 이미지에 그리기
    except Exception as e:
        return jsonify({'error': f'객체 탐지 중 오류 발생: {e}'}), 500

    high_conf_barcodes = process_detected_objects(results, img)

    if not high_conf_barcodes:  # 정확도가 0.6 이상인 바코드를 인식하지 못한 경우
        return jsonify({'result': 0, 'message': 'No high-confidence barcodes detected'}), 200
    else:   # 정확도가 0.6 이상인 바코드 숫자 추출 결과 반환
        return jsonify({'result': 1, 'barcodes': high_conf_barcodes}), 200



#     객체 탐지 결과 처리
def process_detected_objects(results, img):
    high_conf_barcodes = []
    for i, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
        # 좌표와 정확도 점수를 파이썬 정수와 부동소수점으로 명시적으로 변환
        x1, y1, x2, y2 = map(int, map(lambda x: x.item(), xyxy))
        conf = conf.item()

        if conf >= 0.6:  # 정확도가 0.6 이상인 경우만 처리
            try:
                crop_img = img.crop((x1, y1, x2, y2))
                barcodes = decode(crop_img)
                barcode_numbers = [barcode.data.decode('utf-8') for barcode in barcodes]
                high_conf_barcodes.extend(barcode_numbers)
            except Exception as e:
                app.logger.error(f'Error processing detected objects: {e}')
    return high_conf_barcodes

if __name__ == '__main__':
    app.run(debug=True, port=5001)

'''
    high_conf_barcodes = []
    for i, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
        # 좌표와 정확도 점수를 파이썬 정수와 부동소수점으로 명시적으로 변환
        x1, y1, x2, y2 = map(int, map(lambda x: x.item(), xyxy))
        conf = conf.item()

        if conf >= 0.6:  # 정확도가 0.6 이상인 경우만 처리
            crop_img = img.crop((x1, y1, x2, y2))
            barcodes = decode(crop_img)
            barcode_numbers = [barcode.data.decode('utf-8') for barcode in barcodes]
            high_conf_barcodes.extend(barcode_numbers)

    if not high_conf_barcodes:  # 정확도가 0.6 이상인 바코드를 인식하지 못한 경우
        return jsonify({'result': 0, 'message': 'No high-confidence barcodes detected'}), 200
    else:
        # 정확도가 0.6 이상인 바코드 숫자 추출 결과 반환
        return jsonify({'result': 1, 'barcodes': high_conf_barcodes}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
'''
    