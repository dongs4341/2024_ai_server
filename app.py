from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import os
import mimetypes
import json
import torchvision.models as models
import torchvision.transforms as transforms

app = Flask(__name__)

# 모델 로드
model_path = 'C:/Users/23/Desktop/웹개발양성과정_자료/AI모델/ai_server_pillDetect/평가용 데이터셋/pill_data/proj_pill/pill_resnet152_dataclass0_aug0.pt'
try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model = models.resnet152()
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
except Exception as e:
    app.logger.error(f'모델 로딩 오류: {e}')
    model = None

# 알약 데이터 매핑 로드
json_file_path = 'C:/Users/23/Desktop/웹개발양성과정_자료/AI모델/ai_server_pillDetect/평가용 데이터셋/pill_data/pill_data_croped/pill_label_path_sharp_score.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    pill_data = json.load(f)

# 이미지별 JSON 파일에서 알약 세부정보 로드
def load_pill_details(drug_id):
    pill_detail_path = f'C:/Users/23/Desktop/웹개발양성과정_자료/AI모델/ai_server_pillDetect/평가용 데이터셋/pill_data/pill_data_croped/{drug_id}/{drug_id}_0_0_0_0_60_000_200.json'
    try:
        with open(pill_detail_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        app.logger.error(f'알약 세부정보 로드 오류: {e}')
        return None

def get_pill_info(class_id):
    for pill_info in pill_data['pill_label_path_sharp_score']:
        if pill_info[0] == class_id:
            return pill_info[1]  # 알약 코드 반환
    return "Unknown"  # 클래스 ID가 없을 경우

@app.route('/')
def index():
    return "알약 탐지 서버가 실행 중입니다!"

@app.route('/detect_pill', methods=['POST'])
def detect_pill():
    if 'file' not in request.files:
        return jsonify({'error': "파일 부분이 없습니다"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': "선택된 파일이 없습니다"}), 400

    # MIME 타입 검사
    mime_type = mimetypes.guess_type(file.filename)[0]
    if not mime_type or not mime_type.startswith('image'):
        return jsonify({'error': "업로드된 파일이 이미지가 아닙니다"}), 400

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'이미지 파일을 처리하는 중 오류 발생: {e}'}), 400

    if model is None:
        return jsonify({'error': '모델 로딩 실패, 알약 탐지를 수행할 수 없습니다.'}), 500

    try:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

        # 알약 코드 및 세부정보 로드
        pill_code = get_pill_info(predicted_class)
        pill_details = load_pill_details(pill_code)

        dl_name = pill_details['images'][0]['dl_name'] if pill_details else "Unknown"
        di_edi_code = pill_details['images'][0]['di_edi_code'] if pill_details else "Unknown"

        return jsonify({
            'result': 1,
            'predicted_class': predicted_class,
            'dl_mapping_code': pill_code,
            'dl_name': dl_name,
            'di_edi_code': di_edi_code
        }), 200
    except Exception as e:
        return jsonify({'error': f'알약 탐지 중 오류 발생: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)