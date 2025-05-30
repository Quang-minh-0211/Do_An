# app.py - Flask API Server
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Cho phép frontend gọi API

# Load mô hình và các thành phần khi khởi động server
print("🔄 Đang load mô hình...")

try:
    # Load model
    model = load_model(r'D:\BigData And DataMining\Đồ án\model\disease_predict_ann.h5')
    print("✅ Đã load mô hình ANN!")
    
    # Load scaler
    scaler = joblib.load(r'D:\BigData And DataMining\Đồ án\model\scaler.pkl')
    print("✅ Đã load scaler!")
    
    # Load label encoder
    label_encoder = joblib.load(r'D:\BigData And DataMining\Đồ án\model\label_encoder.pkl')
    print("✅ Đã load label encoder!")
    
    # Load feature names
    with open(r'D:\BigData And DataMining\Đồ án\model\feature_names.json', 'r') as f:
        feature_names = json.load(f)
    print("✅ Đã load feature names!")
    
    # Load disease classes
    with open(r'D:\BigData And DataMining\Đồ án\model\disease_classes.json', 'r') as f:
        disease_classes = json.load(f)
    print("✅ Đã load disease classes!")
    
    print(f"🎯 Mô hình có {len(feature_names)} features và {len(disease_classes)} loại bệnh")
    
except Exception as e:
    print(f"❌ Lỗi khi load mô hình: {e}")
    exit(1)

def preprocess_input(data):
    """Xử lý dữ liệu đầu vào từ frontend"""
    try:
        # Tạo array theo đúng thứ tự features
        input_array = []
        
        # Mapping từ form field names sang feature names
        field_mapping = {
            'age': 'Age',
            'gender': 'Gender', 
            'temperature': 'Temperature (C)',
            'humidity': 'Humidity',
            'windspeed': 'Wind Speed (km/h)'
        }
        
        # Thêm thông tin cơ bản
        for field, feature in field_mapping.items():
            input_array.append(float(data.get(field, 0)))
        
        # Thêm các triệu chứng (binary: 0 hoặc 1)
        symptom_features = [
            'nausea', 'joint_pain', 'abdominal_pain', 'high_fever', 'chills', 'fatigue',
            'runny_nose', 'pain_behind_the_eyes', 'dizziness', 'headache', 'chest_pain',
            'vomiting', 'cough', 'shivering', 'asthma_history', 'high_cholesterol',
            'diabetes', 'obesity', 'hiv_aids', 'nasal_polyps', 'asthma', 'high_blood_pressure',
            'severe_headache', 'weakness', 'trouble_seeing', 'fever', 'body_aches',
            'sore_throat', 'sneezing', 'diarrhea', 'rapid_breathing', 'rapid_heart_rate',
            'pain_behind_eyes', 'swollen_glands', 'rashes', 'sinus_headache', 'facial_pain',
            'shortness_of_breath', 'reduced_smell_and_taste', 'skin_irritation', 'itchiness',
            'throbbing_headache', 'confusion', 'back_pain', 'knee_ache'
        ]
        
        for symptom in symptom_features:
            input_array.append(int(data.get(symptom, 0)))
        
        return np.array(input_array).reshape(1, -1)
        
    except Exception as e:
        raise Exception(f"Lỗi xử lý dữ liệu: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Kiểm tra API có hoạt động không"""
    return jsonify({
        'status': 'healthy',
        'message': 'Disease Prediction API is running!',
        'model_info': {
            'features': len(feature_names),
            'classes': len(disease_classes)
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """API endpoint chính để dự đoán bệnh"""
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Không có dữ liệu đầu vào',
                'success': False
            }), 400
        
        # Xử lý dữ liệu đầu vào
        input_data = preprocess_input(data)
        print(f"📊 Input shape: {input_data.shape}")
        
        # Chuẩn hóa dữ liệu với scaler đã train
        input_scaled = scaler.transform(input_data)
        print(f"📊 Scaled input shape: {input_scaled.shape}")
        
        # Dự đoán với mô hình
        predictions = model.predict(input_scaled)
        print(f"📊 Raw predictions shape: {predictions.shape}")
        
        # Lấy class có xác suất cao nhất
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Decode về tên bệnh
        predicted_disease = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Lấy top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            disease_name = label_encoder.inverse_transform([idx])[0]
            prob = float(predictions[0][idx])
            top_3_predictions.append({
                'disease': disease_name,
                'probability': round(prob * 100, 2)
            })
        
        # Tạo khuyến nghị dựa trên kết quả
        recommendation = generate_recommendation(predicted_disease, confidence, data)
        
        # Kết quả trả về
        result = {
            'success': True,
            'prediction': {
                'disease': predicted_disease,
                'confidence': round(confidence * 100, 2),
                'confidence_level': get_confidence_level(confidence)
            },
            'top_predictions': top_3_predictions,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Dự đoán thành công: {predicted_disease} ({confidence:.2%})")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Lỗi dự đoán: {e}")
        return jsonify({
            'error': f'Lỗi khi dự đoán: {str(e)}',
            'success': False
        }), 500

def get_confidence_level(confidence):
    """Phân loại mức độ tin cậy"""
    if confidence >= 0.8:
        return 'Rất cao'
    elif confidence >= 0.6:
        return 'Cao'
    elif confidence >= 0.4:
        return 'Trung bình'
    else:
        return 'Thấp'

def generate_recommendation(disease, confidence, symptoms):
    """Tạo khuyến nghị dựa trên kết quả dự đoán"""
    base_recommendations = []
    
    # Khuyến nghị dựa trên độ tin cậy
    if confidence >= 0.7:
        base_recommendations.append("Kết quả dự đoán có độ tin cậy cao")
    else:
        base_recommendations.append("Nên theo dõi thêm các triệu chứng")
    
    # Khuyến nghị dựa trên loại bệnh
    emergency_keywords = ['severe', 'acute', 'emergency', 'critical']
    if any(keyword in disease.lower() for keyword in emergency_keywords):
        base_recommendations.extend([
            "⚠️ Cần đi khám bác sĩ ngay lập tức",
            "Không tự điều trị tại nhà"
        ])
    else:
        base_recommendations.extend([
            "Nghỉ ngơi đầy đủ và uống nhiều nước",
            "Theo dõi triệu chứng trong 24-48h",
            "Nếu triệu chứng nặng hơn, hãy đi khám bác sĩ"
        ])
    
    # Khuyến nghị dựa trên triệu chứng nghiêm trọng
    serious_symptoms = ['high_fever', 'severe_headache', 'shortness_of_breath', 'chest_pain']
    if any(symptoms.get(symptom, 0) == 1 for symptom in serious_symptoms):
        base_recommendations.insert(0, "⚠️ Có triệu chứng nghiêm trọng - nên đi khám sớm")
    
    return base_recommendations

@app.route('/api/info', methods=['GET'])
def get_model_info():
    """Lấy thông tin về mô hình"""
    return jsonify({
        'features': feature_names,
        'diseases': disease_classes,
        'total_features': len(feature_names),
        'total_diseases': len(disease_classes)
    })

if __name__ == '__main__':
    print("\n🚀 Khởi động Disease Prediction API Server...")
    print("📍 API endpoints:")
    print("  - GET  /api/health  - Kiểm tra trạng thái")
    print("  - POST /api/predict - Dự đoán bệnh") 
    print("  - GET  /api/info    - Thông tin mô hình")
    print("\n🌐 Server đang chạy tại: http://localhost:5000")
    print("🔗 Test API: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)