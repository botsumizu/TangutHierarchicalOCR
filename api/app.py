import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from flask import Flask, request, jsonify
import io
import base64
from datetime import datetime

app = Flask(__name__)

# THOCR系统类保持不变
class THOCRSystem:
    """THOCR系统"""
    
    def __init__(self, model_dir='.'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        
        self.transform = transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.structure_classes = ['enclosed', 'horizontal', 'single', 'vertical']
        self.recognition_classes = {
            'S': [
                'U+178A8', 'U+17E15', 'U+17E6D', 'U+17F03', 'U+17F86',
                'U+180D9', 'U+181E5', 'U+181F0', 'U+1821A', 'U+1824B',
                'U+1825D', 'U+1825F', 'U+1828F', 'U+182DD', 'U+18322',
                'U+18350', 'U+1835D', 'U+185E4', 'U+185EA', 'U+1866C',
            ],
            'V': [
                'U+17006', 'U+17016', 'U+1701F', 'U+17100', 'U+17108',
                'U+17109', 'U+17116', 'U+1742E', 'U+17431', 'U+17460',
                'U+17467', 'U+1748C', 'U+174C1', 'U+174EB', 'U+17552',
                'U+17564', 'U+17572', 'U+17683', 'U+17684', 'U+1768B',
                'U+1768C', 'U+1768F', 'U+176DC', 'U+1771A', 'U+17C86',
                'U+17CBA', 'U+17D33', 'U+17D35', 'U+17D3F', 'U+17D40',
                'U+17D49', 'U+17D4A', 'U+17D54', 'U+17D55', 'U+17D65',
                'U+17DA0', 'U+17DA7', 'U+17DB2', 'U+17DB4', 'U+17DB7',
                'U+17DB9', 'U+18191', 'U+1848A', 'U+18497', 'U+18527',
                'U+18797', 'U+187BC', 'U+187C0', 'U+187C5', 'U+187E0',
            ],
            'H': [
                'U+17030', 'U+1712C', 'U+1726D', 'U+1732F', 'U+17335',
                'U+17339', 'U+1733E', 'U+1734F', 'U+17376', 'U+17381',
                'U+173AC', 'U+1757C', 'U+17591', 'U+1760B', 'U+1760C',
                'U+17619', 'U+1764B', 'U+1764F', 'U+178B3', 'U+178CA',
                'U+17B7D', 'U+17BE3', 'U+17D7F', 'U+17D8E', 'U+17DDD',
                'U+17DF7', 'U+17E16', 'U+17E5D', 'U+17E9B', 'U+17F24',
                'U+17FDD', 'U+1804C', 'U+180BB', 'U+180BE', 'U+1812F',
                'U+18133', 'U+18159', 'U+18167', 'U+181AD', 'U+181BE',
                'U+1826B', 'U+1839B', 'U+1845B', 'U+18474', 'U+18517',
                'U+185FD', 'U+18698', 'U+186BC', 'U+186E3', 'U+187EE',
            ],
            'E': [
                'U+1711D', 'U+171C5', 'U+171CC', 'U+17407', 'U+1740A',
                'U+1741E', 'U+17422', 'U+17424', 'U+17426', 'U+17AF1',
                'U+17AF2', 'U+17AF6', 'U+17AF8', 'U+17AF9', 'U+17AFA',
                'U+17AFB', 'U+17AFC', 'U+17AFD', 'U+17AFE', 'U+17B01',
                'U+17B02', 'U+17B03', 'U+17B04', 'U+17B05', 'U+17B07',
                'U+17B08', 'U+17B0A', 'U+17B36', 'U+17B64', 'U+17B66',
                'U+17B9A', 'U+17B9C', 'U+17B9E', 'U+17B9F', 'U+17BA0',
                'U+17BA8', 'U+17BA9', 'U+17BB2', 'U+17BB3', 'U+17BB9',
                'U+17BC2', 'U+17DE2', 'U+17E1F', 'U+1817E', 'U+182C6',
                'U+1860B', 'U+1860C', 'U+1860F', 'U+186C3', 'U+1871B',
            ],
        }
        
        self.structure_classifier = self.load_structure_classifier()
        self.recognizers = {
            'S': self.load_recognizer('S'),
            'V': self.load_recognizer('V'),
            'H': self.load_recognizer('H'),
            'E': self.load_recognizer('E')
        }
    
    def load_structure_classifier(self):
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            structure_model_path = os.path.join(BASE_DIR, "best_tangut_structure_classifier_balanced.pth")
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 4)
            model.load_state_dict(torch.load(structure_model_path, 
                                            map_location=self.device))
            model.eval()
            print("结构分类器加载成功")
            return model.to(self.device)
        except Exception as e:
            print(f"结构分类器加载失败: {e}")
            return None
    
    def load_recognizer(self, structure_type):
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            recognizer_model_path = os.path.join(BASE_DIR, f'tangut_recognizer_{structure_type}_v2.pth')
            model = models.resnet18()
            num_classes = len(self.recognition_classes[structure_type])
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(recognizer_model_path, 
                                            map_location=self.device))
            model.eval()
            print(f"{structure_type}-识别器加载成功 ({num_classes}类)")
            return model.to(self.device)
        except Exception as e:
            print(f"{structure_type}识别器加载失败: {e}")
            return None
    
    def predict_image(self, image):
        """预测单张图片"""
        if self.structure_classifier is None:
            return {"error": "结构分类器加载失败"}
        
        for structure_type, recognizer in self.recognizers.items():
            if recognizer is None:
                return {"error": f"{structure_type}识别器加载失败"}
        
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                structure_output = self.structure_classifier(input_tensor)
                structure_pred = torch.argmax(structure_output, 1).item()
                structure_label = self.structure_classes[structure_pred]
                
                structure_map = {'enclosed': 'E', 'horizontal': 'H',
                               'single': 'S', 'vertical': 'V'}
                structure_code = structure_map[structure_label]
                
                recognizer = self.recognizers[structure_code]
                char_output = recognizer(input_tensor)
                char_pred = torch.argmax(char_output, 1).item()
                char_label = self.recognition_classes[structure_code][char_pred]
            
            return {
                'status': 'success',
                'structure': structure_label,
                'structure_code': structure_code,
                'character': char_label,
                'confidence': {
                    'structure': float(torch.softmax(structure_output, 1)[0][structure_pred].item()),
                    'character': float(torch.softmax(char_output, 1)[0][char_pred].item())
                },
                'character_count': sum(len(chars) for chars in self.recognition_classes.values())
            }
        except Exception as e:
            return {"status": "error", "error": f"识别失败: {str(e)}"}

# ========== CORS 中间件 ==========
@app.after_request
def add_cors_headers(response):
    """为所有响应添加CORS头"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Max-Age'] = '86400'  # 24小时
    return response

# 处理OPTIONS预检请求的装饰器
def handle_options_request(func):
    def wrapper(*args, **kwargs):
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
            return response
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# ========== API 路由 ==========
@app.route('/')
def index():
    """API首页"""
    character_count = sum(len(chars) for chars in thocr.recognition_classes.values())
    return jsonify({
        'status': 'success',
        'service': 'THOCR西夏文识别API',
        'description': '基于深度学习的西夏文分层识别系统',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API信息',
            '/recognize': 'POST - 识别西夏文字符',
            '/health': 'GET - 服务健康检查'
        },
        'total_characters': character_count,
        'structure_types': {
            'S': '单一结构',
            'V': '垂直结构', 
            'H': '水平结构',
            'E': '包围结构'
        },
        'cors_enabled': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/recognize', methods=['POST', 'OPTIONS'])
@handle_options_request
def recognize():
    """识别西夏文字符"""
    try:
        # 检查请求中是否包含图片
        if 'image' not in request.files and 'image_base64' not in request.form:
            return jsonify({
                'status': 'error',
                'error': '请提供图片文件或Base64编码的图片',
                'supported_methods': ['file upload (multipart/form-data)', 'base64 (application/x-www-form-urlencoded)']
            }), 400
        
        # 处理文件上传
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'status': 'error', 'error': '未选择文件'}), 400
            
            # 验证文件类型
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                return jsonify({
                    'status': 'error', 
                    'error': f'不支持的图片格式。支持的格式: {", ".join(allowed_extensions)}'
                }), 400
            
            try:
                image = Image.open(file.stream).convert('RGB')
            except Exception as e:
                return jsonify({'status': 'error', 'error': f'无法打开图片文件: {str(e)}'}), 400
        
        # 处理Base64编码的图片
        elif 'image_base64' in request.form:
            base64_str = request.form['image_base64']
            try:
                # 去除data:image/jpeg;base64,前缀
                if ',' in base64_str:
                    base64_str = base64_str.split(',')[1]
                
                image_data = base64.b64decode(base64_str)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                return jsonify({'status': 'error', 'error': f'Base64解码失败: {str(e)}'}), 400
        
        # 验证图片尺寸
        if image.width < 50 or image.height < 50:
            return jsonify({'status': 'error', 'error': '图片尺寸过小，建议至少50x50像素'}), 400
        
        # 进行识别
        result = thocr.predict_image(image)
        
        # 添加时间戳
        if isinstance(result, dict) and 'status' in result and result['status'] == 'success':
            result['timestamp'] = datetime.now().isoformat()
            result['processing_time'] = '实时'
        
        return jsonify(result)
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'处理请求时出错: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET', 'OPTIONS'])
@handle_options_request
def health_check():
    """健康检查端点"""
    try:
        # 检查所有模型是否已加载
        models_loaded = {
            'structure_classifier': thocr.structure_classifier is not None,
            'S_recognizer': thocr.recognizers['S'] is not None,
            'V_recognizer': thocr.recognizers['V'] is not None,
            'H_recognizer': thocr.recognizers['H'] is not None,
            'E_recognizer': thocr.recognizers['E'] is not None
        }
        
        all_loaded = all(models_loaded.values())
        
        return jsonify({
            'status': 'healthy' if all_loaded else 'unhealthy',
            'models_loaded': models_loaded,
            'device': str(thocr.device),
            'total_characters': sum(len(chars) for chars in thocr.recognition_classes.values()),
            'cors_enabled': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """处理404错误"""
    return jsonify({
        'status': 'error',
        'error': '请求的API端点不存在',
        'available_endpoints': ['/', '/recognize', '/health'],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    return jsonify({
        'status': 'error',
        'error': '服务器内部错误',
        'timestamp': datetime.now().isoformat()
    }), 500

# ========== 主程序 ==========
if __name__ == '__main__':
    # 初始化THOCR系统
    print("正在初始化THOCR系统...")
    try:
        thocr = THOCRSystem()
        character_count = sum(len(chars) for chars in thocr.recognition_classes.values())
        print(f"THOCR系统初始化成功")
        print(f"支持的字符总数: {character_count}")
        print(f"运行设备: {thocr.device}")
        print(f"CORS已启用，支持跨域请求")
    except Exception as e:
        print(f"THOCR系统初始化失败: {e}")
        thocr = None
    
    print("\n" + "="*60)
    print("THOCR API 服务已启动")
    print("="*60)
    print("API端点:")
    print("  GET  /          - 获取API信息")
    print("  POST /recognize - 识别西夏文字符")
    print("  GET  /health    - 健康检查")
    print("\nCORS配置:")
    print("  Access-Control-Allow-Origin: *")
    print("  支持跨域请求，前端可正常访问")
    print("\n启动信息:")
    print(f"  服务地址: http://0.0.0.0:5000")
    print(f"  本地访问: http://localhost:5000")
    print(f"  网络访问: http://[你的IP地址]:5000")
    print("="*60)
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)