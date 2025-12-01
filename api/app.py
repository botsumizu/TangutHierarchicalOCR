import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import time
from werkzeug.utils import secure_filename
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 最大文件大小

# 创建上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class THOCRSystem:
    """THOCR 西夏文识别系统"""

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
                'U+178A8',
                'U+17E15',
                'U+17E6D',
                'U+17F03',
                'U+17F86',
                'U+180D9',
                'U+181E5',
                'U+181F0',
                'U+1821A',
                'U+1824B',
                'U+1825D',
                'U+1825F',
                'U+1828F',
                'U+182DD',
                'U+18322',
                'U+18350',
                'U+1835D',
                'U+185E4',
                'U+185EA',
                'U+1866C',
            ],
            'V': [
                'U+17006',
                'U+17016',
                'U+1701F',
                'U+17100',
                'U+17108',
                'U+17109',
                'U+17116',
                'U+1742E',
                'U+17431',
                'U+17460',
                'U+17467',
                'U+1748C',
                'U+174C1',
                'U+174EB',
                'U+17552',
                'U+17564',
                'U+17572',
                'U+17683',
                'U+17684',
                'U+1768B',
                'U+1768C',
                'U+1768F',
                'U+176DC',
                'U+1771A',
                'U+17C86',
                'U+17CBA',
                'U+17D33',
                'U+17D35',
                'U+17D3F',
                'U+17D40',
                'U+17D49',
                'U+17D4A',
                'U+17D54',
                'U+17D55',
                'U+17D65',
                'U+17DA0',
                'U+17DA7',
                'U+17DB2',
                'U+17DB4',
                'U+17DB7',
                'U+17DB9',
                'U+18191',
                'U+1848A',
                'U+18497',
                'U+18527',
                'U+18797',
                'U+187BC',
                'U+187C0',
                'U+187C5',
                'U+187E0',
            ],
            'H': [
                'U+17030',
                'U+1712C',
                'U+1726D',
                'U+1732F',
                'U+17335',
                'U+17339',
                'U+1733E',
                'U+1734F',
                'U+17376',
                'U+17381',
                'U+173AC',
                'U+1757C',
                'U+17591',
                'U+1760B',
                'U+1760C',
                'U+17619',
                'U+1764B',
                'U+1764F',
                'U+178B3',
                'U+178CA',
                'U+17B7D',
                'U+17BE3',
                'U+17D7F',
                'U+17D8E',
                'U+17DDD',
                'U+17DF7',
                'U+17E16',
                'U+17E5D',
                'U+17E9B',
                'U+17F24',
                'U+17FDD',
                'U+1804C',
                'U+180BB',
                'U+180BE',
                'U+1812F',
                'U+18133',
                'U+18159',
                'U+18167',
                'U+181AD',
                'U+181BE',
                'U+1826B',
                'U+1839B',
                'U+1845B',
                'U+18474',
                'U+18517',
                'U+185FD',
                'U+18698',
                'U+186BC',
                'U+186E3',
                'U+187EE',
            ],
            'E': [
                'U+1711D',
                'U+171C5',
                'U+171CC',
                'U+17407',
                'U+1740A',
                'U+1741E',
                'U+17422',
                'U+17424',
                'U+17426',
                'U+17AF1',
                'U+17AF2',
                'U+17AF6',
                'U+17AF8',
                'U+17AF9',
                'U+17AFA',
                'U+17AFB',
                'U+17AFC',
                'U+17AFD',
                'U+17AFE',
                'U+17B01',
                'U+17B02',
                'U+17B03',
                'U+17B04',
                'U+17B05',
                'U+17B07',
                'U+17B08',
                'U+17B0A',
                'U+17B36',
                'U+17B64',
                'U+17B66',
                'U+17B9A',
                'U+17B9C',
                'U+17B9E',
                'U+17B9F',
                'U+17BA0',
                'U+17BA8',
                'U+17BA9',
                'U+17BB2',
                'U+17BB3',
                'U+17BB9',
                'U+17BC2',
                'U+17DE2',
                'U+17E1F',
                'U+1817E',
                'U+182C6',
                'U+1860B',
                'U+1860C',
                'U+1860F',
                'U+186C3',
                'U+1871B',
            ],
        }

        # 初始化模型
        self.structure_classifier = self.load_structure_classifier()
        self.recognizers = {
            'S': self.load_recognizer('S'),
            'V': self.load_recognizer('V'),
            'H': self.load_recognizer('H'),
            'E': self.load_recognizer('E')
        }

        # 统计字符总数
        self.total_chars = sum(len(chars) for chars in self.recognition_classes.values())
        logger.info(f"THOCR系统初始化完成，共支持 {self.total_chars} 个字符")

    def load_structure_classifier(self):
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            structure_model_path = os.path.join(BASE_DIR, "best_tangut_structure_classifier_balanced.pth")
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 4)
            model.load_state_dict(torch.load(structure_model_path,
                                             map_location=self.device))
            model.eval()
            logger.info("结构分类器加载成功")
            return model.to(self.device)
        except Exception as e:
            logger.error(f"结构分类器加载失败: {e}")
            return None

    def load_recognizer(self, structure_type):
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            recognizer_model_path = os.path.join(BASE_DIR, f'tangut_recognizer_{structure_type}_v2.pth')
            model = models.resnet18()

            # 根据结构类型设置分类数
            num_classes = len(self.recognition_classes[structure_type])
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            model.load_state_dict(torch.load(recognizer_model_path,
                                             map_location=self.device))
            model.eval()
            logger.info(f"{structure_type}类型识别器加载成功 ({num_classes}个字符)")
            return model.to(self.device)
        except Exception as e:
            logger.error(f"{structure_type}类型识别器加载失败: {e}")
            return None

    def predict(self, image_path):
        """识别单个图像"""
        if self.structure_classifier is None:
            return {"error": "结构分类器加载失败"}

        for structure_type, recognizer in self.recognizers.items():
            if recognizer is None:
                return {"error": f"{structure_type}类型识别器加载失败"}

        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 结构分类
            with torch.no_grad():
                structure_output = self.structure_classifier(input_tensor)
                structure_pred = torch.argmax(structure_output, 1).item()
                structure_label = self.structure_classes[structure_pred]

                # 映射到识别器类型
                structure_map = {'enclosed': 'E', 'horizontal': 'H',
                                 'single': 'S', 'vertical': 'V'}
                structure_code = structure_map[structure_label]

            # 字符识别
            recognizer = self.recognizers[structure_code]
            with torch.no_grad():
                char_output = recognizer(input_tensor)
                char_pred = torch.argmax(char_output, 1).item()
                char_label = self.recognition_classes[structure_code][char_pred]

            # 计算置信度
            structure_conf = torch.softmax(structure_output, 1)[0][structure_pred].item()
            char_conf = torch.softmax(char_output, 1)[0][char_pred].item()

            return {
                'success': True,
                'structure': structure_label,
                'structure_code': structure_code,
                'character': char_label,
                'unicode': char_label,
                'confidence': {
                    'structure': structure_conf,
                    'character': char_conf,
                    'overall': (structure_conf + char_conf) / 2
                },
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"识别失败: {e}")
            return {"success": False, "error": f"识别失败: {str(e)}"}

    def get_system_info(self):
        """获取系统信息"""
        return {
            'device': str(self.device),
            'total_characters': self.total_chars,
            'characters_by_type': {
                'S': len(self.recognition_classes['S']),
                'V': len(self.recognition_classes['V']),
                'H': len(self.recognition_classes['H']),
                'E': len(self.recognition_classes['E'])
            },
            'structure_classes': self.structure_classes,
            'models_loaded': {
                'structure_classifier': self.structure_classifier is not None,
                'S_recognizer': self.recognizers['S'] is not None,
                'V_recognizer': self.recognizers['V'] is not None,
                'H_recognizer': self.recognizers['H'] is not None,
                'E_recognizer': self.recognizers['E'] is not None
            }
        }


# 全局THOCR实例
thocr = None


def init_thocr():
    """初始化THOCR系统"""
    global thocr
    try:
        logger.info("正在初始化THOCR系统...")
        thocr = THOCRSystem()
        logger.info("THOCR系统初始化完成")
        return True
    except Exception as e:
        logger.error(f"THOCR系统初始化失败: {e}")
        return False


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """首页"""
    return jsonify({
        'name': 'THOCR 西夏文识别API',
        'version': '1.0.0',
        'description': '基于深度学习的西夏文分层识别系统',
        'endpoints': {
            '/': 'API信息',
            '/api/info': '系统信息',
            '/api/recognize': '单字符识别 (POST)',
            '/api/batch_recognize': '批量识别 (POST)',
            '/api/health': '健康检查'
        }
    })


@app.route('/api/info', methods=['GET'])
def get_info():
    """获取系统信息"""
    if thocr is None:
        return jsonify({'error': 'THOCR系统未初始化'}), 503
    
    info = thocr.get_system_info()
    return jsonify(info)


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """单字符识别接口"""
    if thocr is None:
        return jsonify({'success': False, 'error': 'THOCR系统未初始化'}), 503
    
    # 检查文件上传
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有上传文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': '不支持的文件类型',
            'allowed_types': list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        # 保存上传的文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        logger.info(f"开始识别文件: {filename}")
        
        # 进行识别
        result = thocr.predict(filepath)
        
        # 添加文件信息
        if result.get('success', False):
            result['filename'] = filename
            result['file_id'] = unique_filename
        
        # 清理临时文件（可选）
        # os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"识别处理失败: {e}")
        return jsonify({
            'success': False,
            'error': f'处理失败: {str(e)}'
        }), 500


@app.route('/api/batch_recognize', methods=['POST'])
def batch_recognize():
    """批量识别接口"""
    if thocr is None:
        return jsonify({'success': False, 'error': 'THOCR系统未初始化'}), 503
    
    # 检查文件上传
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': '没有上传文件'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'success': False, 'error': '没有选择文件'}), 400
    
    if len(files) > 10:  # 限制批量处理数量
        return jsonify({'success': False, 'error': '一次最多处理10个文件'}), 400
    
    results = []
    processed_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # 保存文件
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                processed_files.append(filepath)
                
                # 识别
                result = thocr.predict(filepath)
                result['filename'] = filename
                result['file_id'] = unique_filename
                results.append(result)
                
            except Exception as e:
                logger.error(f"文件 {file.filename} 处理失败: {e}")
                results.append({
                    'success': False,
                    'filename': file.filename,
                    'error': f'处理失败: {str(e)}'
                })
    
    # 清理临时文件
    for filepath in processed_files:
        try:
            os.remove(filepath)
        except:
            pass
    
    return jsonify({
        'success': True,
        'total': len(results),
        'results': results,
        'timestamp': time.time()
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    if thocr is None:
        return jsonify({'status': 'unhealthy', 'error': 'THOCR未初始化'}), 503
    
    try:
        # 简单测试模型是否可用
        info = thocr.get_system_info()
        models_loaded = info['models_loaded']
        
        all_loaded = all(models_loaded.values())
        
        return jsonify({
            'status': 'healthy' if all_loaded else 'degraded',
            'models_loaded': models_loaded,
            'device': str(thocr.device),
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': '服务器内部错误'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'error': '文件太大'}), 413


if __name__ == '__main__':
    # 初始化THOCR系统
    if init_thocr():
        # 启动Flask应用
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
    logger.error("THOCR系统初始化失败，无法启动API服务")