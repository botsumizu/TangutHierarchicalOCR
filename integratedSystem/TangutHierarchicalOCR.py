import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json


class THOCRSystem:

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
        self.structure_classifier = self.load_structure_classifier()
        self.recognizers = {
            'S': self.load_recognizer('S'),
            'V': self.load_recognizer('V'),
            'H': self.load_recognizer('H'),
            'E': self.load_recognizer('E')
        }

    def load_structure_classifier(self):
        try:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 4)
            model.load_state_dict(torch.load('best_tangut_structure_classifier_balanced.pth',
                                             map_location=self.device))
            model.eval()

            return model.to(self.device)
        except Exception as e:
            print(f"结构分类器加载失败: {e}")
            return None

    def load_recognizer(self, structure_type):
        try:
            model = models.resnet18()
            num_classes = len(self.recognition_classes[structure_type])
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            model.load_state_dict(torch.load(f'tangut_recognizer_{structure_type}_v2.pth',
                                             map_location=self.device))
            model.eval()
            print(f"{structure_type}类型识别器加载成功 ({num_classes}类)")
            return model.to(self.device)
        except Exception as e:
            print(f"{structure_type}类型识别器加载失败: {e}")
            return None

    def predict(self, image_path):
        if self.structure_classifier is None:
            return {"error": "结构分类器加载失败"}

        for structure_type, recognizer in self.recognizers.items():
            if recognizer is None:
                return {"error": f"{structure_type}类型识别器加载失败"}

        try:

            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)


            with torch.no_grad():
                structure_output = self.structure_classifier(input_tensor)
                structure_pred = torch.argmax(structure_output, 1).item()
                structure_label = self.structure_classes[structure_pred]


                structure_map = {'enclosed': 'E', 'horizontal': 'H',
                                 'single': 'S', 'vertical': 'V'}
                structure_code = structure_map[structure_label]


            recognizer = self.recognizers[structure_code]
            with torch.no_grad():
                char_output = recognizer(input_tensor)
                char_pred = torch.argmax(char_output, 1).item()
                char_label = self.recognition_classes[structure_code][char_pred]

            return {
                'structure': structure_label,
                'character': char_label,
                'confidence': {
                    'structure': torch.softmax(structure_output, 1)[0][structure_pred].item(),
                    'character': torch.softmax(char_output, 1)[0][char_pred].item()
                }
            }
        except Exception as e:
            return {"error": f"识别过程中出错: {e}"}



if __name__ == "__main__":
    print("初始化THOCR系统...")
    thocr = THOCRSystem()

    print("\n系统初始化完成！")
    print("可识别的字符总数:", sum(len(chars) for chars in thocr.recognition_classes.values()))


    test_image = 'test_char.png'
    if os.path.exists(test_image):
        result = thocr.predict(test_image)
        if 'error' in result:
            print(f"识别失败: {result['error']}")
        else:
            print(f"\n识别结果:")
            print(f"  结构: {result['structure']}")
            print(f"  字符: {result['character']}")
            print(f"  置信度: 结构{result['confidence']['structure']:.2%}, "
                  f"字符{result['confidence']['character']:.2%}")
    else:
        print(f"\n测试图片 {test_image} 不存在，请提供有效的测试图片路径")