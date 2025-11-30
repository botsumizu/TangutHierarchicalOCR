import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns



def setup_chinese_font():
    """设置中文字体支持"""
    try:

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 中文字体设置成功")
    except:
        print("⚠️ 中文字体设置失败，使用默认字体")


class THOCRSystem:
    """完整的THOCR西夏文识别系统"""

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

        # 加载模型
        self.structure_classifier = self.load_structure_classifier()
        self.recognizers = {
            'S': self.load_recognizer('S'),
            'V': self.load_recognizer('V'),
            'H': self.load_recognizer('H'),
            'E': self.load_recognizer('E')
        }

    def load_structure_classifier(self):
        """加载结构分类器"""
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load('best_tangut_structure_classifier_balanced.pth',
                                         map_location=self.device))
        model.eval()
        return model.to(self.device)

    def load_recognizer(self, structure_type):
        """加载文字识别器"""
        model = models.resnet18()
        num_classes = len(self.recognition_classes[structure_type])
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(f'tangut_recognizer_{structure_type}_v2.pth',
                                         map_location=self.device))
        model.eval()
        return model.to(self.device)

    def predict(self, image_path):
        """完整识别流程"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 结构分类
        with torch.no_grad():
            structure_output = self.structure_classifier(input_tensor)
            structure_pred = torch.argmax(structure_output, 1).item()
            structure_label = self.structure_classes[structure_pred]
            structure_map = {'enclosed': 'E', 'horizontal': 'H',
                             'single': 'S', 'vertical': 'V'}
            structure_code = structure_map[structure_label]

        # 文字识别
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


def test_integrated_system():
    """测试集成系统并生成论文用图表"""

    print("开始集成系统测试...")
    thocr = THOCRSystem()

    test_dir = 'testDatabase'
    results = []
    confusion_data = []

    # 遍历测试数据集
    for filename in os.listdir(test_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(test_dir, filename)

            # 从文件名解析真实标签
            basename = os.path.splitext(filename)[0]
            parts = basename.split('+')
            if len(parts) >= 2:
                true_char = 'U+' + parts[1][:5]
                true_structure = parts[1][-1]
                structure_map = {'S': 'single', 'V': 'vertical', 'H': 'horizontal', 'E': 'enclosed'}
                true_structure_label = structure_map.get(true_structure, 'unknown')
            else:
                continue

            # 进行预测
            try:
                result = thocr.predict(filepath)

                # 记录结果
                test_result = {
                    'filename': filename,
                    'true_char': true_char,
                    'true_structure': true_structure_label,
                    'pred_char': result['character'],
                    'pred_structure': result['structure'],
                    'char_confidence': result['confidence']['character'],
                    'structure_confidence': result['confidence']['structure'],
                    'char_correct': true_char == result['character'],
                    'structure_correct': true_structure_label == result['structure'],
                    'both_correct': (true_char == result['character']) and (true_structure_label == result['structure'])
                }
                results.append(test_result)

                # 记录混淆矩阵数据
                confusion_data.append({
                    'true_structure': true_structure_label,
                    'pred_structure': result['structure'],
                    'true_char': true_char,
                    'pred_char': result['character']
                })

                status = "✅" if test_result['both_correct'] else "❌"
                print(f"{status} {filename}: 结构({true_structure_label}→{result['structure']}) "
                      f"字符({true_char}→{result['character']}) "
                      f"置信度: 结构{result['confidence']['structure']:.2%}, 字符{result['confidence']['character']:.2%}")

            except Exception as e:
                print(f"❌ {filename}: 识别失败 - {e}")

    # 生成统计报告
    generate_test_report(results, confusion_data)

    # 生成可视化图表
    generate_visualizations(results, confusion_data)

    # 生成示例识别结果图
    generate_example_results(thocr, test_dir, results)


def generate_test_report(results, confusion_data):
    """生成测试报告"""

    print("\n" + "=" * 60)
    print("集成系统测试报告")
    print("=" * 60)

    df = pd.DataFrame(results)

    # 基础统计
    total_tests = len(results)
    structure_accuracy = df['structure_correct'].mean() * 100
    char_accuracy = df['char_correct'].mean() * 100
    both_accuracy = df['both_correct'].mean() * 100

    print(f"测试样本总数: {total_tests}")
    print(f"结构分类准确率: {structure_accuracy:.2f}%")
    print(f"文字识别准确率: {char_accuracy:.2f}%")
    print(f"端到端准确率: {both_accuracy:.2f}%")


    print("\n按结构类型统计:")
    structure_stats = df.groupby('true_structure').agg({
        'structure_correct': 'mean',
        'char_correct': 'mean',
        'both_correct': 'mean',
        'filename': 'count'
    }).round(4) * 100

    structure_stats.columns = ['结构准确率%', '文字准确率%', '端到端准确率%', '样本数']
    print(structure_stats)


    print(f"\n平均置信度:")
    print(f"  结构分类: {df['structure_confidence'].mean():.2%}")
    print(f"  文字识别: {df['char_confidence'].mean():.2%}")


    df.to_csv('thocr_test_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至: thocr_test_results.csv")


def generate_visualizations(results, confusion_data):
    """生成可视化图表"""

    df = pd.DataFrame(results)
    confusion_df = pd.DataFrame(confusion_data)

    # 设置中文字体
    setup_chinese_font()

    # 修复样式问题
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 准确率对比图 - 使用英文避免字体问题
    accuracy_data = {
        'Structure\nClassification': df['structure_correct'].mean() * 100,
        'Character\nRecognition': df['char_correct'].mean() * 100,
        'End-to-End': df['both_correct'].mean() * 100
    }

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = axes[0, 0].bar(accuracy_data.keys(), accuracy_data.values(), color=colors)
    axes[0, 0].set_title('THOCR System Performance', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_ylim(0, 105)

    # 在柱状图上添加数值
    for bar, v in zip(bars, accuracy_data.values()):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 2. 结构类型性能对比 - 使用英文
    structure_performance = df.groupby('true_structure')['both_correct'].mean() * 100
    structure_names = {
        'enclosed': 'Enclosed',
        'horizontal': 'Horizontal',
        'single': 'Single',
        'vertical': 'Vertical'
    }
    structure_labels = [structure_names.get(s, s) for s in structure_performance.index]

    bars = axes[0, 1].bar(structure_labels, structure_performance.values, color='#2E86AB')
    axes[0, 1].set_title('Accuracy by Structure Type', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_ylim(0, 105)

    for bar, v in zip(bars, structure_performance.values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 3. 置信度分布
    axes[1, 0].hist(df['structure_confidence'], bins=20, alpha=0.7,
                    label='Structure Classification', color='#2E86AB')
    axes[1, 0].hist(df['char_confidence'], bins=20, alpha=0.7,
                    label='Character Recognition', color='#A23B72')
    axes[1, 0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # 4. 结构分类混淆矩阵 - 使用英文
    if len(confusion_df) > 0:
        # 将标签转换为英文
        confusion_df_eng = confusion_df.copy()
        label_map = {
            'enclosed': 'Enclosed',
            'horizontal': 'Horizontal',
            'single': 'Single',
            'vertical': 'Vertical'
        }
        confusion_df_eng['true_structure'] = confusion_df_eng['true_structure'].map(label_map)
        confusion_df_eng['pred_structure'] = confusion_df_eng['pred_structure'].map(label_map)

        structure_confusion = pd.crosstab(
            confusion_df_eng['true_structure'],
            confusion_df_eng['pred_structure'],
            rownames=['True Structure'],
            colnames=['Predicted Structure']
        )
        sns.heatmap(structure_confusion, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Structure Classification\nConfusion Matrix', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No confusion matrix data',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Structure Classification\nConfusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('thocr_performance_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    print(f" Performance analysis chart saved to: thocr_performance_analysis.png")


def generate_example_results(thocr, test_dir, results, num_examples=12):
    """生成示例识别结果图"""

    if len(results) == 0:
        print("No result data available for example generation")
        return

    # 选择一些有代表性的例子
    df = pd.DataFrame(results)
    correct_examples = df[df['both_correct'] == True].head(6)
    wrong_examples = df[df['both_correct'] == False].head(6)

    selected_examples = pd.concat([correct_examples, wrong_examples]).head(num_examples)

    # 创建结果图
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (_, example) in enumerate(selected_examples.iterrows()):
        if idx >= len(axes):
            break

        filepath = os.path.join(test_dir, example['filename'])
        try:
            image = Image.open(filepath)
            axes[idx].imshow(image)
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Image load failed\n{example["filename"]}',
                           ha='center', va='center', fontsize=10)

        axes[idx].axis('off')

        # 设置标题颜色：正确为绿色，错误为红色
        color = 'green' if example['both_correct'] else 'red'

        # 使用英文标签避免字体问题
        structure_map = {
            'enclosed': 'E', 'horizontal': 'H', 'single': 'S', 'vertical': 'V'
        }

        title = (f"True: {example['true_char']}({structure_map[example['true_structure']]})\n"
                 f"Pred: {example['pred_char']}({structure_map[example['pred_structure']]})\n"
                 f"Conf: {example['char_confidence']:.1%}")

        axes[idx].set_title(title, color=color, fontsize=9, pad=6)

    # 隐藏多余的子图
    for idx in range(len(selected_examples), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('THOCR Recognition Examples (Green: Correct, Red: Incorrect)',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('thocr_example_results.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    print(f"🖼️ Example results saved to: thocr_example_results.png")


if __name__ == "__main__":
    # 初始化中文字体支持
    setup_chinese_font()
    test_integrated_system()