import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)


# 1. 数据加载和探索
def load_data(train_path, test_path):
    """
    加载训练集和测试集数据
    """
    train_images = []
    train_labels = []
    test_images = []
    test_filenames = []

    # 类别映射
    class_names = ['Sugar beet', 'Scentless Mayweed', 'Loose Silky-bent',
                   'Common wheat', 'Black-grass']

    print("加载训练数据...")
    # 加载训练数据
    for class_name in class_names:
        class_path = os.path.join(train_path, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    train_images.append(img)
                    train_labels.append(class_name)

    print("加载测试数据...")
    # 加载测试数据
    test_path_full = test_path
    for img_file in os.listdir(test_path_full):
        img_path = os.path.join(test_path_full, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            test_images.append(img)
            test_filenames.append(img_file)

    print(f"训练集大小: {len(train_images)}, 测试集大小: {len(test_images)}")
    print(f"类别分布:")
    for class_name in class_names:
        count = train_labels.count(class_name)
        print(f"  {class_name}: {count}张")

    return train_images, train_labels, test_images, test_filenames, class_names


def preprocess_image(image, target_size=(256, 256)):
    """
    图像预处理：调整大小并标准化
    """
    img = cv2.resize(image, target_size)
    return img


def extract_green_mask_features(image):
    """
    提取绿色区域掩码相关特征 - 固定维度
    """
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义绿色范围的HSV值
    lower_green1 = np.array([35, 40, 40])
    upper_green1 = np.array([85, 255, 255])
    lower_green2 = np.array([25, 40, 40])
    upper_green2 = np.array([95, 255, 255])

    # 创建绿色掩码
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    green_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # 固定维度的掩码特征数组
    mask_features = np.zeros(23)  # 预先分配固定大小

    # 1. 绿色区域占比 (1维)
    mask_area = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
    mask_features[0] = mask_area

    # 2. 颜色统计特征 (6维: BGR均值和标准差)
    if np.sum(green_mask > 0) > 0:
        mean_bgr, std_bgr = cv2.meanStdDev(image, mask=green_mask)
        mask_features[1:7] = np.concatenate([mean_bgr.flatten(), std_bgr.flatten()])

    # 3. Hu矩特征 (7维)
    moments = cv2.moments(green_mask)
    if moments['m00'] != 0:
        hu_moments = cv2.HuMoments(moments).flatten()
        # 对数变换使Hu矩更稳定
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        mask_features[7:14] = hu_moments

    # 4. 形状特征 (9维)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
        else:
            compactness = 0

        # 计算边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        # 椭圆拟合
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_w, ellipse_h = ellipse[1]
        else:
            ellipse_w, ellipse_h = 0, 0

        # 凸包特征
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        shape_features = np.array([area, perimeter, compactness, aspect_ratio,
                                   w, h, ellipse_w, ellipse_h, solidity])
        mask_features[14:23] = shape_features

    return mask_features, green_mask


def extract_gray_texture_features(image, green_mask=None):
    """
    提取灰度纹理特征 - 固定维度
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 如果提供了绿色掩码，则只考虑绿色区域的纹理
    if green_mask is not None and np.sum(green_mask > 0) > 0:
        gray_masked = cv2.bitwise_and(gray, gray, mask=green_mask)
        mask_pixels = gray_masked[green_mask > 0]
        if len(mask_pixels) > 0:
            mean_val = np.mean(mask_pixels)
            std_val = np.std(mask_pixels)
            var_val = np.var(mask_pixels)
            max_val = np.max(mask_pixels)
            min_val = np.min(mask_pixels)
        else:
            mean_val = std_val = var_val = max_val = min_val = 0
    else:
        gray_masked = gray
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        var_val = np.var(gray)
        max_val = np.max(gray)
        min_val = np.min(gray)

    # 固定维度的纹理特征数组
    texture_features = np.zeros(35)  # 预先分配固定大小

    # 1. 灰度统计特征 (5维)
    texture_features[0:5] = [mean_val, std_val, var_val, max_val, min_val]

    # 2. Canny边缘特征 (5维)
    edges = cv2.Canny(gray_masked, 100, 200)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

    # 计算梯度
    sobelx = cv2.Sobel(gray_masked.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_masked.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    orientation = np.arctan2(sobely, sobelx)

    texture_features[5:10] = [edge_density,
                              np.mean(magnitude),
                              np.std(magnitude),
                              np.mean(orientation),
                              np.std(orientation)]

    # 3. LBP纹理特征 (10维)
    radius = 1
    n_points = 8 * radius
    try:
        lbp = local_binary_pattern(gray_masked, n_points, radius, method='uniform')
        n_bins = 10  # 固定bins数量
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        hist = hist / (hist.sum() + 1e-10)  # 归一化
        texture_features[10:20] = hist
    except:
        texture_features[10:20] = 0

    # 4. GLCM-like特征 (5维)
    # 计算局部方差
    gray_float = gray_masked.astype(np.float32)
    local_mean = cv2.blur(gray_float, (5, 5))
    local_mean_sq = cv2.blur(gray_float ** 2, (5, 5))
    local_var = np.maximum(local_mean_sq - local_mean ** 2, 0)

    texture_features[20:25] = [np.mean(local_var),
                               np.std(local_var),
                               np.max(local_var),
                               np.mean(local_mean),
                               np.std(local_mean)]

    # 5. Gabor滤波器响应 (10维)
    ksize = 31
    gabor_features = []

    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for sigma in [3.0, 5.0]:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            response = cv2.filter2D(gray_float, cv2.CV_32F, kernel)
            gabor_features.extend([np.mean(response), np.std(response)])

    texture_features[25:35] = gabor_features[:10]  # 确保不超过10个

    return texture_features


def extract_color_features(image, green_mask):
    """
    提取颜色空间特征 - 固定维度
    """
    # 固定维度的颜色特征数组
    color_features = np.zeros(100)  # 预先分配固定大小

    # 转换到不同颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    idx = 0

    # 对每个颜色空间计算掩码区域的统计
    color_spaces = [('BGR', image), ('HSV', hsv), ('Lab', lab)]

    for space_name, img_converted in color_spaces:
        if np.sum(green_mask > 0) > 0:
            # 计算掩码区域的均值和标准差 (6维: 3个通道的均值和标准差)
            mean_vals, std_vals = cv2.meanStdDev(img_converted, mask=green_mask)
            color_features[idx:idx + 6] = np.concatenate([mean_vals.flatten(), std_vals.flatten()])
            idx += 6

            # 计算简化颜色直方图 (每个通道8个bin，共24维)
            for channel in range(3):
                hist = cv2.calcHist([img_converted], [channel], green_mask, [8], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                color_features[idx:idx + 8] = hist
                idx += 8
        else:
            # 如果没有绿色区域，跳过这个颜色空间的特征
            idx += 30  # 6 + 24 = 30

    return color_features


def extract_hog_features_fixed(image):
    """
    提取固定维度的HOG特征
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算HOG特征
    try:
        features = hog(gray,
                       orientations=9,
                       pixels_per_cell=(32, 32),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       feature_vector=True)
        # 确保特征维度固定
        if len(features) > 81:
            features = features[:81]
        elif len(features) < 81:
            # 如果特征不足，填充零
            features = np.pad(features, (0, 81 - len(features)), 'constant')
    except:
        # 如果HOG提取失败，返回零向量
        features = np.zeros(81)

    return features


def extract_all_features(image):
    """
    提取所有特征 - 确保固定维度
    """
    try:
        # 预处理图像
        img_processed = preprocess_image(image)

        # 1. 提取绿色掩码特征 (23维)
        mask_features, green_mask = extract_green_mask_features(img_processed)

        # 2. 提取灰度纹理特征 (35维)
        texture_features = extract_gray_texture_features(img_processed, green_mask)

        # 3. 提取颜色特征 (100维)
        color_features = extract_color_features(img_processed, green_mask)

        # 4. 提取HOG特征 (81维)
        hog_features = extract_hog_features_fixed(img_processed)

        # 合并所有特征 (总共239维)
        all_features = np.concatenate([
            mask_features,  # 23维
            texture_features,  # 35维
            color_features,  # 100维
            hog_features  # 81维
        ])

        return all_features

    except Exception as e:
        # 如果特征提取失败，返回零向量
        print(f"特征提取错误: {e}")
        return np.zeros(239)  # 返回统一维度的零向量


def extract_features_for_dataset(images, max_images=None):
    """
    为整个数据集提取特征
    """
    features = []

    if max_images is not None:
        images = images[:max_images]

    print("提取特征中...")
    for i, image in enumerate(tqdm(images)):
        feature = extract_all_features(image)
        features.append(feature)

    # 转换为numpy数组前检查维度
    features_array = np.array(features)
    print(f"特征矩阵形状: {features_array.shape}")
    print(f"特征维度: {features_array.shape[1]}")

    return features_array


def train_and_evaluate(X_train, y_train, X_test, test_filenames, class_names):
    """
    训练模型并生成预测结果
    """
    # 编码标签
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"训练集特征维度: {X_train_scaled.shape}")
    print(f"测试集特征维度: {X_test_scaled.shape}")

    # 尝试不同的模型
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_split=30,
            min_samples_leaf=8,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'SVM': svm.SVC(
            C=1,
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.001,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        )
    }

    # 使用交叉验证评估模型
    print("\n模型交叉验证结果 (Mean F1 Score):")
    best_model = None
    best_score = 0
    best_model_name = ""

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 减少折数以加快速度

    for model_name, model in models.items():
        try:
            f1_scores = cross_val_score(
                model, X_train_scaled, y_train_encoded,
                cv=skf, scoring='f1_macro', n_jobs=-1
            )
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)

            print(f"{model_name}: {mean_f1:.4f} (+/- {std_f1:.4f})")

            if mean_f1 > best_score:
                best_score = mean_f1
                best_model = model
                best_model_name = model_name
        except Exception as e:
            print(f"{model_name} 交叉验证失败: {e}")

    if best_model is None:
        print("\n所有模型都失败了，使用默认的RandomForest")
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model_name = "RandomForest_Default"

    print(f"\n选择最佳模型: {best_model_name}")

    # 使用最佳模型在整个训练集上训练
    print("训练最佳模型...")
    best_model.fit(X_train_scaled, y_train_encoded)

    # 在训练集上评估
    y_pred_train = best_model.predict(X_train_scaled)
    train_f1 = f1_score(y_train_encoded, y_pred_train, average='macro')
    print(f"训练集F1分数: {train_f1:.4f}")

    # 生成测试集预测
    print("生成测试集预测...")
    y_pred_test = best_model.predict(X_test_scaled)
    y_pred_test_labels = le.inverse_transform(y_pred_test)

    # 创建预测结果的DataFrame
    predictions_df = pd.DataFrame({
        'ID': test_filenames,
        'Category': y_pred_test_labels
    })

    # 显示预测结果分布
    print("\n预测结果分布:")
    for class_name in class_names:
        count = (y_pred_test_labels == class_name).sum()
        print(f"  {class_name}: {count}张 ({count / len(y_pred_test_labels) * 100:.1f}%)")

    return predictions_df, best_model_name, best_score


def main():
    # 设置路径
    train_path = "/kaggle/input/neu2123/dataset-for-task2/dataset-for-task2/train"
    test_path = "/kaggle/input/neu2123/dataset-for-task2/dataset-for-task2/test"

    # 1. 加载数据
    print("=" * 50)
    print("数据加载")
    print("=" * 50)
    train_images, train_labels, test_images, test_filenames, class_names = load_data(train_path, test_path)

    # 2. 提取特征
    print("\n" + "=" * 50)
    print("特征提取")
    print("=" * 50)
    X_train = extract_features_for_dataset(train_images)
    X_test = extract_features_for_dataset(test_images)

    # 3. 训练模型和生成预测
    print("\n" + "=" * 50)
    print("模型训练与评估")
    print("=" * 50)
    predictions_df, best_model_name, best_score = train_and_evaluate(
        X_train, train_labels, X_test, test_filenames, class_names
    )

    # 4. 保存预测结果
    print("\n" + "=" * 50)
    print("保存结果")
    print("=" * 50)

    # 保存到Kaggle工作目录
    output_file = "/kaggle/working/submission-for-task1.csv"
    predictions_df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")

    # 检查文件是否保存成功
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"文件大小: {file_size:.2f} KB")
        print(f"文件前10行内容:")
        print(predictions_df.head(10))
    else:
        print("警告: 文件保存失败!")

    print(f"\n最佳模型: {best_model_name}")
    print(f"交叉验证Mean F1 Score: {best_score:.4f}")
    print(f"测试集预测数量: {len(predictions_df)}")

    # 显示文件路径信息
    print("\n" + "=" * 50)
    print("文件信息")
    print("=" * 50)
    print(f"Kaggle工作目录: /kaggle/working/")
    print(f"提交文件位置: {output_file}")

    return predictions_df


if __name__ == "__main__":
    # 运行主程序
    predictions = main()