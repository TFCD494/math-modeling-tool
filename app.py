# ============================================================
# 数学建模大赛前期数据分析工具（修正版）
# ============================================================
import re
import io
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, shapiro, probplot
from sklearn.model_selection import train_test_split
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix
)
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
    OLSInfluence
)
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white
)
from statsmodels.stats.stattools import durbin_watson
# 新增：解析 PDF 和 Word 文件
import PyPDF2
import docx

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="数学建模前期数据分析工具",
    layout="wide"
)
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 一、通用工具函数
# ============================================================
def clear_model_session_state():
    """清除旧的模型结果，避免页面显示与当前设置不一致。"""
    model_keys = [
        "fitted_result",
        "fitted_model",
        "final_model_type",
        "model_meta",
        "X_for_assumption",
        "vif_table"
    ]

    for key in model_keys:
        st.session_state.pop(key, None)

def build_analysis_signature(
    target,
    predictors,
    variable_types,
    group_col,
    missing_method,
    outlier_method,
    outlier_action,
    robust_se,
    use_test_set,
    test_size
):
    """根据当前分析设置生成唯一签名。"""
    signature_data = {
        "target": target,
        "predictors": sorted(predictors),
        "variable_types": variable_types,
        "group_col": group_col,
        "missing_method": missing_method,
        "outlier_method": outlier_method,
        "outlier_action": outlier_action,
        "robust_se": robust_se,
        "use_test_set": use_test_set,
        "test_size": test_size
    }

    return repr(signature_data)
def clean_column_name(name):
    """规范列名。"""
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\u4e00-\u9fff]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    if not name:
        name = "未命名变量"
    return name

def make_unique_columns(columns):
    """处理重复列名。"""
    result = []
    counter = {}
    for col in columns:
        base = clean_column_name(col)
        counter[base] = counter.get(base, 0) + 1
        if counter[base] == 1:
            result.append(base)
        else:
            result.append(f"{base}_{counter[base]}")
    return result

def dataframe_download(df, filename):
    """生成 CSV 下载按钮。"""
    data = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label=f"下载 {filename}",
        data=data,
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}"
    )

def try_parse_datetime(series):
    """尝试识别日期时间变量。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        valid_rate = parsed.notna().mean()
        if valid_rate >= 0.8:
            return parsed
    return None

def is_suspicious_id_column(series, name):
    """
    判断某列是否可能是 ID、序号或对象编号。
    只用于默认排除，不会自动永久删除。
    """
    name_text = str(name).lower()
    id_keywords = [
        "id", "编号", "序号", "代码", "编码", "样本号",
        "学生号", "患者号", "姓名", "name"
    ]
    keyword_flag = any(keyword in name_text for keyword in id_keywords)
    unique_rate = series.nunique(dropna=True) / max(len(series), 1)
    numeric = pd.to_numeric(series, errors="coerce")
    sequential_flag = False
    if numeric.notna().mean() >= 0.95:
        values = numeric.dropna().sort_values().values
        if len(values) >= 3:
            differences = np.diff(values)
            sequential_flag = np.allclose(differences, differences[0])
    return keyword_flag or (unique_rate >= 0.95 and sequential_flag)

# ============================================================
# 新增：从上传的 PDF / Word / TXT 文件中提取文本
# ============================================================
def extract_text_from_file(uploaded_file):
    """从上传的 PDF、Word 或 TXT 文件中提取文本。"""
    if uploaded_file is None:
        return ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'pdf':
            text = ""
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        elif file_type == 'txt':
            return uploaded_file.read().decode('utf-8')
        else:
            return ""
    except Exception as e:
        return f"[文件解析失败: {e}]"

# ============================================================
# 二、题型智能识别
# ============================================================
def clean_extracted_problem_text(text):
    """统一换行、清理多余空白字符、统一小写"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text
# ========== 新增：基于 TF-IDF 和余弦相似度的题型识别 ==========
def classify_by_tfidf(problem_text):
    """
    使用 jieba 分词 + TF-IDF 余弦相似度进行题型分类。
    返回各个题型的相似度分数。
    """
    if not isinstance(problem_text, str) or not problem_text.strip():
        return {}

    # 1. 准备每个题型的“标准描述文本”（语料库）
    #    这些描述应该包含该题型最典型的方法、术语
    corpus = {
        "评价类": (
            "层次分析法 AHP 模糊综合评价 TOPSIS 熵权法 灰色关联度 数据包络分析 "
            "优劣解距离法 指标权重 指标体系 综合得分 排序 评估 绩效考核 可行性研究 "
            "多属性决策 主成分分析 因子分析 优劣评价 综合排名 满意度评价"
        ),
        "预测类": (
            "时间序列 ARIMA 指数平滑 灰色预测 GM(1,1) 回归分析 趋势外推 "
            "神经网络 深度学习 LSTM 支持向量回归 SVR 随机森林回归 XGBoost "
            "增长率 预测值 未来走势 短期预测 中长期预测 预报 拟合 插值 变化规律"
        ),
        "优化类": (
            "线性规划 整数规划 0-1规划 非线性规划 动态规划 多目标优化 目标规划 "
            "遗传算法 粒子群算法 模拟退火 蚁群算法 禁忌搜索 启发式算法 "
            "最短路径 最大流 最小生成树 网络流 调度 排班 指派 运输问题 "
            "库存优化 资源分配 成本最小化 利润最大化 约束条件 最优解 方案优选"
        ),
        "机理分析类": (
            "微分方程 偏微分方程 常微分方程 动力学模型 传染病模型 SIR SEIR "
            "Logistic方程 捕食者-猎物模型 种群增长 扩散方程 反应扩散 对流-扩散 "
            "牛顿力学 流体力学 传热 电磁场 化学反应动力学 恒温恒湿 "
            "稳定平衡 分岔 相图 数值解 欧拉法 龙格-库塔法 有限元"
        ),
        "分类类": (
            "逻辑回归 Logistic回归 支持向量机 SVM 决策树 随机森林 K近邻 KNN "
            "朴素贝叶斯 神经网络 深度学习 CNN 判别分析 聚类分析 K-means 层次聚类 "
            "二分类 多分类 混淆矩阵 准确率 精确率 召回率 F1值 ROC曲线 AUC "
            "异常检测 模式识别 图像识别 文本分类 诊断 判定 识别 筛选 检出"
        )
    }

    # 2. 构建“待分类文本”与“各类别标准文本”组成的列表
    categories = list(corpus.keys())
    documents = [problem_text]  # 第一个文档是赛题原文
    for cat in categories:
        documents.append(corpus[cat])  # 后面跟着每个类别的描述

    # 3. 用 jieba 分词，并用 TfidfVectorizer 转换
    def tokenize(text):
        # jieba 分词，返回用空格分隔的词
        return " ".join(jieba.cut(text))

    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    except Exception:
        return {}

    # 4. 计算“赛题文本”与“每个类别描述”的余弦相似度
    #    tfidf_matrix 的第 0 行是赛题，第 1~最后是各类别
    problem_vec = tfidf_matrix[0:1]        # 赛题向量
    category_vecs = tfidf_matrix[1:]       # 各类别向量
    similarities = cosine_similarity(problem_vec, category_vecs).flatten()

    # 5. 整理输出：每个类别得到一个 0~1 之间的相似度分数
    result = {}
    for i, cat in enumerate(categories):
        result[cat] = float(similarities[i])

    return result
def multi_label_classify_problem_text(problem_text):
    """多标签题型识别：融合关键词匹配 + TF-IDF 相似度"""
    text = clean_extracted_problem_text(problem_text)
    if not text:
        return {
            "main_type": "未识别",
            "all_detected_labels": [],
            "label_scores": {},
            "sub_question_context": []
        }

    # ====== 方法一：传统关键词匹配（保留，作为辅助） ======
    keyword_sets = {
        "评价类": [
            "评价", "排序", "打分", "评选", "满意度", "权重", "层次分析",
            "topsis", "优劣", "综合评价", "评估", "排名", "优劣分析",
            "优劣等级", "综合评估", "指标体系", "可信度", "可行性", "综合排名"
        ],
        "预测类": [
            "预测", "趋势", "未来", "增长", "回归", "时间序列", "估计",
            "预报", "拟合", "外推", "增长率", "变化规律", "推算", "短期",
            "中期", "长期", "走势", "预测模型"
        ],
        "优化类": [
            "最大", "最小", "最优", "成本", "利润", "资源", "调度", "运输",
            "规划", "约束", "路径", "分配", "库存", "0-1", "整数", "安排",
            "调度", "收益最大", "总成本最小", "效率最高", "最优方案", "优化方案",
            "约束条件", "最优决策", "资源配置", "路径规划"
        ],
        "机理分析类": [
            "微分方程", "变化率", "传染病", "扩散", "物理", "力学", "温度",
            "浓度", "平衡点", "稳定性", "logistic", "sir", "动力学", "演化",
            "传播", "运动方程", "物理规律", "相互作用", "演化规律", "动态变化"
        ],
        "分类类": [
            "分类", "判别", "识别", "判定", "判断", "异常检测", "诊断",
            "聚类", "模式识别", "区分", "判定标准", "类别归属", "异常判定"
        ]
    }

    keyword_scores = {}
    for label, keywords in keyword_sets.items():
        hit_count = sum(1 for kw in keywords if kw in text)
        keyword_scores[label] = hit_count

    # ====== 方法二：TF-IDF 余弦相似度 ======
    tfidf_scores = classify_by_tfidf(text)

    # ====== 融合两种方法的分数 ======
    # 把关键词分数归一化到 0~1 并加权，再与 TF-IDF 分数线性组合
    combined_scores = {}
    for label in keyword_sets.keys():
        # 关键词得分：最高10分封顶，除以10映射到0~1
        kw_norm = min(keyword_scores.get(label, 0), 10) / 10.0
        # TF-IDF 得分：本身就是 0~1
        tfidf_norm = tfidf_scores.get(label, 0.0)
        # 组合：关键词权重 0.3，TF-IDF 权重 0.7
        combined_scores[label] = 0.3 * kw_norm + 0.7 * tfidf_norm

    # 按组合分数排序
    all_detected_labels = sorted(
        [label for label, score in combined_scores.items() if score > 0.1],
        key=lambda x: combined_scores[x],
        reverse=True
    )
    main_type = all_detected_labels[0] if all_detected_labels else "未识别"

    # 提取子问题片段（保留原逻辑）
    sub_question_patterns = [
        r"问题\s*[一二三四五六七八九十0-9]+[、\s.]",
        r"\d+[\)）、.]"
    ]
    sub_question_context = []
    for pattern in sub_question_patterns:
        matches = list(re.finditer(pattern, problem_text))
        for m in matches:
            start_pos = m.start()
            snippet_end = min(start_pos + 250, len(problem_text))
            snippet = problem_text[start_pos:snippet_end].strip()
            sub_question_context.append(snippet)
    sub_question_context = list(dict.fromkeys(sub_question_context))[:10]

    return {
        "main_type": main_type,
        "all_detected_labels": all_detected_labels,
        "label_scores": combined_scores,
        "sub_question_context": sub_question_context
    }

def auto_detect_problem_type(problem_text):
    """向后兼容旧接口"""
    result = multi_label_classify_problem_text(problem_text)
    return result["main_type"]

# ============================================================
# 三、变量类型识别
# ============================================================
def classify_variable(series):
    """自动识别变量类型。"""
    parsed_time = try_parse_datetime(series)
    if parsed_time is not None:
        return "时间"
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_rate = numeric.notna().mean()
    if numeric_rate < 0.8:
        return "分类"
    values = numeric.dropna()
    if len(values) == 0:
        return "连续"
    unique_values = set(values.unique().tolist())
    unique_count = values.nunique()
    if unique_count <= 2 and unique_values.issubset({0, 1}):
        return "分类"
    is_integer = np.all(np.isclose(values, np.round(values)))
    is_nonnegative = (values >= 0).all()
    if is_integer and is_nonnegative and unique_count <= 20 and values.max() >= 5:
        return "次数"
    return "连续"

def convert_types(df, variable_types):
    """依据用户确认的变量类型转换数据。"""
    result = df.copy()
    for col, var_type in variable_types.items():
        if col not in result.columns:
            continue
        if var_type in ["连续", "次数"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        elif var_type == "时间":
            result[col] = pd.to_datetime(result[col], errors="coerce")
        elif var_type == "分类":
            result[col] = result[col].astype("string")
    return result

# ============================================================
# 四、符号表
# ============================================================
def create_sample_symbol_table(df):
    """将第一列中的每一行映射为样本符号。"""
    if df.shape[1] == 0:
        return pd.DataFrame()
    first_col = df.columns[0]
    return pd.DataFrame({
        "样本原始名称": df[first_col].astype(str),
        "样本符号": [f"A{i + 1}" for i in range(len(df))]
    })

def create_variable_symbol_table(df, target, predictors, variable_types):
    """生成变量符号表。"""
    rows = []
    all_variables = [target] + predictors
    for i, col in enumerate(all_variables):
        rows.append({
            "变量符号": f"x{i + 1}" if col != target else "y",
            "原始列名": col,
            "变量角色": "因变量" if col == target else "自变量",
            "变量类型": variable_types.get(col, ""),
            "单位": "",
            "变量含义": ""
        })
    return pd.DataFrame(rows)

# ============================================================
# 五、缺失值处理
# ============================================================
def fill_missing_values(df, variable_types, method):
    """
    对自变量进行缺失值处理。

    注意：
    因变量不应该在这里进行插补。
    因变量缺失样本应在调用本函数前删除。
    
    返回：
    result, report, deleted_rows, imputed_cells
    """
    result = df.copy()
    report_rows = []
    deleted_rows = 0
    imputed_cells = 0
    original_row_count = len(result)

    # ✅ 修正：匹配侧边栏的选项 "删除含缺失值的行"
    if method == "删除含缺失值的行":
        missing_rows = result[list(variable_types.keys())].isna().any(axis=1)
        deleted_rows = int(missing_rows.sum())
        result = result.loc[~missing_rows].copy()
        for col in variable_types:
            before = int(df[col].isna().sum())
            report_rows.append({
                "变量": col,
                "原始缺失数": before,
                "处理方式": "删除所在行",
                "实际插补数": 0,
                "剩余缺失数": int(result[col].isna().sum())
            })
        return result, pd.DataFrame(report_rows), deleted_rows, 0

    for col, var_type in variable_types.items():
        if col not in result.columns:
            continue
        before = int(result[col].isna().sum())
        if before == 0:
            report_rows.append({
                "变量": col,
                "原始缺失数": 0,
                "处理方式": "无缺失",
                "实际插补数": 0,
                "剩余缺失数": 0
            })
            continue
        if var_type in ["连续", "次数"]:
            result[col] = result[col].interpolate(method="linear", limit_direction="both")
            median_value = result[col].median()
            if pd.notna(median_value):
                result[col] = result[col].fillna(median_value)
            treatment = "线性插值+中位数"
        elif var_type == "时间":
            result[col] = result[col].ffill().bfill()
            treatment = "前向填充+后向填充"
        else:
            mode = result[col].mode(dropna=True)
            if len(mode) > 0:
                result[col] = result[col].fillna(mode.iloc[0])
            treatment = "众数填补"
        after = int(result[col].isna().sum())
        actual_imputed = max(0, before - after)
        imputed_cells += actual_imputed
        report_rows.append({
            "变量": col,
            "原始缺失数": before,
            "处理方式": treatment,
            "实际插补数": actual_imputed,
            "剩余缺失数": after
        })
    return result, pd.DataFrame(report_rows), deleted_rows, imputed_cells

# ============================================================
# 六、异常值处理
# ============================================================
def detect_outliers(df, numeric_columns, method):
    """返回逐列异常值统计和行级异常标记。"""
    result = df.copy()
    result["_异常行"] = False
    report_rows = []
    for col in numeric_columns:
        values = pd.to_numeric(result[col], errors="coerce")
        if method == "3σ":
            mean_value = values.mean()
            std_value = values.std()
            if pd.isna(std_value) or std_value == 0:
                outlier_mask = pd.Series(False, index=result.index)
            else:
                outlier_mask = ~values.between(mean_value - 3 * std_value, mean_value + 3 * std_value)
        elif method == "IQR":
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                outlier_mask = pd.Series(False, index=result.index)
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = ~values.between(lower, upper)
        else:
            outlier_mask = pd.Series(False, index=result.index)
        outlier_mask = outlier_mask.fillna(False)
        result["_异常行"] = result["_异常行"] | outlier_mask
        report_rows.append({"变量": col, "异常值数量": int(outlier_mask.sum())})
    return result, pd.DataFrame(report_rows)

# ============================================================
# 七、相关性分析
# ============================================================
def correlation_table(df, target, predictors, variable_types):
    """计算连续变量和次数变量与因变量的相关性。"""
    rows = []
    target_values = pd.to_numeric(df[target], errors="coerce")
    for col in predictors:
        if variable_types.get(col) not in ["连续", "次数"]:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        valid = pd.concat([target_values, values], axis=1).dropna()
        if len(valid) < 3:
            continue
        if valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
            continue
        try:
            pearson_value, pearson_p = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
            spearman_value, spearman_p = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
            rows.append({
                "变量": col,
                "Pearson相关系数": pearson_value,
                "Pearson_P值": pearson_p,
                "Spearman相关系数": spearman_value,
                "Spearman_P值": spearman_p
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

# ============================================================
# 八、模型数据构造
# ============================================================
def build_model_data(df, target, predictors, variable_types, group_col=None):
    """构造模型数据。分类自变量自动独热编码。"""
    use_cols = [target] + predictors
    if group_col not in [None, "无"]:
        use_cols.append(group_col)
    use_cols = list(dict.fromkeys(use_cols))
    selected = df[use_cols].copy()

    target_mapping = None
    target_type = variable_types[target]

    # 处理因变量
    if target_type in ["连续", "次数"]:
        selected[target] = pd.to_numeric(selected[target], errors="coerce")
    elif target_type == "分类":
        selected[target] = selected[target].astype("string")
        categories = sorted(selected[target].dropna().unique().tolist())
        if len(categories) < 2:
            raise ValueError("分类因变量至少需要两个类别。")
        target_mapping = {category: i for i, category in enumerate(categories)}
        selected[target] = selected[target].map(target_mapping)
    elif target_type == "时间":
        raise ValueError("时间型因变量不适合直接用于当前回归模块。")

    # 处理自变量
    for col in predictors:
        var_type = variable_types[col]
        if var_type in ["连续", "次数"]:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
        elif var_type == "时间":
            dates = pd.to_datetime(selected[col], errors="coerce")
            selected[col] = (dates - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400
        elif var_type == "分类":
            selected[col] = selected[col].astype("string")

    required_cols = [target] + predictors
    if group_col not in [None, "无"]:
        required_cols.append(group_col)
    selected = selected.dropna(subset=required_cols)

    y = selected[target].copy()
    X = selected[predictors].copy()
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan)

    # 删除无变化的列
    constant_columns = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    if constant_columns:
        X = X.drop(columns=constant_columns)

    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if X.shape[1] == 0:
        raise ValueError("处理后没有有效的自变量。")
    X = sm.add_constant(X, has_constant="add")

    groups = None
    if group_col not in [None, "无"]:
        groups = selected.loc[valid_rows, group_col].astype(str)

    return (
        y.reset_index(drop=True),
        X.reset_index(drop=True),
        None if groups is None else groups.reset_index(drop=True),
        {
            "target_mapping": target_mapping,
            "feature_names": list(X.columns),
            "n_rows": len(y),
            "n_features": X.shape[1],
            "constant_columns": constant_columns
        }
    )

# ============================================================
# 九、模型推荐和拟合
# ============================================================
def detect_model_type(y, target_type, groups=None):
    """自动推荐候选模型。"""
    y_numeric = pd.to_numeric(y, errors="coerce").dropna()
    if target_type == "分类":
        unique_count = y.nunique()
        if unique_count == 2:
            return {"model_type": "二项Logistic回归", "reason": "因变量为二分类变量"}
        if unique_count > 2:
            return {"model_type": "多项Logistic回归", "reason": "因变量为多分类变量"}
    if target_type == "时间":
        return {"model_type": "时间序列或生存分析", "reason": "因变量为时间变量"}
    if target_type == "次数":
        if len(y_numeric) > 0 and (y_numeric >= 0).all() and np.allclose(y_numeric, np.round(y_numeric)):
            mean_value = y_numeric.mean()
            variance_value = y_numeric.var()
            if mean_value > 0 and variance_value > 1.5 * mean_value:
                return {"model_type": "负二项回归", "reason": "计数变量可能存在过度离散"}
            return {"model_type": "Poisson回归", "reason": "因变量为非负计数变量"}
    if target_type == "连续":
        if len(y_numeric) == 0:
            return {"model_type": "未识别", "reason": "因变量没有有效数值"}
        in_unit_interval = ((y_numeric >= 0).all() and (y_numeric <= 1).all())
        repeated = False
        if groups is not None:
            repeated = groups.nunique() < len(groups)
        if in_unit_interval and repeated:
            return {"model_type": "比例型混合效应模型", "reason": "比例型因变量且存在重复观测"}
        if in_unit_interval:
            return {"model_type": "Logit变换线性回归", "reason": "因变量取值在0到1之间"}
        if repeated:
            return {"model_type": "线性混合效应模型", "reason": "连续型因变量存在重复观测"}
        return {"model_type": "多元线性回归", "reason": "因变量为一般连续变量"}
    return {"model_type": "未识别", "reason": "无法判断模型类型"}

def logit_transform(y):
    """比例变量Logit变换。"""
    y = pd.Series(y, dtype=float)
    eps = 1e-6
    y = y.clip(lower=eps, upper=1 - eps)
    return np.log(y / (1 - y))

def inverse_logit(z):
    """Logit逆变换。"""
    z = np.asarray(z)
    z = np.clip(z, -700, 700)
    return 1 / (1 + np.exp(-z))

def validate_model_selection(y, target_type, model_type, groups=None):
    """
    检查因变量类型、取值范围、模型类型和分组变量是否匹配。
    返回：
        (True, "")       表示检查通过
        (False, 错误信息) 表示不允许拟合
    """

    if y is None or len(y) == 0:
        return False, "当前没有可用于建模的数据。"

    y_numeric = pd.to_numeric(pd.Series(y), errors="coerce")

    if y_numeric.isna().any():
        return False, "因变量中存在无法转换为数值的内容。"

    # 多元线性回归
    if model_type == "多元线性回归":
        if target_type not in ["连续", "次数"]:
            return (
                False,
                "多元线性回归通常要求因变量为连续变量或次数变量。"
            )

    # Logit 变换线性回归
    elif model_type == "Logit变换线性回归":
        if target_type not in ["连续", "次数"]:
            return (
                False,
                "Logit变换线性回归要求因变量为数值型变量。"
            )

        if ((y_numeric < 0) | (y_numeric > 1)).any():
            return (
                False,
                "Logit变换线性回归要求因变量取值位于0到1之间。"
            )

        if y_numeric.nunique() < 2:
            return (
                False,
                "Logit变换线性回归要求因变量至少包含两个不同取值。"
            )

    # 二项 Logistic 回归
    elif model_type == "二项Logistic回归":
        if target_type != "分类":
            return (
                False,
                "二项Logistic回归要求因变量类型为分类变量。"
            )

        if pd.Series(y).nunique() != 2:
            return (
                False,
                "二项Logistic回归要求因变量恰好包含两个类别。"
            )

    # 多项 Logistic 回归
    elif model_type == "多项Logistic回归":
        if target_type != "分类":
            return (
                False,
                "多项Logistic回归要求因变量类型为分类变量。"
            )

        if pd.Series(y).nunique() < 3:
            return (
                False,
                "多项Logistic回归至少需要三个类别。"
            )

    # Poisson 回归
    elif model_type == "Poisson回归":
        if target_type != "次数":
            return (
                False,
                "Poisson回归要求因变量类型为次数变量。"
            )

        if (y_numeric < 0).any():
            return (
                False,
                "Poisson回归要求因变量不能小于0。"
            )

        if not np.allclose(
            y_numeric.to_numpy(),
            np.round(y_numeric.to_numpy())
        ):
            return (
                False,
                "Poisson回归要求因变量为非负整数，当前数据中存在小数。"
            )

    # 负二项回归
    elif model_type == "负二项回归":
        if target_type != "次数":
            return (
                False,
                "负二项回归要求因变量类型为次数变量。"
            )

        if (y_numeric < 0).any():
            return (
                False,
                "负二项回归要求因变量不能小于0。"
            )

        if not np.allclose(
            y_numeric.to_numpy(),
            np.round(y_numeric.to_numpy())
        ):
            return (
                False,
                "负二项回归要求因变量为非负整数，当前数据中存在小数。"
            )

    # 线性混合效应模型
    elif model_type == "线性混合效应模型":
        if target_type not in ["连续", "次数"]:
            return (
                False,
                "线性混合效应模型要求因变量为数值型变量。"
            )

        if groups is None:
            return (
                False,
                "线性混合效应模型必须指定分组变量。"
            )

        if pd.Series(groups).nunique() < 2:
            return (
                False,
                "分组变量至少需要包含两个不同的组。"
            )

    # 比例型混合效应模型
    elif model_type == "比例型混合效应模型":
        if target_type not in ["连续", "次数"]:
            return (
                False,
                "比例型混合效应模型要求因变量为数值型变量。"
            )

        if ((y_numeric < 0) | (y_numeric > 1)).any():
            return (
                False,
                "比例型混合效应模型要求因变量取值位于0到1之间。"
            )

        if groups is None:
            return (
                False,
                "比例型混合效应模型必须指定分组变量。"
            )

        if pd.Series(groups).nunique() < 2:
            return (
                False,
                "分组变量至少需要包含两个不同的组。"
            )

    return True, ""

def fit_model(y, X, groups, model_type, robust_se=False):
    """拟合用户最终选择的模型。"""
    if model_type == "多元线性回归":
        if robust_se:
            model = sm.OLS(y, X).fit(cov_type="HC3")
        else:
            model = sm.OLS(y, X).fit()
        prediction = model.predict(X)
        return {"model": model, "display_y": y, "prediction": prediction, "model_type": model_type}

    if model_type == "Logit变换线性回归":
        y_transformed = logit_transform(y)
        if robust_se:
            model = sm.OLS(y_transformed, X).fit(cov_type="HC3")
        else:
            model = sm.OLS(y_transformed, X).fit()
        prediction_transformed = model.predict(X)
        prediction = inverse_logit(prediction_transformed)
        return {"model": model, "display_y": y, "prediction": prediction, "transformed_y": y_transformed, "model_type": model_type}

    if model_type == "线性混合效应模型":
        if groups is None:
            raise ValueError("线性混合效应模型必须指定分组变量。")
        model = sm.MixedLM(endog=y, exog=X, groups=groups).fit(reml=False, method="lbfgs")
        return {"model": model, "display_y": y, "prediction": model.predict(X), "model_type": model_type}

    if model_type == "比例型混合效应模型":
        if groups is None:
            raise ValueError("比例型混合效应模型必须指定分组变量。")
        y_transformed = logit_transform(y)
        model = sm.MixedLM(endog=y_transformed, exog=X, groups=groups).fit(reml=False, method="lbfgs")
        prediction = inverse_logit(model.predict(X))
        return {"model": model, "display_y": y, "prediction": prediction, "transformed_y": y_transformed, "model_type": model_type}

    if model_type == "Poisson回归":
        if (pd.to_numeric(y) < 0).any():
            raise ValueError("Poisson回归要求因变量为非负整数。")
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        return {"model": model, "display_y": y, "prediction": model.predict(X), "model_type": model_type}

    if model_type == "负二项回归":
        if (pd.to_numeric(y) < 0).any():
            raise ValueError("负二项回归要求因变量为非负数。")
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
        return {"model": model, "display_y": y, "prediction": model.predict(X), "model_type": model_type}

    if model_type == "二项Logistic回归":
        if y.nunique() != 2:
            raise ValueError("二项Logistic回归要求因变量恰好有两个类别。")
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        # statsmodels GLM.predict() 默认返回概率（linear=False）
        probability = model.predict(X)
        prediction = (probability >= 0.5).astype(int)
        return {
            "model": model,
            "display_y": y,
            "prediction": prediction,
            "probability": probability,
            "model_type": model_type
        }

    if model_type == "多项Logistic回归":
        if y.nunique() < 3:
            raise ValueError("多项Logistic回归要求至少三个类别。")
        model = sm.MNLogit(y, X).fit(disp=False, maxiter=300)
        probability = model.predict(X)
        probability_array = np.asarray(probability)
        prediction = probability_array.argmax(axis=1)
        return {
            "model": model,
            "display_y": y,
            "prediction": prediction,
            "probability": probability,
            "model_type": model_type
        }

    raise ValueError(f"暂不支持的模型：{model_type}")

# ============================================================
# 十、结果表和诊断
# ============================================================
def model_is_converged(model):
    """检查模型是否收敛。"""
    if hasattr(model, "converged"):
        return bool(model.converged)
    if hasattr(model, "mle_retvals"):
        return bool(model.mle_retvals.get("converged", True))
    return True

def significance_label(p):
    if pd.isna(p):
        return "无法判断"
    if p < 0.01:
        return "极显著"
    if p < 0.05:
        return "显著"
    if p < 0.10:
        return "边际显著"
    return "不显著"

def make_result_table(model, model_type):
    """生成模型系数表。"""
    params = model.params
    bse = model.bse
    pvalues = model.pvalues
    try:
        conf_int = model.conf_int()
    except Exception:
        conf_int = None

    if isinstance(params, pd.Series):
        result = pd.DataFrame({
            "变量": params.index,
            "回归系数": params.values,
            "标准误": np.asarray(bse),
            "P值": np.asarray(pvalues)
        })
        if conf_int is not None:
            result["置信区间下限"] = conf_int.iloc[:, 0].values
            result["置信区间上限"] = conf_int.iloc[:, 1].values
        if "Logistic" in model_type:
            result["优势比_OR"] = np.exp(result["回归系数"])
        result["显著性判断"] = result["P值"].apply(significance_label)
        return result

    # MNLogit
    params_df = pd.DataFrame(params)
    bse_df = pd.DataFrame(bse)
    pvalues_df = pd.DataFrame(pvalues)
    rows = []
    for variable in params_df.index:
        for category in params_df.columns:
            coef = params_df.loc[variable, category]
            p_value = pvalues_df.loc[variable, category]
            rows.append({
                "变量": variable,
                "类别": category,
                "回归系数": coef,
                "标准误": bse_df.loc[variable, category],
                "P值": p_value,
                "优势比_OR": np.exp(coef),
                "显著性判断": significance_label(p_value)
            })
    return pd.DataFrame(rows)

def calculate_vif(X):
    """计算VIF。"""
    X_df = X.copy()
    if "const" in X_df.columns:
        X_df = X_df.drop(columns=["const"])
    X_df = X_df.loc[:, X_df.nunique(dropna=True) > 1]
    if X_df.shape[1] <= 1:
        return pd.DataFrame(columns=["变量", "VIF"])
    rows = []
    for i, col in enumerate(X_df.columns):
        try:
            value = variance_inflation_factor(X_df.astype(float).values, i)
        except Exception:
            value = np.inf
        rows.append({"变量": col, "VIF": value})
    return pd.DataFrame(rows)

def make_metric_table(fitted_result):
    """依据模型类别生成合适的评价指标。"""
    model = fitted_result["model"]
    model_type = fitted_result["model_type"]
    y = np.asarray(fitted_result["display_y"]).astype(float)
    prediction = np.asarray(fitted_result["prediction"]).astype(float)
    rows = [
        ["模型类型", model_type],
        ["样本量", len(y)],
        ["参数数量", len(np.asarray(model.params).reshape(-1))]
    ]
    if hasattr(model, "aic"):
        rows.append(["AIC", model.aic])
    if hasattr(model, "bic"):
        try:
            rows.append(["BIC", model.bic])
        except Exception:
            pass

    if model_type in ["多元线性回归", "Logit变换线性回归"]:
        rows.extend([
            ["RMSE", np.sqrt(mean_squared_error(y, prediction))],
            ["MAE", mean_absolute_error(y, prediction)],
            ["R²", r2_score(y, prediction)]
        ])
        if hasattr(model, "rsquared"):
            rows.append(["模型R²", model.rsquared])
        if hasattr(model, "rsquared_adj"):
            rows.append(["调整R²", model.rsquared_adj])
        if hasattr(model, "fvalue"):
            rows.append(["F统计量", model.fvalue])
        if hasattr(model, "f_pvalue"):
            rows.append(["F检验P值", model.f_pvalue])

    elif model_type in ["Poisson回归", "负二项回归"]:
        rows.extend([
            ["RMSE", np.sqrt(mean_squared_error(y, prediction))],
            ["MAE", mean_absolute_error(y, prediction)]
        ])
        if hasattr(model, "deviance"):
            rows.append(["偏差Deviance", model.deviance])
        if hasattr(model, "null_deviance"):
            pseudo_r2 = 1 - (model.deviance / model.null_deviance)
            rows.append(["伪R²", pseudo_r2])

    elif model_type == "二项Logistic回归":
        probability = np.asarray(fitted_result["probability"]).astype(float)
        predicted_class = (probability >= 0.5).astype(int)
        rows.extend([
            ["准确率", accuracy_score(y.astype(int), predicted_class)],
            ["平衡准确率", balanced_accuracy_score(y.astype(int), predicted_class)],
            ["精确率", precision_score(y.astype(int), predicted_class, zero_division=0)],
            ["召回率", recall_score(y.astype(int), predicted_class, zero_division=0)],
            ["F1值", f1_score(y.astype(int), predicted_class, zero_division=0)],
            ["Log Loss", log_loss(y.astype(int), probability)]
        ])
        if len(np.unique(y)) == 2:
            rows.append(["ROC-AUC", roc_auc_score(y, probability)])

    elif model_type == "多项Logistic回归":
        predicted_class = np.asarray(fitted_result["prediction"]).astype(int)
        rows.extend([
            ["准确率", accuracy_score(y.astype(int), predicted_class)],
            ["平衡准确率", balanced_accuracy_score(y.astype(int), predicted_class)],
            ["宏平均F1", f1_score(y.astype(int), predicted_class, average="macro", zero_division=0)]
        ])

    elif model_type in ["线性混合效应模型", "比例型混合效应模型"]:
        rows.extend([
            ["RMSE", np.sqrt(mean_squared_error(y, prediction))],
            ["MAE", mean_absolute_error(y, prediction)]
        ])
        if hasattr(model, "llf"):
            rows.append(["对数似然", model.llf])

    return pd.DataFrame(rows, columns=["指标", "数值"])

def make_diagnostic_table(model, y):
    """生成线性回归诊断指标。"""
    rows = []
    residuals = np.asarray(model.resid).astype(float)
    fitted = np.asarray(model.fittedvalues).astype(float)
    rows.append(["Durbin-Watson", durbin_watson(residuals)])
    if len(residuals) >= 3:
        try:
            shapiro_values = residuals
            if len(shapiro_values) > 5000:
                rng = np.random.default_rng(42)
                index = rng.choice(len(shapiro_values), 5000, replace=False)
                shapiro_values = shapiro_values[index]
            shapiro_stat, shapiro_p = shapiro(shapiro_values)
            rows.append(["残差Shapiro-Wilk P值", shapiro_p])
        except Exception:
            pass
    try:
        bp_result = het_breuschpagan(residuals, model.model.exog)
        rows.extend([["Breusch-Pagan统计量", bp_result[0]], ["Breusch-Pagan P值", bp_result[1]]])
    except Exception:
        pass
    try:
        white_result = het_white(residuals, model.model.exog)
        rows.extend([["White检验统计量", white_result[0]], ["White检验P值", white_result[1]]])
    except Exception:
        pass
    return pd.DataFrame(rows, columns=["诊断指标", "数值"])

def create_prediction_table(fitted_result):
    """生成实际值、预测值和残差表。"""
    result = pd.DataFrame({
        "实际值": np.asarray(fitted_result["display_y"]),
        "预测值": np.asarray(fitted_result["prediction"])
    })
    result["残差"] = result["实际值"] - result["预测值"]
    if "probability" in fitted_result:
        probability = np.asarray(fitted_result["probability"])
        if probability.ndim == 1:
            result["预测概率"] = probability
        result["预测类别"] = np.asarray(fitted_result["prediction"])
    return result

# ============================================================
# 十一、模型假设
# ============================================================
def generate_assumptions(target, predictors, variable_types, model_type, vif_table=None, group_col=None):
    """自动生成论文模型假设。"""
    assumptions = [
        "假设题目所给数据真实可靠，能够反映研究对象的主要特征。",
        "假设各变量的单位、编码方式和统计口径保持一致。",
        "假设缺失值及异常值处理不会改变数据的主要统计规律。"
    ]
    if vif_table is not None and not vif_table.empty:
        finite_vif = vif_table[np.isfinite(vif_table["VIF"])]
        if not finite_vif.empty:
            max_vif = finite_vif["VIF"].max()
            if max_vif < 5:
                assumptions.append("各解释变量VIF均小于5，暂未发现明显的多重共线性。")
            elif max_vif < 10:
                assumptions.append("部分解释变量VIF介于5和10之间，可能存在一定的多重共线性。")
            else:
                assumptions.append("部分解释变量VIF不低于10，存在较严重的多重共线性风险。")

    if model_type == "多元线性回归":
        assumptions.extend([
            "假设因变量与解释变量之间的条件均值关系可以用线性函数近似表示。",
            "假设误差项条件均值为零，且不同观测之间相互独立。",
            "假设误差项具有近似恒定的方差。",
            "假设不存在对估计结果产生决定性影响的极端观测。"
        ])
    elif model_type == "Logit变换线性回归":
        assumptions.extend([
            "假设因变量为0到1之间的比例变量。",
            "对因变量进行Logit变换后，其与解释变量之间近似满足线性关系。",
            "假设变换后的误差项相互独立且方差相对稳定。",
            "模型预测结果通过逆Logit变换还原为比例形式。"
        ])
    elif model_type == "线性混合效应模型":
        assumptions.extend([
            "假设同一分组内的多次观测可能存在相关性。",
            "通过随机截距描述不同分组对象之间的个体差异。",
            "假设随机效应均值为零，并与固定效应结构相互独立。"
        ])
    elif model_type == "比例型混合效应模型":
        assumptions.extend([
            "假设因变量为0到1之间的比例变量。",
            "对因变量进行Logit变换，并通过随机截距处理组内相关性。",
            "模型预测结果通过逆Logit变换还原为比例形式。"
        ])
    elif model_type == "二项Logistic回归":
        assumptions.extend([
            "假设因变量为二分类变量，并编码为0和1。",
            "假设事件发生的对数优势比与解释变量近似线性相关。",
            "假设观测样本之间相互独立。"
        ])
    elif model_type == "多项Logistic回归":
        assumptions.extend([
            "假设因变量为多分类变量。",
            "以一个类别作为参照类别，比较其他类别的发生优势。",
            "假设不同观测样本之间相互独立。"
        ])
    elif model_type == "Poisson回归":
        assumptions.extend([
            "假设因变量为非负整数计数变量。",
            "假设计数变量服从Poisson分布或近似服从Poisson分布。",
            "采用对数连接函数描述解释变量与计数均值的关系。",
            "需要检查数据是否存在过度离散。"
        ])
    elif model_type == "负二项回归":
        assumptions.extend([
            "假设因变量为非负整数计数变量。",
            "假设数据存在超过Poisson分布的额外离散程度。",
            "采用负二项分布和对数连接函数进行建模。"
        ])

    # ✅ 修复：外层改用单引号，避免中文双引号与字符串界定符冲突
    if group_col not in [None, "无"]:
        assumptions.append(f'以"{group_col}"作为重复观测分组变量，同一分组内的观测可能存在相关性。')
    return list(dict.fromkeys(assumptions))

# ============================================================
# 十二、页面侧边栏
# ============================================================
st.title("数学建模大赛前期数据分析工具")
st.caption("数据清洗、变量识别、可视化、相关性分析、模型推荐与统计建模")

with st.sidebar:
    st.header("基本设置")
    st.subheader("赛题描述")
    uploaded_problem_file = st.file_uploader(
        "上传赛题文件（PDF / Word / TXT）",
        type=["pdf", "docx", "txt"],
        key="problem_file_uploader"
    )
    if "problem_text" not in st.session_state:
        st.session_state["problem_text"] = ""
    if uploaded_problem_file is not None:
        extracted = extract_text_from_file(uploaded_problem_file)
        st.session_state["problem_text"] = extracted
    problem_text = st.text_area(
        "粘贴赛题原文（或显示上传文件内容）",
        value=st.session_state.get("problem_text", ""),
        height=160,
        placeholder="上传文件后自动显示内容，也可以直接在此处粘贴。",
        key="problem_text_area"
    )
    st.session_state["problem_text"] = problem_text

    if st.button("自动检测题型"):
        result = multi_label_classify_problem_text(problem_text)
        st.session_state["detect_result"] = result
        st.session_state["problem_type"] = result["main_type"]
        col1, col2, col3 = st.columns(3)
        col1.info(f"主题型：**{result['main_type']}**")
        col2.info(f"检测到多标签：**{', '.join(result['all_detected_labels']) if result['all_detected_labels'] else '无'}**")
        col3.info(f"命中最高得分类别得分：{result['label_scores'].get(result['main_type'], 0)}")
        if result["sub_question_context"]:
            with st.expander("查看自动提取到的赛题各子问题片段"):
                for idx, snippet in enumerate(result["sub_question_context"], 1):
                    st.markdown(f"**子问题 {idx}**：{snippet} ...")
        if result["main_type"] == "未识别":
            st.warning("当前关键词命中不足，你可以在下方手动指定题型。")

    detect_result = st.session_state.get("detect_result", {})
    if detect_result:
        with st.expander("多问题赛题结构化建模建议（自动生成）"):
            st.markdown("### 自动识别的多标签题型：")
            st.dataframe(pd.DataFrame({
                "题型类别": list(detect_result.get("label_scores", {}).keys()),
                "命中关键词数量": list(detect_result.get("label_scores", {}).values())
            }), use_container_width=True)
            st.markdown("### 各题型适合采用的数学建模方向：")
            direction_map = {
                "评价类": "层次分析、TOPSIS、模糊综合评价、灰色关联分析、主成分分析、指标权重设计",
                "预测类": "多元线性回归、时间序列ARIMA、灰色预测GM(1,1)、神经网络预测、逻辑回归",
                "优化类": "线性规划、整数规划、0-1规划、多目标优化、启发式算法、遗传算法、模拟退火",
                "机理分析类": "微分方程建模、动力学模型、Logistic模型、SIR模型、扩散方程、稳定性分析",
                "分类类": "Logistic回归、决策树、随机森林、支持向量机SVM、聚类分析、判别分析"
            }
            for label in detect_result.get("all_detected_labels", []):
                direction = direction_map.get(label, "需要根据赛题背景选择对应方法")
                st.info(f"**{label}**：推荐方向 → {direction}")
            st.success("你可以根据上方提取到的子问题片段逐个子问题选择对应模型。")

    problem_type = st.text_input(
        "赛题类型",
        value=st.session_state.get("problem_type", ""),
        placeholder="例如：预测类、评价类、优化类"
    )

    uploaded_file = st.file_uploader("上传数据表", type=["csv", "xlsx", "xls"])
    st.subheader("缺失值处理")
    missing_method = st.selectbox("处理方式", ["分类型处理", "删除含缺失值的行"])
    st.subheader("异常值处理")
    outlier_method = st.selectbox("识别方法", ["不处理", "3σ", "IQR"])
    outlier_action = st.selectbox("异常值处理动作", ["仅标记，不删除", "删除异常行"])
    robust_se = st.checkbox("线性回归使用HC3稳健标准误", value=True)
    use_test_set = st.checkbox("进行训练集/测试集评估", value=True)
    test_size = st.slider("测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

if uploaded_file is None:
    st.info("请在左侧上传 CSV 或 Excel 数据表。")
    st.stop()

# ============================================================
# 十三、读取数据
# ============================================================
raw_df = None
try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"读取文件失败：{exc}")
    st.stop()

# 当用户重新上传文件或更改关键变量时，清除旧的模型缓存
if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
    for key in ["fitted_result", "fitted_model", "final_model_type", "model_meta", "X_for_assumption", "vif_table"]:
        st.session_state[key] = None
    st.session_state["last_uploaded_file"] = uploaded_file.name

# ============================================================
# 十四、数据读取后的主面板逻辑
# ============================================================
if raw_df is None or raw_df.empty:
    st.error("数据未成功加载，请检查文件格式")
    st.stop()

original_shape = raw_df.shape
raw_df.columns = make_unique_columns(raw_df.columns)
all_columns = list(raw_df.columns)

# ------------------------------------------------------------
# 1. 数据初探
# ------------------------------------------------------------
st.subheader("1. 数据初探")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("原始行数", original_shape[0])
metric_col2.metric("原始列数", original_shape[1])
metric_col3.metric("缺失单元格数", int(raw_df.isna().sum().sum()))
metric_col4.metric("重复行数", int(raw_df.duplicated().sum()))
st.write("数据前20行")
st.dataframe(raw_df.head(20), use_container_width=True)

with st.expander("查看字段基本信息"):
    info_table = pd.DataFrame({
        "列名": raw_df.columns,
        "数据类型": raw_df.dtypes.astype(str).values,
        "非空数量": raw_df.notna().sum().values,
        "缺失数量": raw_df.isna().sum().values,
        "唯一值数量": raw_df.nunique(dropna=True).values,
        "是否疑似ID列": ["是" if is_suspicious_id_column(raw_df[col], col) else "否" for col in raw_df.columns]
    })
    st.dataframe(info_table, use_container_width=True)
    dataframe_download(info_table, "字段基本信息.csv")

# ------------------------------------------------------------
# 2. 变量设置
# ------------------------------------------------------------
st.subheader("2. 因变量、自变量与分组变量设置")
if len(all_columns) < 2:
    st.error("数据表至少需要包含一列因变量和一列自变量。")
    st.stop()

target = st.selectbox("请选择因变量", all_columns, index=0)

default_predictors = []
for col in all_columns:
    if col == target:
        continue
    if is_suspicious_id_column(raw_df[col], col):
        continue
    default_predictors.append(col)

predictors = st.multiselect(
    "请选择自变量",
    [col for col in all_columns if col != target],
    default=default_predictors
)
if not predictors:
    st.warning("请至少选择一个自变量。")
    st.stop()

# 清除因变量/自变量改变时的旧模型结果
if "last_target" not in st.session_state or st.session_state["last_target"] != target:
    for key in ["fitted_result", "fitted_model", "final_model_type"]:
        st.session_state[key] = None
    st.session_state["last_target"] = target
if "last_predictors" not in st.session_state or set(st.session_state.get("last_predictors", [])) != set(predictors):
    for key in ["fitted_result", "fitted_model", "final_model_type"]:
        st.session_state[key] = None
    st.session_state["last_predictors"] = list(predictors)

# 分组变量
group_candidates = ["无"]
for col in all_columns:
    if col == target or col in predictors:
        continue
    non_missing = raw_df[col].dropna()
    if len(non_missing) == 0:
        continue
    group_count = non_missing.nunique()
    group_sizes = non_missing.value_counts()
    if group_count >= 2 and group_sizes.max() >= 2 and group_count < len(raw_df):
        group_candidates.append(col)

group_col = st.selectbox(
    "重复观测分组变量",
    group_candidates,
    help="如果同一对象有多次观测，请选择对应的对象编号。"
)
if group_col != "无" and group_col in predictors:
    st.warning(f"分组变量 `{group_col}` 不会作为普通自变量进入模型，程序将自动从自变量中移除。")
    predictors = [col for col in predictors if col != group_col]
if not predictors:
    st.error("移除分组变量后没有剩余自变量。")
    st.stop()

# ------------------------------------------------------------
# 3. 符号表（样本）
# ------------------------------------------------------------
st.subheader("3. 数学建模符号表")
symbol_mode = st.radio("符号表类型", ["样本符号表", "变量符号表"], horizontal=True)
if symbol_mode == "样本符号表":
    sample_symbol_table = create_sample_symbol_table(raw_df)
    st.info("该表将第一列中的每个样本或对象映射为A1、A2、A3……。")
    st.dataframe(sample_symbol_table, use_container_width=True)
    dataframe_download(sample_symbol_table, "样本符号表.csv")

# ------------------------------------------------------------
# 4. 变量类型识别与确认
# ------------------------------------------------------------
st.subheader("4. 变量类型识别与确认")
initial_variable_types = {}
for col in [target] + predictors:
    initial_variable_types[col] = classify_variable(raw_df[col])
initial_type_table = pd.DataFrame({
    "变量": list(initial_variable_types.keys()),
    "自动识别类型": list(initial_variable_types.values()),
    "原始数据类型": [str(raw_df[col].dtype) for col in initial_variable_types],
    "唯一值数量": [raw_df[col].nunique(dropna=True) for col in initial_variable_types]
})
st.dataframe(initial_type_table, use_container_width=True)
st.caption("自动识别结果仅供参考，正式建模前请根据变量实际含义确认类型。")

variable_types = {}
type_options = ["连续", "分类", "时间", "次数"]
for col in [target] + predictors:
    default_type = initial_variable_types[col]
    if default_type not in type_options:
        default_type = "分类"
    default_index = type_options.index(default_type)
    variable_types[col] = st.selectbox(
        f"确认变量 `{col}` 的类型",
        type_options,
        index=default_index,
        key=f"variable_type_{col}"
    )
# 根据当前数据分析设置判断旧模型是否仍然有效
current_analysis_signature = build_analysis_signature(
    target=target,
    predictors=predictors,
    variable_types=variable_types,
    group_col=group_col,
    missing_method=missing_method,
    outlier_method=outlier_method,
    outlier_action=outlier_action,
    robust_se=robust_se,
    use_test_set=use_test_set,
    test_size=test_size
)

previous_analysis_signature = st.session_state.get(
    "analysis_signature"
)

if previous_analysis_signature != current_analysis_signature:
    clear_model_session_state()
    st.session_state["analysis_signature"] = current_analysis_signature
    
# ✅ 修正：变量符号表移至 variable_types 确认之后
if symbol_mode == "变量符号表":
    variable_symbol_table = create_variable_symbol_table(raw_df, target, predictors, variable_types)
    st.info("变量符号表中的变量含义和单位需要根据赛题背景手动补充。")
    edited_symbol_table = st.data_editor(variable_symbol_table, use_container_width=True, num_rows="fixed")
    dataframe_download(edited_symbol_table, "变量符号表.csv")

# ------------------------------------------------------------
# 5. 数据清洗
# ------------------------------------------------------------
st.subheader("5. 数据清洗")
st.info(
    "因变量缺失的样本无法用于监督建模，因此程序会直接删除；"
    "缺失值填补仅针对自变量。"
)
typed_df = convert_types(raw_df, variable_types)

before_missing_cells = int(typed_df.isna().sum().sum())

# 因变量缺失时不进行插补，直接删除对应样本
target_missing_mask = typed_df[target].isna()
deleted_target_missing_rows = int(target_missing_mask.sum())

typed_df = typed_df.loc[~target_missing_mask].copy()

# 缺失值处理只针对自变量
predictor_variable_types = {
    col: variable_types[col]
    for col in predictors
    if col in variable_types
}

(
    clean_df,
    missing_detail_table,
    deleted_missing_rows,
    imputed_cells
) = fill_missing_values(
    typed_df,
    predictor_variable_types,
    missing_method
)

# 异常值处理
numeric_columns_for_outlier = [
    col for col in [target] + predictors
    if variable_types.get(col) in ["连续", "次数"]
]
outlier_detail_table = pd.DataFrame()
outlier_row_count = 0
deleted_outlier_rows = 0
if outlier_method != "不处理":
    marked_df, outlier_detail_table = detect_outliers(clean_df, numeric_columns_for_outlier, outlier_method)
    outlier_row_count = int(marked_df["_异常行"].sum())
    if outlier_action == "删除异常行":
        deleted_outlier_rows = outlier_row_count
        clean_df = marked_df.loc[~marked_df["_异常行"]].copy()
    else:
        clean_df = marked_df.copy()
else:
    clean_df["_异常行"] = False

# 删除无用列
used_columns = [target] + predictors
if group_col != "无":
    used_columns.append(group_col)
used_columns = list(dict.fromkeys(used_columns))
if len(all_columns) > 0:
    used_columns.append(all_columns[0])
used_columns = list(dict.fromkeys(used_columns))
unused_columns = [col for col in clean_df.columns if (col not in used_columns and col != "_异常行")]
clean_df = clean_df.drop(columns=unused_columns, errors="ignore")

after_missing_cells = int(clean_df.isna().sum().sum())
clean_df = clean_df.reset_index(drop=True)
clean_data_for_model = clean_df.drop(columns=["_异常行"], errors="ignore")

cleaning_summary = pd.DataFrame({
    "项目": ["处理时间", "赛题类型", "原始行数", "清洗后行数", "原始列数", "清洗后列数",
             "原始缺失单元格数", "缺失值插补数量",
             "因变量缺失删除的行数",
             "自变量缺失删除的行数",
             "检测到的异常行数",
             "因异常删除的行数", "清洗后剩余缺失单元格数", "缺失值处理方式", "异常值识别方法",
             "异常值处理动作", "重复观测分组变量", "删除的无用列"],
    "结果": [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        problem_type if problem_type else "未填写",
        original_shape[0], clean_data_for_model.shape[0],
        original_shape[1], clean_data_for_model.shape[1],
        before_missing_cells,
        imputed_cells,
        deleted_target_missing_rows,
        deleted_missing_rows,
        outlier_row_count,
        deleted_outlier_rows,
        after_missing_cells,
        missing_method, outlier_method, outlier_action,
        group_col, ", ".join(unused_columns) if unused_columns else "无"
    ]
})
st.write("清洗结果概览")
clean_metric1, clean_metric2, clean_metric3, clean_metric4 = st.columns(4)
clean_metric1.metric("缺失值插补数量", imputed_cells)
clean_metric2.metric(
    "因缺失删除行数",
    deleted_target_missing_rows + deleted_missing_rows
)
clean_metric3.metric("检测到异常行数", outlier_row_count)
clean_metric4.metric("清洗后样本数", clean_data_for_model.shape[0])

st.dataframe(cleaning_summary, use_container_width=True)
st.dataframe(missing_detail_table, use_container_width=True)
if not outlier_detail_table.empty:
    st.dataframe(outlier_detail_table, use_container_width=True)
st.dataframe(clean_data_for_model.head(20), use_container_width=True)

dataframe_download(cleaning_summary, "清洗汇总报告.csv")
dataframe_download(missing_detail_table, "缺失值处理明细.csv")
if not outlier_detail_table.empty:
    dataframe_download(outlier_detail_table, "异常值处理明细.csv")
dataframe_download(clean_data_for_model, "清洗后数据.csv")

st.text_area("可直接用于论文的数据清洗表述", f"""
本文首先对原始数据进行完整性、一致性和变量类型检查。
原始数据共包含{original_shape[0]}条样本和{original_shape[1]}个变量。
针对缺失数据，本文采用"{missing_method}"方法进行处理，
共插补{imputed_cells}个缺失单元格。
对于因变量缺失的样本，由于无法提供有效的被解释变量观测值，
直接删除{deleted_target_missing_rows}条样本；
另外因自变量缺失处理删除{deleted_missing_rows}条样本。
针对异常数据，本文采用"{outlier_method}"方法进行识别，
共检测到{outlier_row_count}条异常样本，其中因异常值删除{deleted_outlier_rows}条样本。
经过数据类型转换、缺失值处理、异常值处理和无用列处理后，
最终获得{clean_data_for_model.shape[0]}条有效样本，用于后续统计分析和模型建立。
""".strip(), height=220)

# ------------------------------------------------------------
# 6. 数据可视化
# ------------------------------------------------------------
st.subheader("6. 数据可视化")
numeric_variables = [
    col for col in [target] + predictors
    if variable_types.get(col) in ["连续", "次数"] and col in clean_data_for_model.columns
]
categorical_variables = [
    col for col in [target] + predictors
    if variable_types.get(col) == "分类" and col in clean_data_for_model.columns
]
chart_type = st.selectbox("选择图形类型", [
    "变量分布图", "变量箱线图", "因变量-自变量散点图", "因变量-自变量曲线图",
    "数值变量相关性热力图", "分类变量频数图"
])

if chart_type == "变量分布图":
    if not numeric_variables:
        st.info("当前没有连续型或次数型变量。")
    else:
        selected_col = st.selectbox("选择变量", numeric_variables)
        plot_values = pd.to_numeric(clean_data_for_model[selected_col], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(plot_values, kde=True, ax=ax)
        ax.set_title(f"{selected_col} 分布图")
        ax.set_xlabel(selected_col)
        st.pyplot(fig)
        plt.close(fig)

elif chart_type == "变量箱线图":
    if not numeric_variables:
        st.info("当前没有连续型或次数型变量。")
    else:
        selected_columns = st.multiselect("选择变量", numeric_variables, default=numeric_variables)
        if selected_columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=clean_data_for_model[selected_columns], ax=ax)
            ax.tick_params(axis="x", rotation=35)
            ax.set_title("变量箱线图")
            st.pyplot(fig)
            plt.close(fig)

elif chart_type == "因变量-自变量散点图":
    x_candidates = [col for col in numeric_variables if col != target]
    if not x_candidates:
        st.info("没有适合绘制散点图的数值型自变量。")
    else:
        x_col = st.selectbox("选择横轴自变量", x_candidates)
        plot_df = clean_data_for_model[[x_col, target]].copy()
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
        plot_df[target] = pd.to_numeric(plot_df[target], errors="coerce")
        plot_df = plot_df.dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=plot_df, x=x_col, y=target, ax=ax)
        ax.set_title(f"{target} 与 {x_col} 的散点图")
        st.pyplot(fig)
        plt.close(fig)

elif chart_type == "因变量-自变量曲线图":
    x_candidates = [col for col in numeric_variables if col != target]
    if not x_candidates:
        st.info("没有适合绘制曲线图的数值型自变量。")
    else:
        x_col = st.selectbox("选择横轴自变量", x_candidates)
        plot_df = clean_data_for_model[[x_col, target]].copy()
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
        plot_df[target] = pd.to_numeric(plot_df[target], errors="coerce")
        plot_df = plot_df.dropna().sort_values(x_col)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(plot_df[x_col], plot_df[target], marker="o", linewidth=1.5)
        ax.set_xlabel(x_col); ax.set_ylabel(target)
        ax.set_title(f"{target} 与 {x_col} 的变化曲线")
        st.pyplot(fig)
        plt.close(fig)

elif chart_type == "数值变量相关性热力图":
    if len(numeric_variables) < 2:
        st.info("至少需要两个数值型变量。")
    else:
        corr = clean_data_for_model[numeric_variables].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("数值变量Pearson相关性热力图")
        st.pyplot(fig)
        plt.close(fig)

elif chart_type == "分类变量频数图":
    if not categorical_variables:
        st.info("当前没有分类变量。")
    else:
        selected_col = st.selectbox("选择分类变量", categorical_variables)
        count_table = clean_data_for_model[selected_col].astype("string").value_counts(dropna=False).rename_axis(selected_col).reset_index(name="数量")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=count_table, x=selected_col, y="数量", ax=ax)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(f"{selected_col} 频数图")
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------
# 7. 相关性分析
# ------------------------------------------------------------
st.subheader("7. 相关性分析")
corr_table = correlation_table(clean_data_for_model, target, predictors, variable_types)
if corr_table.empty:
    st.info("没有足够的连续型或次数型变量用于相关性分析。")
else:
    st.dataframe(corr_table, use_container_width=True)
    dataframe_download(corr_table, "相关性分析.csv")

# ------------------------------------------------------------
# 8. 自动模型推荐与人工确认
# ------------------------------------------------------------
st.subheader("8. 自动模型推荐与人工确认")
fitted_result = None
fitted_model = None
model_info = None
X_for_assumption = None
vif_table = pd.DataFrame()
y = None
X = None
groups = None
try:
    y, X, groups, model_meta = build_model_data(
        clean_data_for_model, target, predictors, variable_types, group_col=group_col
    )
    X_for_assumption = X
    if len(y) <= X.shape[1]:
        st.error("有效样本数不大于模型参数数量，无法稳定建立模型。")
        st.stop()
    matrix_rank = np.linalg.matrix_rank(X.values)
    if matrix_rank < X.shape[1]:
        st.warning("设计矩阵可能存在完全多重共线性，部分参数可能无法稳定估计。")

    model_info = detect_model_type(y, variable_types[target], groups=groups)
    recommended_model = model_info["model_type"]
    st.info(f"程序推荐模型：**{recommended_model}**\n\n判断依据：{model_info['reason']}")

    model_options = [
        "多元线性回归", "Logit变换线性回归", "线性混合效应模型", "比例型混合效应模型",
        "二项Logistic回归", "多项Logistic回归", "Poisson回归", "负二项回归"
    ]
    recommended_index = model_options.index(recommended_model) if recommended_model in model_options else 0
    final_model_type = st.selectbox("请选择最终拟合模型", model_options, index=recommended_index)
    # 如果用户更换了最终模型类型，清除之前的模型结果
    previous_model_type = st.session_state.get(
        "selected_model_type"
    )

    if previous_model_type != final_model_type:
        clear_model_session_state()
        st.session_state["selected_model_type"] = final_model_type
    if final_model_type in ["线性混合效应模型", "比例型混合效应模型"] and group_col == "无":
        st.error("当前模型需要分组变量，请先在上方选择重复观测分组变量。")
        st.stop()

    vif_table = calculate_vif(X)
    st.write("多重共线性诊断")
    if vif_table.empty:
        st.info("当前没有足够的自变量计算VIF。")
    else:
        st.dataframe(vif_table, use_container_width=True)
        dataframe_download(vif_table, "VIF多重共线性诊断.csv")

    fit_button = st.button("开始拟合最终模型", type="primary")

    if fit_button:

        # 在真正拟合前检查模型是否适用于当前因变量
        is_valid, validation_message = validate_model_selection(
            y=y,
            target_type=variable_types[target],
            model_type=final_model_type,
            groups=groups
        )

        if not is_valid:
            st.error(validation_message)
            st.stop()

        # 检查通过后，才进行模型拟合
        fitted_result = fit_model(
            y,
            X,
            groups,
            final_model_type,
            robust_se=robust_se
        )

        fitted_model = fitted_result["model"]

        if model_is_converged(fitted_model):
            st.success(
                "模型拟合完成，模型已收敛或未检测到明显收敛问题。"
            )
        else:
            st.warning(
                "模型未收敛，当前系数、P值和预测结果不建议直接用于论文。"
            )

        st.session_state["fitted_result"] = fitted_result
        st.session_state["fitted_model"] = fitted_model
        st.session_state["final_model_type"] = final_model_type
        st.session_state["model_meta"] = model_meta
        st.session_state["X_for_assumption"] = X
        st.session_state["vif_table"] = vif_table

    prediction_table = create_prediction_table(fitted_result)
    st.write("实际值、预测值与残差")
    st.dataframe(prediction_table.head(100), use_container_width=True)
    dataframe_download(prediction_table, "实际值预测值残差.csv")

    # 训练集/测试集评估
    st.write("训练集/测试集评估")
    if use_test_set and model_type not in ["线性混合效应模型", "比例型混合效应模型"]:
        if y is None or X is None:
            st.warning("模型数据未能正确构造，无法进行训练集/测试集评估。")
        else:
            try:
                if len(y) < 10:
                    st.warning("样本量少于10，训练集/测试集划分结果可能不稳定。")
                indices = np.arange(len(y))
                train_index, test_index = train_test_split(indices, test_size=test_size, random_state=42)
                y_train = y.iloc[train_index]; y_test = y.iloc[test_index]
                X_train = X.iloc[train_index]; X_test = X.iloc[test_index]

                test_model_result = fit_model(y_train, X_train, None, model_type, robust_se=robust_se)
                test_model = test_model_result["model"]
                test_prediction = np.asarray(test_model.predict(X_test))

                if model_type == "二项Logistic回归":
                    test_probability = test_prediction
                    test_class = (test_probability >= 0.5).astype(int)
                    test_metric_table = pd.DataFrame({
                        "测试集指标": ["准确率", "平衡准确率", "精确率", "召回率", "F1值", "Log Loss", "ROC-AUC"],
                        "数值": [
                            accuracy_score(y_test.astype(int), test_class),
                            balanced_accuracy_score(y_test.astype(int), test_class),
                            precision_score(y_test.astype(int), test_class, zero_division=0),
                            recall_score(y_test.astype(int), test_class, zero_division=0),
                            f1_score(y_test.astype(int), test_class, zero_division=0),
                            log_loss(y_test.astype(int), test_probability),
                            roc_auc_score(y_test.astype(int), test_probability)
                        ]
                    })
                elif model_type == "多项Logistic回归":
                    test_probability = np.asarray(test_model.predict(X_test))
                    test_class = test_probability.argmax(axis=1)
                    test_metric_table = pd.DataFrame({
                        "测试集指标": ["准确率", "平衡准确率", "宏平均F1"],
                        "数值": [
                            accuracy_score(y_test.astype(int), test_class),
                            balanced_accuracy_score(y_test.astype(int), test_class),
                            f1_score(y_test.astype(int), test_class, average="macro", zero_division=0)
                        ]
                    })
                else:
                    test_metric_table = pd.DataFrame({
                        "测试集指标": ["RMSE", "MAE", "R²"],
                        "数值": [
                            np.sqrt(mean_squared_error(y_test, test_prediction)),
                            mean_absolute_error(y_test, test_prediction),
                            r2_score(y_test, test_prediction)
                        ]
                    })
                st.dataframe(test_metric_table, use_container_width=True)
                dataframe_download(test_metric_table, "测试集评价指标.csv")
            except Exception as exc:
                st.warning(f"测试集评估失败：{exc}")
    elif not use_test_set:
        st.info("当前已关闭训练集/测试集评估。上方模型评价指标为样本内指标。")
    else:
        st.info("混合效应模型暂未自动进行普通随机划分，建议按照分组变量进行分组交叉验证。")

    # 线性模型诊断
    if model_type in ["多元线性回归", "Logit变换线性回归"]:
        st.subheader("线性模型诊断")
        diagnostic_table = make_diagnostic_table(fitted_model, fitted_result["display_y"])
        st.dataframe(diagnostic_table, use_container_width=True)
        dataframe_download(diagnostic_table, "线性模型诊断.csv")
        residuals = np.asarray(fitted_model.resid).astype(float)
        fitted_values = np.asarray(fitted_model.fittedvalues).astype(float)
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            fig_residual, ax_residual = plt.subplots(figsize=(7, 5))
            sns.scatterplot(x=fitted_values, y=residuals, ax=ax_residual)
            ax_residual.axhline(0, color="red", linestyle="--")
            ax_residual.set_xlabel("拟合值"); ax_residual.set_ylabel("残差")
            ax_residual.set_title("残差-拟合值图")
            st.pyplot(fig_residual); plt.close(fig_residual)
        with diag_col2:
            fig_qq, ax_qq = plt.subplots(figsize=(7, 5))
            probplot(residuals, dist="norm", plot=ax_qq)
            ax_qq.set_title("残差正态QQ图")
            st.pyplot(fig_qq); plt.close(fig_qq)

        try:
            influence = OLSInfluence(fitted_model)
            cooks_distance = influence.cooks_distance[0]
            cooks_table = pd.DataFrame({
                "样本序号": np.arange(len(cooks_distance)),
                "Cook距离": cooks_distance
            }).sort_values("Cook距离", ascending=False)
            st.write("Cook距离较大的观测")
            st.dataframe(cooks_table.head(20), use_container_width=True)
            dataframe_download(cooks_table, "Cook距离.csv")
        except Exception as exc:
            st.info(f"Cook距离计算失败：{exc}")

except Exception as exc:
    st.warning(f"当前模型无法生成标准摘要：{exc}")

    # 分类模型诊断
    if model_type == "二项Logistic回归":
        st.subheader("二分类模型诊断")
        cm = confusion_matrix(
            prediction_table["实际值"].astype(int),
            prediction_table["预测类别"].astype(int)
        )
        cm_table = pd.DataFrame(cm, index=["真实0", "真实1"], columns=["预测0", "预测1"])
        st.write("混淆矩阵")
        st.dataframe(cm_table, use_container_width=True)
        dataframe_download(cm_table.reset_index(), "二分类混淆矩阵.csv")

    # 模型假设
    st.subheader("10. 模型假设")
    assumptions = generate_assumptions(target, predictors, variable_types, model_type, vif_table=vif_table, group_col=group_col)
    assumptions_text = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(assumptions)])
    st.text_area("可直接复制到论文的模型假设", assumptions_text, height=350)

    # 论文表述
    st.subheader("11. 论文表述草稿")
    paper_text = f"""
本文首先对原始数据进行完整性、一致性和变量类型检查。
原始数据包含{original_shape[0]}条样本和{original_shape[1]}个变量。
针对缺失数据，本文采用"{missing_method}"方法进行处理，
共插补{imputed_cells}个缺失单元格，并因缺失值删除{deleted_missing_rows}条样本。
针对异常数据，本文采用"{outlier_method}"方法进行识别。
共检测到{outlier_row_count}条异常样本，并根据处理策略删除{deleted_outlier_rows}条异常样本。
本文将"{target}"作为因变量，将{", ".join(predictors)}作为解释变量。
"""
    if group_col != "无":
        paper_text += f"""考虑到同一"{group_col}"下可能存在多次观测，本文将其作为重复观测的分组变量。"""
    paper_text += f"""在综合考虑因变量类型、数据取值范围和观测结构后，最终采用"{model_type}"进行建模。清洗后最终保留{clean_data_for_model.shape[0]}条有效样本，并据此开展后续统计分析和模型估计。"""
    st.text_area("论文表述草稿", paper_text.strip(), height=300)

else:
    st.info('请点击"开始拟合最终模型"后查看模型结果。')

# ------------------------------------------------------------
# 12. 使用说明
# ------------------------------------------------------------
with st.expander("使用和解释模型时的注意事项"):
    st.markdown("""
### 1. 自动推荐不是最终结论
程序根据变量类型和数据结构进行初步推荐，最终模型应结合题目目标、变量含义和模型诊断结果确定。

### 2. 相关性不等于因果关系
Pearson/Spearman 相关系数和回归系数只能反映统计关联，不能单独证明因果关系。

### 3. 异常值不一定是错误数据
异常值可能是真实存在的极端情况。正式论文中删除异常数据时，应说明识别方法和删除依据。

### 4. R² 不能单独判断模型好坏
应结合调整 R²、AIC/BIC、残差图、测试集误差、显著性检验和实际解释意义共同评价模型。

### 5. 分类模型不要只看准确率
如果类别分布不平衡，还应重点查看平衡准确率、精确率、召回率、F1 值和 ROC-AUC。

### 6. 混合效应模型需要真实的重复观测结构
不能仅因为某列取值重复就直接使用混合效应模型。分组变量应确实代表同一对象、地区、企业或其他层级单位的重复观测。
""")
