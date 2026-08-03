from pathlib import Path
import io
import base64
import os
import re
import warnings
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
import statsmodels.api as sm

from scipy.stats import pearsonr, spearmanr, shapiro, probplot
from sklearn.model_selection import train_test_split
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
    confusion_matrix,
)
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
    OLSInfluence,
)
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
)
from statsmodels.stats.stattools import durbin_watson

import jieba
import PyPDF2
import docx


# ============================================================
# 一、页面和全局设置
# ============================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="数学建模前期数据分析工具",
    layout="wide",
)

# ===== 页面背景图片 =====
import base64
from pathlib import Path

_bg_path = Path(__file__).resolve().parent / "assets" / "background.png"

if _bg_path.exists():
    _bg_base64 = base64.b64encode(
        _bg_path.read_bytes()
    ).decode("ascii")

    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            background: transparent !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background: transparent !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        #custom-background-image {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            object-fit: cover;
            z-index: -1;
            pointer-events: none;
        }}

        [data-testid="stMain"] {{
            background: rgba(255, 255, 255, 0.10) !important;
        }}
        </style>

        <img
            id="custom-background-image"
            src="data:image/png;base64,{_bg_base64}"
        />
        """,
        unsafe_allow_html=True,
    )
else:
    st.error(f"找不到背景图片：{_bg_path}")

# ===== 页面背景图片结束 =====


# ===== 页面标题图片路径 =====
_fursona_file = Path(__file__).parent / "assets" / "fursona_shu.png"
# ===== 页面标题图片路径结束 =====

# ===== 页面标题 =====

st.markdown(
    """
    <style>
    /* 标题图片和文字垂直居中 */
    div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }

    /* 标题文字 */
    .old-main-title {
        color: #26354a;
        font-size: 72px;
        font-weight: 800;
        line-height: 220px;
        white-space: nowrap;
        letter-spacing: 1px;
        margin: 0;
        padding: 0;
    }

    /* 标题区域列的上下空白 */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /*
    标题图片位置微调：
    正值向右，负值向左；
    正值向下，负值向上。
    */
    div[data-testid="stHorizontalBlock"]:has(.old-main-title)
    > div[data-testid="column"]:first-child
    [data-testid="stImage"] {
        transform: translate(28px, -5px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 标题区域
_title_col_image, _title_col_text = st.columns(
    [1.5, 8.5],
    gap="small",
)

with _title_col_image:
    if _fursona_file.exists():
        st.image(str(_fursona_file), width=220)
    else:
        st.error(f"找不到图片：{_fursona_file}")

with _title_col_text:
    st.markdown(
        """
        <div class="old-main-title">
            学建模前期数据分析工具
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== 页面标题结束 =====








sns.set_theme(style="whitegrid")

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 二、字体设置
# ============================================================

def setup_chinese_font():
    """
    尝试配置中文字体。
    如果网络字体下载失败，也不会影响主程序运行。
    """
    font_candidates = [
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    available_fonts = {
        font.name for font in fm.fontManager.ttflist
    }

    selected_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            selected_font = font_name
            break

    if selected_font is not None:
        plt.rcParams["font.sans-serif"] = [selected_font]
        return

    # 尝试下载 Noto Sans CJK 中文字体
    try:
        font_dir = os.path.expanduser("~/.cache/matplotlib/fonts")
        os.makedirs(font_dir, exist_ok=True)

        font_path = os.path.join(
            font_dir,
            "NotoSansCJKsc-Regular.otf",
        )

        if not os.path.exists(font_path):
            url = (
                "https://github.com/GoogleFonts/"
                "noto-cjk/raw/main/Sans/OTF/"
                "SimplifiedChinese/NotoSansCJKsc-Regular.otf"
            )
            urllib.request.urlretrieve(url, font_path)

        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams["font.sans-serif"] = [
                font_prop.get_name(),
                "DejaVu Sans",
            ]
            return

    except Exception:
        pass

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


setup_chinese_font()
def load_image_as_base64(image_path):
    """读取本地图片并转换为 Base64，供网页嵌入。"""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return encoded

    except Exception:
        return ""
fursona_path = str(Path(__file__).resolve().parent / "assets" / "fursona_shu.png")
fursona_base64 = load_image_as_base64(fursona_path)

# ============================================================
# 三、Session State 管理
# ============================================================

MODEL_STATE_KEYS = [
    "fitted_result",
    "fitted_model",
    "final_model_type",
    "model_meta",
    "X_for_assumption",
    "vif_table",
    "target_mapping",
    "model_signature",
]


def clear_model_session_state():
    """清除旧模型结果。"""
    for key in MODEL_STATE_KEYS:
        st.session_state.pop(key, None)


def reset_model_if_signature_changed(signature):
    """
    如果数据、变量、模型设置发生变化，则清除旧模型结果。
    """
    previous_signature = st.session_state.get("analysis_signature")

    if previous_signature != signature:
        clear_model_session_state()
        st.session_state["analysis_signature"] = signature


def build_analysis_signature(
    file_name,
    target,
    predictors,
    variable_types,
    group_col,
    missing_method,
    outlier_method,
    outlier_action,
    robust_se,
    use_test_set,
    test_size,
):
    """根据当前分析设置生成唯一签名。"""
    signature_data = {
        "file_name": file_name,
        "target": target,
        "predictors": sorted(predictors),
        "variable_types": variable_types,
        "group_col": group_col,
        "missing_method": missing_method,
        "outlier_method": outlier_method,
        "outlier_action": outlier_action,
        "robust_se": robust_se,
        "use_test_set": use_test_set,
        "test_size": test_size,
    }

    return repr(signature_data)


# ============================================================
# 四、通用工具函数
# ============================================================

def clean_column_name(name):
    """规范列名。"""
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\u4e00-\u9fff]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    if not name:
        return "未命名变量"

    return name


def make_unique_columns(columns):
    """清理并处理重复列名。"""
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


def dataframe_download(df, filename, key=None):
    """生成 CSV 下载按钮。"""
    if df is None:
        return

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    data = df.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")

    download_key = key or f"download_{filename}"

    st.download_button(
        label=f"下载 {filename}",
        data=data,
        file_name=filename,
        mime="text/csv",
        key=download_key,
    )


def safe_numeric(series):
    """安全转换为数值。"""
    return pd.to_numeric(series, errors="coerce")


def safe_float(value):
    """安全转换为浮点数。"""
    try:
        return float(value)
    except Exception:
        return np.nan


def model_is_converged(model):
    """检查模型是否收敛。"""
    if model is None:
        return False

    if hasattr(model, "converged"):
        return bool(model.converged)

    if hasattr(model, "mle_retvals"):
        return bool(
            model.mle_retvals.get(
                "converged",
                True,
            )
        )

    return True


# ============================================================
# 五、赛题文件读取和题型识别
# ============================================================

def extract_text_from_file(uploaded_file):
    """提取 PDF、Word、TXT 文件中的文本。"""
    if uploaded_file is None:
        return ""

    file_type = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_type == "pdf":
            text_parts = []
            reader = PyPDF2.PdfReader(uploaded_file)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            return "\n".join(text_parts).strip()

        if file_type == "docx":
            document = docx.Document(uploaded_file)
            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
            ]
            return "\n".join(paragraphs).strip()

        if file_type == "txt":
            raw = uploaded_file.read()

            for encoding in ["utf-8", "gbk", "gb18030"]:
                try:
                    return raw.decode(encoding)
                except UnicodeDecodeError:
                    continue

            return raw.decode("utf-8", errors="ignore")

        return ""

    except Exception as exc:
        return f"[文件解析失败：{exc}]"


def clean_problem_text(text):
    """清洗赛题文本。"""
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def classify_by_tfidf(problem_text):
    """使用 TF-IDF 和余弦相似度识别赛题类型。"""
    if not isinstance(problem_text, str) or not problem_text.strip():
        return {}

    corpus = {
        "评价类": (
            "层次分析法 AHP 模糊综合评价 TOPSIS 熵权法 "
            "灰色关联度 数据包络分析 优劣解距离法 指标权重 "
            "指标体系 综合得分 排序 评估 绩效考核 可行性研究 "
            "多属性决策 主成分分析 因子分析 满意度评价"
        ),
        "预测类": (
            "时间序列 ARIMA 指数平滑 灰色预测 GM(1,1) 回归分析 "
            "趋势外推 神经网络 深度学习 LSTM 支持向量回归 SVR "
            "随机森林回归 XGBoost 增长率 预测值 未来走势 "
            "短期预测 中长期预测 预报 拟合 插值 变化规律"
        ),
        "优化类": (
            "线性规划 整数规划 0-1规划 非线性规划 动态规划 "
            "多目标优化 遗传算法 粒子群算法 模拟退火 蚁群算法 "
            "最短路径 最大流 最小生成树 网络流 调度 排班 指派 "
            "运输问题 库存优化 资源分配 成本最小化 利润最大化 "
            "约束条件 最优解 方案优选"
        ),
        "机理分析类": (
            "微分方程 偏微分方程 常微分方程 动力学模型 "
            "传染病模型 SIR SEIR Logistic方程 捕食者猎物模型 "
            "种群增长 扩散方程 反应扩散 对流扩散 牛顿力学 "
            "流体力学 传热 电磁场 化学反应动力学 稳定平衡 "
            "分岔 相图 数值解 欧拉法 龙格库塔法 有限元"
        ),
        "分类类": (
            "逻辑回归 Logistic回归 支持向量机 SVM 决策树 "
            "随机森林 K近邻 KNN 朴素贝叶斯 神经网络 聚类分析 "
            "K-means 层次聚类 二分类 多分类 混淆矩阵 准确率 "
            "精确率 召回率 F1值 ROC曲线 AUC 异常检测 模式识别 "
            "图像识别 文本分类 诊断 判定 识别 筛选 检出"
        ),
    }

    categories = list(corpus.keys())
    documents = [problem_text] + [
        corpus[category]
        for category in categories
    ]

    def tokenize(text):
        return " ".join(jieba.cut(text))

    try:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
        )
        matrix = vectorizer.fit_transform(documents)
        similarities = cosine_similarity(
            matrix[0:1],
            matrix[1:],
        ).flatten()

    except Exception:
        return {}

    return {
        category: float(similarities[index])
        for index, category in enumerate(categories)
    }


def multi_label_classify_problem_text(problem_text):
    """融合关键词和 TF-IDF 的多标签题型识别。"""
    text = clean_problem_text(problem_text)

    if not text:
        return {
            "main_type": "未识别",
            "all_detected_labels": [],
            "label_scores": {},
            "sub_question_context": [],
        }

    keyword_sets = {
        "评价类": [
            "评价",
            "排序",
            "打分",
            "评选",
            "满意度",
            "权重",
            "层次分析",
            "topsis",
            "优劣",
            "综合评价",
            "评估",
            "排名",
            "指标体系",
        ],
        "预测类": [
            "预测",
            "趋势",
            "未来",
            "增长",
            "回归",
            "时间序列",
            "估计",
            "预报",
            "拟合",
            "外推",
            "增长率",
            "变化规律",
            "走势",
        ],
        "优化类": [
            "最大",
            "最小",
            "最优",
            "成本",
            "利润",
            "资源",
            "调度",
            "运输",
            "规划",
            "约束",
            "路径",
            "分配",
            "库存",
            "0-1",
            "整数",
            "安排",
        ],
        "机理分析类": [
            "微分方程",
            "变化率",
            "传染病",
            "扩散",
            "物理",
            "力学",
            "温度",
            "浓度",
            "平衡点",
            "稳定性",
            "logistic",
            "sir",
            "动力学",
            "传播",
            "运动方程",
            "相互作用",
        ],
        "分类类": [
            "分类",
            "判别",
            "识别",
            "判定",
            "判断",
            "异常检测",
            "诊断",
            "聚类",
            "模式识别",
            "区分",
            "类别归属",
        ],
    }

    keyword_scores = {}

    for label, keywords in keyword_sets.items():
        keyword_scores[label] = sum(
            1
            for keyword in keywords
            if keyword in text
        )

    tfidf_scores = classify_by_tfidf(text)
    combined_scores = {}

    for label in keyword_sets:
        keyword_score = min(
            keyword_scores.get(label, 0),
            10,
        ) / 10.0

        tfidf_score = tfidf_scores.get(label, 0.0)

        combined_scores[label] = (
            0.3 * keyword_score
            + 0.7 * tfidf_score
        )

    detected_labels = [
        label
        for label, score in combined_scores.items()
        if score > 0.1
    ]

    detected_labels.sort(
        key=lambda label: combined_scores[label],
        reverse=True,
    )

    main_type = (
        detected_labels[0]
        if detected_labels
        else "未识别"
    )

    patterns = [
        r"问题\s*[一二三四五六七八九十0-9]+[、\s.]",
        r"\d+[\)）、.]",
    ]

    contexts = []

    for pattern in patterns:
        for match in re.finditer(pattern, problem_text):
            start = match.start()
            end = min(
                start + 250,
                len(problem_text),
            )
            contexts.append(
                problem_text[start:end].strip()
            )

    contexts = list(dict.fromkeys(contexts))[:10]

    return {
        "main_type": main_type,
        "all_detected_labels": detected_labels,
        "label_scores": combined_scores,
        "sub_question_context": contexts,
    }


# ============================================================
# 六、变量识别和数据处理
# ============================================================

def try_parse_datetime(series):
    """尝试判断变量是否为日期时间变量。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    if (
        series.dtype == "object"
        or pd.api.types.is_string_dtype(series)
    ):
        parsed = pd.to_datetime(
            series,
            errors="coerce",
        utc=True,
        )

        if parsed.notna().mean() >= 0.8:
            return parsed

    return None


def classify_variable(series):
    """自动识别变量类型。"""
    parsed_time = try_parse_datetime(series)

    if parsed_time is not None:
        return "时间"

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_rate = numeric.notna().mean()

    if numeric_rate < 0.8:
        return "分类"

    values = numeric.dropna()

    if len(values) == 0:
        return "连续"

    unique_values = set(values.unique().tolist())
    unique_count = values.nunique()

    if (
        unique_count <= 2
        and unique_values.issubset({0, 1})
    ):
        return "分类"

    is_integer = np.allclose(
        values,
        np.round(values),
    )

    is_nonnegative = (values >= 0).all()

    if (
        is_integer
        and is_nonnegative
        and unique_count <= 20
        and values.max() >= 5
    ):
        return "次数"

    return "连续"


def is_suspicious_id_column(series, name):
    """判断某列是否疑似 ID、编号或序号。"""
    name_text = str(name).lower()

    keywords = [
        "id",
        "编号",
        "序号",
        "代码",
        "编码",
        "样本号",
        "学生号",
        "患者号",
        "姓名",
        "name",
    ]

    keyword_flag = any(
        keyword in name_text
        for keyword in keywords
    )

    unique_rate = (
        series.nunique(dropna=True)
        / max(len(series), 1)
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    sequential_flag = False

    if numeric.notna().mean() >= 0.95:
        values = (
            numeric.dropna()
            .sort_values()
            .to_numpy()
        )

        if len(values) >= 3:
            differences = np.diff(values)
            sequential_flag = np.allclose(
                differences,
                differences[0],
            )

    return keyword_flag or (
        unique_rate >= 0.95
        and sequential_flag
    )


def convert_types(df, variable_types):
    """按照用户确认的变量类型转换数据。"""
    result = df.copy()

    for col, var_type in variable_types.items():
        if col not in result.columns:
            continue

        if var_type in ["连续", "次数"]:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce",
            )

        elif var_type == "时间":
            result[col] = pd.to_datetime(
                result[col],
                errors="coerce",
            )

        elif var_type == "分类":
            result[col] = result[col].astype("string")

    return result


def fill_missing_values(
    df,
    variable_types,
    method,
):
    """
    处理自变量缺失值。

    因变量缺失值在调用本函数前删除。
    """
    result = df.copy()
    report_rows = []
    deleted_rows = 0
    imputed_cells = 0

    if not variable_types:
        return (
            result,
            pd.DataFrame(),
            0,
            0,
        )

    if method == "删除含缺失值的行":
        missing_mask = result[
            list(variable_types.keys())
        ].isna().any(axis=1)

        deleted_rows = int(missing_mask.sum())
        result = result.loc[~missing_mask].copy()

        for col in variable_types:
            before = int(df[col].isna().sum())

            report_rows.append(
                {
                    "变量": col,
                    "原始缺失数": before,
                    "处理方式": "删除所在行",
                    "实际插补数": 0,
                    "剩余缺失数": int(
                        result[col].isna().sum()
                    ),
                }
            )

        return (
            result,
            pd.DataFrame(report_rows),
            deleted_rows,
            0,
        )

    for col, var_type in variable_types.items():
        if col not in result.columns:
            continue

        before = int(result[col].isna().sum())

        if before == 0:
            report_rows.append(
                {
                    "变量": col,
                    "原始缺失数": 0,
                    "处理方式": "无缺失",
                    "实际插补数": 0,
                    "剩余缺失数": 0,
                }
            )
            continue

        if var_type in ["连续", "次数"]:
            result[col] = result[col].interpolate(
                method="linear",
                limit_direction="both",
            )

            median_value = result[col].median()

            if pd.notna(median_value):
                result[col] = result[col].fillna(
                    median_value
                )

            treatment = "线性插值+中位数"

        elif var_type == "时间":
            result[col] = result[col].ffill().bfill()
            treatment = "前向填充+后向填充"

        else:
            mode = result[col].mode(dropna=True)

            if len(mode) > 0:
                result[col] = result[col].fillna(
                    mode.iloc[0]
                )

            treatment = "众数填补"

        after = int(result[col].isna().sum())
        actual_imputed = max(0, before - after)
        imputed_cells += actual_imputed

        report_rows.append(
            {
                "变量": col,
                "原始缺失数": before,
                "处理方式": treatment,
                "实际插补数": actual_imputed,
                "剩余缺失数": after,
            }
        )

    return (
        result,
        pd.DataFrame(report_rows),
        deleted_rows,
        imputed_cells,
    )


def detect_outliers(
    df,
    numeric_columns,
    method,
):
    """检测异常值并返回逐行异常标记。"""
    result = df.copy()
    result["_异常行"] = False
    report_rows = []

    for col in numeric_columns:
        values = pd.to_numeric(
            result[col],
            errors="coerce",
        )

        if method == "3σ":
            mean_value = values.mean()
            std_value = values.std()

            if (
                pd.isna(std_value)
                or std_value == 0
            ):
                mask = pd.Series(
                    False,
                    index=result.index,
                )
            else:
                mask = ~values.between(
                    mean_value - 3 * std_value,
                    mean_value + 3 * std_value,
                )

        elif method == "IQR":
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                mask = pd.Series(
                    False,
                    index=result.index,
                )
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                mask = ~values.between(
                    lower,
                    upper,
                )

        else:
            mask = pd.Series(
                False,
                index=result.index,
            )

        mask = mask.fillna(False)
        result["_异常行"] |= mask

        report_rows.append(
            {
                "变量": col,
                "异常值数量": int(mask.sum()),
            }
        )

    return (
        result,
        pd.DataFrame(report_rows),
    )


# ============================================================
# 七、相关性分析和符号表
# ============================================================

def correlation_table(
    df,
    target,
    predictors,
    variable_types,
):
    """计算 Pearson 和 Spearman 相关系数。"""
    rows = []

    if target not in df.columns:
        return pd.DataFrame()

    target_values = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    for col in predictors:
        if variable_types.get(col) not in [
            "连续",
            "次数",
        ]:
            continue

        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        valid = pd.concat(
            [target_values, values],
            axis=1,
        ).dropna()

        if len(valid) < 3:
            continue

        if (
            valid.iloc[:, 0].nunique() <= 1
            or valid.iloc[:, 1].nunique() <= 1
        ):
            continue

        try:
            pearson_value, pearson_p = pearsonr(
                valid.iloc[:, 0],
                valid.iloc[:, 1],
            )

            spearman_value, spearman_p = spearmanr(
                valid.iloc[:, 0],
                valid.iloc[:, 1],
            )

            rows.append(
                {
                    "变量": col,
                    "Pearson相关系数": pearson_value,
                    "Pearson_P值": pearson_p,
                    "Spearman相关系数": spearman_value,
                    "Spearman_P值": spearman_p,
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)


def create_variable_symbol_table(
    target,
    predictors,
    variable_types,
):
    """创建变量符号表。"""
    rows = []

    for index, col in enumerate(
        [target] + predictors
    ):
        if col == target:
            default_symbol = "y"
        else:
            default_symbol = f"x_{index}"

        rows.append(
            {
                "变量符号": default_symbol,
                "原始列名": col,
                "变量角色": (
                    "因变量"
                    if col == target
                    else "自变量"
                ),
                "变量类型": variable_types.get(
                    col,
                    "",
                ),
                "单位": "",
                "变量含义": "",
            }
        )

    return pd.DataFrame(rows)

# ============================================================
# 八、模型数据构造
# ============================================================

def build_model_data(
    df,
    target,
    predictors,
    variable_types,
    group_col=None,
):
    """构造模型数据并自动处理分类变量。"""
    use_cols = [target] + list(predictors)

    if group_col not in [None, "无"]:
        use_cols.append(group_col)

    use_cols = list(dict.fromkeys(use_cols))
    selected = df[use_cols].copy()

    target_mapping = None
    target_type = variable_types[target]

    if target_type in ["连续", "次数"]:
        selected[target] = pd.to_numeric(
            selected[target],
            errors="coerce",
        )

    elif target_type == "分类":
        selected[target] = selected[target].astype(
            "string"
        )

        categories = sorted(
            selected[target]
            .dropna()
            .unique()
            .tolist()
        )

        if len(categories) < 2:
            raise ValueError(
                "分类因变量至少需要两个类别。"
            )

        target_mapping = {
            category: index
            for index, category in enumerate(categories)
        }

        selected[target] = selected[target].map(
            target_mapping
        )

    elif target_type == "时间":
        raise ValueError(
            "时间型因变量不适合当前回归模块。"
        )

    for col in predictors:
        var_type = variable_types[col]

        if var_type in ["连续", "次数"]:
            selected[col] = pd.to_numeric(
                selected[col],
                errors="coerce",
            )

        elif var_type == "时间":
            dates = pd.to_datetime(
                selected[col],
                errors="coerce",
            )

            selected[col] = (
                dates - pd.Timestamp("1970-01-01")
            ).dt.total_seconds() / 86400

        elif var_type == "分类":
            selected[col] = selected[col].astype(
                "string"
            )

    required_cols = [target] + list(predictors)

    if group_col not in [None, "无"]:
        required_cols.append(group_col)

    selected = selected.dropna(
        subset=required_cols
    )

    y = selected[target].copy()
    X = selected[predictors].copy()

    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=float,
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    constant_columns = [
        col
        for col in X.columns
        if X[col].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        X = X.drop(
            columns=constant_columns
        )

    valid_rows = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if X.shape[1] == 0:
        raise ValueError(
            "处理后没有有效的自变量。"
        )

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    groups = None

    if group_col not in [None, "无"]:
        groups = selected.loc[
            valid_rows,
            group_col,
        ].astype(str)

    metadata = {
        "target_mapping": target_mapping,
        "feature_names": list(X.columns),
        "n_rows": len(y),
        "n_features": X.shape[1],
        "constant_columns": constant_columns,
    }

    return (
        y.reset_index(drop=True),
        X.reset_index(drop=True),
        (
            None
            if groups is None
            else groups.reset_index(drop=True)
        ),
        metadata,
    )


# ============================================================
# 九、模型推荐、验证和拟合
# ============================================================

MODEL_OPTIONS = [
    "多元线性回归",
    "Logit变换线性回归",
    "线性混合效应模型",
    "比例型混合效应模型",
    "二项Logistic回归",
    "多项Logistic回归",
    "Poisson回归",
    "负二项回归",
]


def detect_model_type(
    y,
    target_type,
    groups=None,
):
    """根据数据结构推荐模型。"""
    y_numeric = pd.to_numeric(
        y,
        errors="coerce",
    ).dropna()

    if target_type == "分类":
        unique_count = y.nunique()

        if unique_count == 2:
            return {
                "model_type": "二项Logistic回归",
                "reason": "因变量为二分类变量。",
            }

        if unique_count > 2:
            return {
                "model_type": "多项Logistic回归",
                "reason": "因变量为多分类变量。",
            }

    if target_type == "次数":
        if (
            len(y_numeric) > 0
            and (y_numeric >= 0).all()
            and np.allclose(
                y_numeric,
                np.round(y_numeric),
            )
        ):
            mean_value = y_numeric.mean()
            variance_value = y_numeric.var()

            if (
                mean_value > 0
                and variance_value > 1.5 * mean_value
            ):
                return {
                    "model_type": "负二项回归",
                    "reason": (
                        "因变量为计数变量，且方差可能明显大于均值，"
                        "存在过度离散现象。"
                    ),
                }

            return {
                "model_type": "Poisson回归",
                "reason": "因变量为非负整数计数变量。",
            }

    if target_type == "连续":
        if len(y_numeric) == 0:
            return {
                "model_type": "未识别",
                "reason": "因变量没有有效数值。",
            }

        in_unit_interval = (
            (y_numeric >= 0).all()
            and (y_numeric <= 1).all()
        )

        repeated = False

        if groups is not None:
            repeated = (
                groups.nunique()
                < len(groups)
            )

        if in_unit_interval and repeated:
            return {
                "model_type": "比例型混合效应模型",
                "reason": (
                    "因变量取值在0到1之间，且存在重复观测。"
                ),
            }

        if in_unit_interval:
            return {
                "model_type": "Logit变换线性回归",
                "reason": (
                    "因变量为0到1之间的比例变量。"
                ),
            }

        if repeated:
            return {
                "model_type": "线性混合效应模型",
                "reason": (
                    "连续型因变量存在重复观测分组。"
                ),
            }

        return {
            "model_type": "多元线性回归",
            "reason": "因变量为一般连续变量。",
        }

    return {
        "model_type": "未识别",
        "reason": "无法判断合适的模型类型。",
    }


def validate_model_selection(
    y,
    target_type,
    model_type,
    groups=None,
):
    """验证模型和数据类型是否匹配。"""
    if y is None or len(y) == 0:
        return False, "当前没有可用于建模的数据。"

    y_numeric = pd.to_numeric(
        pd.Series(y),
        errors="coerce",
    )

    if y_numeric.isna().any():
        return False, "因变量中存在无法转换为数值的内容。"

    if model_type == "多元线性回归":
        if target_type not in ["连续", "次数"]:
            return False, (
                "多元线性回归要求因变量为数值型变量。"
            )

    elif model_type == "Logit变换线性回归":
        if target_type not in ["连续", "次数"]:
            return False, (
                "Logit变换线性回归要求因变量为数值型变量。"
            )

        if ((y_numeric < 0) | (y_numeric > 1)).any():
            return False, (
                "Logit变换线性回归要求因变量取值在0到1之间。"
            )

        if y_numeric.nunique() < 2:
            return False, (
                "Logit变换线性回归要求因变量至少有两个不同取值。"
            )

    elif model_type == "二项Logistic回归":
        if target_type != "分类":
            return False, (
                "二项Logistic回归要求因变量为分类变量。"
            )

        if pd.Series(y).nunique() != 2:
            return False, (
                "二项Logistic回归要求因变量恰好包含两个类别。"
            )

    elif model_type == "多项Logistic回归":
        if target_type != "分类":
            return False, (
                "多项Logistic回归要求因变量为分类变量。"
            )

        if pd.Series(y).nunique() < 3:
            return False, (
                "多项Logistic回归至少需要三个类别。"
            )

    elif model_type in ["Poisson回归", "负二项回归"]:
        if target_type != "次数":
            return False, (
                f"{model_type}要求因变量为次数变量。"
            )

        if (y_numeric < 0).any():
            return False, (
                f"{model_type}要求因变量不能小于0。"
            )

        if not np.allclose(
            y_numeric.to_numpy(),
            np.round(y_numeric.to_numpy()),
        ):
            return False, (
                f"{model_type}要求因变量为非负整数。"
            )

    elif model_type in [
        "线性混合效应模型",
        "比例型混合效应模型",
    ]:
        if target_type not in ["连续", "次数"]:
            return False, (
                f"{model_type}要求因变量为数值型变量。"
            )

        if groups is None:
            return False, (
                f"{model_type}必须指定分组变量。"
            )

        if pd.Series(groups).nunique() < 2:
            return False, (
                "分组变量至少需要包含两个不同的组。"
            )

        if model_type == "比例型混合效应模型":
            if ((y_numeric < 0) | (y_numeric > 1)).any():
                return False, (
                    "比例型混合效应模型要求因变量取值在0到1之间。"
                )

    return True, ""


def logit_transform(y):
    """比例变量 Logit 变换。"""
    values = pd.Series(
        y,
        dtype=float,
    )

    eps = 1e-6

    values = values.clip(
        lower=eps,
        upper=1 - eps,
    )

    return np.log(
        values / (1 - values)
    )


def inverse_logit(z):
    """Logit 逆变换。"""
    values = np.asarray(z)
    values = np.clip(values, -700, 700)
    return 1 / (1 + np.exp(-values))


def fit_model(
    y,
    X,
    groups,
    model_type,
    robust_se=False,
):
    """拟合最终模型。"""
    if model_type == "多元线性回归":
        if robust_se:
            model = sm.OLS(y, X).fit(
                cov_type="HC3"
            )
        else:
            model = sm.OLS(y, X).fit()

        return {
            "model": model,
            "display_y": y,
            "prediction": model.predict(X),
            "model_type": model_type,
        }

    if model_type == "Logit变换线性回归":
        transformed_y = logit_transform(y)

        if robust_se:
            model = sm.OLS(
                transformed_y,
                X,
            ).fit(cov_type="HC3")
        else:
            model = sm.OLS(
                transformed_y,
                X,
            ).fit()

        prediction = inverse_logit(
            model.predict(X)
        )

        return {
            "model": model,
            "display_y": y,
            "prediction": prediction,
            "transformed_y": transformed_y,
            "model_type": model_type,
        }

    if model_type == "线性混合效应模型":
        if groups is None:
            raise ValueError(
                "线性混合效应模型必须指定分组变量。"
            )

        model = sm.MixedLM(
            endog=y,
            exog=X,
            groups=groups,
        ).fit(
            reml=False,
            method="lbfgs",
            disp=False,
        )

        return {
            "model": model,
            "display_y": y,
            "prediction": model.predict(X),
            "model_type": model_type,
        }

    if model_type == "比例型混合效应模型":
        if groups is None:
            raise ValueError(
                "比例型混合效应模型必须指定分组变量。"
            )

        transformed_y = logit_transform(y)

        model = sm.MixedLM(
            endog=transformed_y,
            exog=X,
            groups=groups,
        ).fit(
            reml=False,
            method="lbfgs",
            disp=False,
        )

        prediction = inverse_logit(
            model.predict(X)
        )

        return {
            "model": model,
            "display_y": y,
            "prediction": prediction,
            "transformed_y": transformed_y,
            "model_type": model_type,
        }

    if model_type == "Poisson回归":
        model = sm.GLM(
            y,
            X,
            family=sm.families.Poisson(),
        ).fit()

        return {
            "model": model,
            "display_y": y,
            "prediction": model.predict(X),
            "model_type": model_type,
        }

    if model_type == "负二项回归":
        model = sm.GLM(
            y,
            X,
            family=sm.families.NegativeBinomial(),
        ).fit()

        return {
            "model": model,
            "display_y": y,
            "prediction": model.predict(X),
            "model_type": model_type,
        }

    if model_type == "二项Logistic回归":
        if y.nunique() != 2:
            raise ValueError(
                "二项Logistic回归要求因变量恰好有两个类别。"
            )

        model = sm.GLM(
            y,
            X,
            family=sm.families.Binomial(),
        ).fit()

        probability = np.asarray(
            model.predict(X),
            dtype=float,
        )

        prediction = (
            probability >= 0.5
        ).astype(int)

        return {
            "model": model,
            "display_y": y,
            "prediction": prediction,
            "probability": probability,
            "model_type": model_type,
        }

    if model_type == "多项Logistic回归":
        if y.nunique() < 3:
            raise ValueError(
                "多项Logistic回归至少需要三个类别。"
            )

        model = sm.MNLogit(
            y,
            X,
        ).fit(
            disp=False,
            maxiter=300,
        )

        probability = np.asarray(
            model.predict(X)
        )

        prediction = probability.argmax(axis=1)

        return {
            "model": model,
            "display_y": y,
            "prediction": prediction,
            "probability": probability,
            "model_type": model_type,
        }

    raise ValueError(
        f"暂不支持的模型：{model_type}"
    )


# ============================================================
# 十、模型结果、诊断和论文文本
# ============================================================

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
        result = pd.DataFrame(
            {
                "变量": params.index,
                "回归系数": params.values,
                "标准误": np.asarray(bse),
                "P值": np.asarray(pvalues),
            }
        )

        if conf_int is not None:
            result["置信区间下限"] = (
                conf_int.iloc[:, 0].values
            )
            result["置信区间上限"] = (
                conf_int.iloc[:, 1].values
            )

        if "Logistic" in model_type:
            result["优势比_OR"] = np.exp(
                result["回归系数"]
            )

        result["显著性判断"] = result[
            "P值"
        ].apply(significance_label)

        return result

    params_df = pd.DataFrame(params)
    bse_df = pd.DataFrame(bse)
    pvalues_df = pd.DataFrame(pvalues)

    rows = []

    for variable in params_df.index:
        for category in params_df.columns:
            coef = params_df.loc[
                variable,
                category,
            ]

            p_value = pvalues_df.loc[
                variable,
                category,
            ]

            rows.append(
                {
                    "变量": variable,
                    "类别": category,
                    "回归系数": coef,
                    "标准误": bse_df.loc[
                        variable,
                        category,
                    ],
                    "P值": p_value,
                    "优势比_OR": np.exp(coef),
                    "显著性判断": significance_label(
                        p_value
                    ),
                }
            )

    return pd.DataFrame(rows)


def calculate_vif(X):
    """计算 VIF。"""
    x_df = X.copy()

    if "const" in x_df.columns:
        x_df = x_df.drop(
            columns=["const"]
        )

    x_df = x_df.loc[
        :,
        x_df.nunique(dropna=True) > 1,
    ]

    if x_df.shape[1] <= 1:
        return pd.DataFrame(
            columns=["变量", "VIF"]
        )

    rows = []

    for index, col in enumerate(x_df.columns):
        try:
            value = variance_inflation_factor(
                x_df.astype(float).values,
                index,
            )
        except Exception:
            value = np.inf

        rows.append(
            {
                "变量": col,
                "VIF": value,
            }
        )

    return pd.DataFrame(rows)


def make_metric_table(fitted_result):
    """生成模型评价指标表。"""
    if fitted_result is None:
        return pd.DataFrame()

    model = fitted_result["model"]
    model_type = fitted_result["model_type"]

    y = np.asarray(
        fitted_result["display_y"],
        dtype=float,
    )

    prediction = np.asarray(
        fitted_result["prediction"],
        dtype=float,
    )

    rows = [
        ["模型类型", model_type],
        ["样本量", len(y)],
        [
            "参数数量",
            len(
                np.asarray(
                    model.params
                ).reshape(-1)
            ),
        ],
    ]

    if hasattr(model, "aic"):
        rows.append(["AIC", model.aic])

    if hasattr(model, "bic"):
        try:
            rows.append(["BIC", model.bic])
        except Exception:
            pass

    if model_type in [
        "多元线性回归",
        "Logit变换线性回归",
        "线性混合效应模型",
        "比例型混合效应模型",
    ]:
        rows.extend(
            [
                [
                    "RMSE",
                    np.sqrt(
                        mean_squared_error(
                            y,
                            prediction,
                        )
                    ),
                ],
                [
                    "MAE",
                    mean_absolute_error(
                        y,
                        prediction,
                    ),
                ],
            ]
        )

        if model_type in [
            "多元线性回归",
            "Logit变换线性回归",
        ]:
            rows.append(
                [
                    "R²",
                    r2_score(y, prediction),
                ]
            )

            if hasattr(model, "rsquared"):
                rows.append(
                    ["模型R²", model.rsquared]
                )

            if hasattr(model, "rsquared_adj"):
                rows.append(
                    ["调整R²", model.rsquared_adj]
                )

            if hasattr(model, "fvalue"):
                rows.append(
                    ["F统计量", model.fvalue]
                )

            if hasattr(model, "f_pvalue"):
                rows.append(
                    ["F检验P值", model.f_pvalue]
                )

        if hasattr(model, "llf"):
            rows.append(
                ["对数似然", model.llf]
            )

    elif model_type in [
        "Poisson回归",
        "负二项回归",
    ]:
        rows.extend(
            [
                [
                    "RMSE",
                    np.sqrt(
                        mean_squared_error(
                            y,
                            prediction,
                        )
                    ),
                ],
                [
                    "MAE",
                    mean_absolute_error(
                        y,
                        prediction,
                    ),
                ],
            ]
        )

        if hasattr(model, "deviance"):
            rows.append(
                ["偏差Deviance", model.deviance]
            )

        if (
            hasattr(model, "deviance")
            and hasattr(model, "null_deviance")
            and model.null_deviance != 0
        ):
            rows.append(
                [
                    "伪R²",
                    1
                    - model.deviance
                    / model.null_deviance,
                ]
            )

    elif model_type == "二项Logistic回归":
        probability = np.asarray(
            fitted_result["probability"],
            dtype=float,
        )

        predicted_class = (
            probability >= 0.5
        ).astype(int)

        rows.extend(
            [
                [
                    "准确率",
                    accuracy_score(
                        y.astype(int),
                        predicted_class,
                    ),
                ],
                [
                    "平衡准确率",
                    balanced_accuracy_score(
                        y.astype(int),
                        predicted_class,
                    ),
                ],
                [
                    "精确率",
                    precision_score(
                        y.astype(int),
                        predicted_class,
                        zero_division=0,
                    ),
                ],
                [
                    "召回率",
                    recall_score(
                        y.astype(int),
                        predicted_class,
                        zero_division=0,
                    ),
                ],
                [
                    "F1值",
                    f1_score(
                        y.astype(int),
                        predicted_class,
                        zero_division=0,
                    ),
                ],
                [
                    "Log Loss",
                    log_loss(
                        y.astype(int),
                        probability,
                    ),
                ],
            ]
        )

        if len(np.unique(y)) == 2:
            rows.append(
                [
                    "ROC-AUC",
                    roc_auc_score(
                        y,
                        probability,
                    ),
                ]
            )

    elif model_type == "多项Logistic回归":
        predicted_class = np.asarray(
            fitted_result["prediction"],
            dtype=int,
        )

        rows.extend(
            [
                [
                    "准确率",
                    accuracy_score(
                        y.astype(int),
                        predicted_class,
                    ),
                ],
                [
                    "平衡准确率",
                    balanced_accuracy_score(
                        y.astype(int),
                        predicted_class,
                    ),
                ],
                [
                    "宏平均F1",
                    f1_score(
                        y.astype(int),
                        predicted_class,
                        average="macro",
                        zero_division=0,
                    ),
                ],
            ]
        )

    return pd.DataFrame(
        rows,
        columns=["指标", "数值"],
    )


def create_prediction_table(fitted_result):
    """创建实际值、预测值和残差表。"""
    if fitted_result is None:
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "实际值": np.asarray(
                fitted_result["display_y"]
            ),
            "预测值": np.asarray(
                fitted_result["prediction"]
            ),
        }
    )

    result["残差"] = (
        result["实际值"]
        - result["预测值"]
    )

    if "probability" in fitted_result:
        probability = np.asarray(
            fitted_result["probability"]
        )

        if probability.ndim == 1:
            result["预测概率"] = probability

        result["预测类别"] = np.asarray(
            fitted_result["prediction"]
        )

    return result


def make_diagnostic_table(model):
    """生成线性模型诊断表。"""
    rows = []

    residuals = np.asarray(
        model.resid,
        dtype=float,
    )

    rows.append(
        [
            "Durbin-Watson",
            durbin_watson(residuals),
        ]
    )

    if len(residuals) >= 3:
        try:
            shapiro_values = residuals

            if len(shapiro_values) > 5000:
                rng = np.random.default_rng(42)
                indices = rng.choice(
                    len(shapiro_values),
                    5000,
                    replace=False,
                )
                shapiro_values = (
                    shapiro_values[indices]
                )

            _, shapiro_p = shapiro(
                shapiro_values
            )

            rows.append(
                [
                    "残差Shapiro-Wilk P值",
                    shapiro_p,
                ]
            )

        except Exception:
            pass

    try:
        bp_result = het_breuschpagan(
            residuals,
            model.model.exog,
        )

        rows.extend(
            [
                [
                    "Breusch-Pagan统计量",
                    bp_result[0],
                ],
                [
                    "Breusch-Pagan P值",
                    bp_result[1],
                ],
            ]
        )

    except Exception:
        pass

    try:
        white_result = het_white(
            residuals,
            model.model.exog,
        )

        rows.extend(
            [
                [
                    "White检验统计量",
                    white_result[0],
                ],
                [
                    "White检验P值",
                    white_result[1],
                ],
            ]
        )

    except Exception:
        pass

    return pd.DataFrame(
        rows,
        columns=["诊断指标", "数值"],
    )


def generate_assumptions(
    target,
    predictors,
    variable_types,
    model_type,
    vif_table=None,
    group_col=None,
):
    """生成可用于论文的模型假设。"""
    assumptions = [
        "假设题目所给数据真实可靠，能够反映研究对象的主要特征。",
        "假设各变量的单位、编码方式和统计口径保持一致。",
        "假设缺失值及异常值处理不会改变数据的主要统计规律。",
    ]

    if (
        vif_table is not None
        and not vif_table.empty
    ):
        finite_vif = vif_table[
            np.isfinite(vif_table["VIF"])
        ]

        if not finite_vif.empty:
            max_vif = finite_vif["VIF"].max()

            if max_vif < 5:
                assumptions.append(
                    "各解释变量VIF均小于5，暂未发现明显的多重共线性。"
                )
            elif max_vif < 10:
                assumptions.append(
                    "部分解释变量VIF介于5和10之间，可能存在一定的多重共线性。"
                )
            else:
                assumptions.append(
                    "部分解释变量VIF不低于10，存在较严重的多重共线性风险。"
                )

    if model_type == "多元线性回归":
        assumptions.extend(
            [
                "假设因变量与解释变量之间的条件均值关系可以用线性函数近似表示。",
                "假设误差项条件均值为零，且不同观测之间相互独立。",
                "假设误差项具有近似恒定的方差。",
                "假设不存在对估计结果产生决定性影响的极端观测。",
            ]
        )

    elif model_type == "Logit变换线性回归":
        assumptions.extend(
            [
                "假设因变量为0到1之间的比例变量。",
                "对因变量进行Logit变换后，其与解释变量之间近似满足线性关系。",
                "假设变换后的误差项相互独立且方差相对稳定。",
                "模型预测结果通过逆Logit变换还原为比例形式。",
            ]
        )

    elif model_type == "线性混合效应模型":
        assumptions.extend(
            [
                "假设同一分组内的多次观测可能存在相关性。",
                "通过随机截距描述不同分组对象之间的个体差异。",
                "假设随机效应均值为零，并与固定效应结构相互独立。",
            ]
        )

    elif model_type == "比例型混合效应模型":
        assumptions.extend(
            [
                "假设因变量为0到1之间的比例变量。",
                "对因变量进行Logit变换，并通过随机截距处理组内相关性。",
                "模型预测结果通过逆Logit变换还原为比例形式。",
            ]
        )

    elif model_type == "二项Logistic回归":
        assumptions.extend(
            [
                "假设因变量为二分类变量，并编码为0和1。",
                "假设事件发生的对数优势比与解释变量近似线性相关。",
                "假设观测样本之间相互独立。",
            ]
        )

    elif model_type == "多项Logistic回归":
        assumptions.extend(
            [
                "假设因变量为多分类变量。",
                "以一个类别作为参照类别，比较其他类别的发生优势。",
                "假设不同观测样本之间相互独立。",
            ]
        )

    elif model_type == "Poisson回归":
        assumptions.extend(
            [
                "假设因变量为非负整数计数变量。",
                "假设计数变量服从Poisson分布或近似服从Poisson分布。",
                "采用对数连接函数描述解释变量与计数均值的关系。",
                "需要检查数据是否存在过度离散。",
            ]
        )

    elif model_type == "负二项回归":
        assumptions.extend(
            [
                "假设因变量为非负整数计数变量。",
                "假设数据存在超过Poisson分布的额外离散程度。",
                "采用负二项分布和对数连接函数进行建模。",
            ]
        )

    if group_col not in [None, "无"]:
        assumptions.append(
            f'以"{group_col}"作为重复观测分组变量，同一分组内的观测可能存在相关性。'
        )

    return list(dict.fromkeys(assumptions))


# ============================================================
# 十一、侧边栏设置
# ============================================================

    header_html = """
    <style>
    .furry-header {
        display: flex;
        align-items: center;
        gap: 18px;
        margin: 8px 0 4px 0;
        min-height: 150px;
    }

    .furry-title {
        flex: 1;
        min-width: 0;
    }

    .furry-title h1 {
        margin: 0;
        color: #243447;
        font-size: clamp(28px, 3.2vw, 48px);
        line-height: 1.2;
        font-weight: 750;
        letter-spacing: 0;
    }

    .furry-title p {
        margin: 12px 0 0 0;
        color: #687586;
        font-size: 16px;
        line-height: 1.6;
    }

    .fursona-image {
        width: min(270px, 30vw);
        max-height: 170px;
        object-fit: contain;
        object-position: center;
        flex-shrink: 0;
    }

    @media (max-width: 700px) {
        .furry-header {
            gap: 8px;
            min-height: 105px;
        }

        .fursona-image {
            width: 125px;
            max-height: 105px;
        }

        .furry-title h1 {
            font-size: 25px;
        }

        .furry-title p {
            font-size: 13px;
            line-height: 1.4;
        }
    }
    </style>

    <div class="furry-header">
        <div class="furry-title">
            <h1>数学建模大赛前期数据分析工具</h1>
            <p>数据清洗、变量识别、可视化、相关性分析、模型推荐与统计建模</p>
        </div>
        <img
            class="fursona-image"
            src="data:image/png;base64,PLACEHOLDER"
            alt="兽设抱着数"
        >
    </div>
    """

    header_html = header_html.replace(
        "PLACEHOLDER",
        fursona_base64,
    )

    st.markdown(
        header_html,
        unsafe_allow_html=True,
    )
    
with st.sidebar:
    st.title('功能导航')
    app_mode = st.radio(
        '请选择功能模块',
        ['数据分析', '优化求解'],
        index=0
    )
    if app_mode != st.session_state.get('app_mode', '数据分析'):
        st.session_state.app_mode = app_mode
        st.rerun()
    st.header("基本设置")

    st.subheader("赛题描述")

    uploaded_problem_file = st.file_uploader(
        "上传赛题文件（PDF / Word / TXT）",
        type=["pdf", "docx", "txt"],
        key="problem_file_uploader",
    )

    if (
        uploaded_problem_file is not None
        and st.session_state.get(
            "last_problem_file"
        )
        != uploaded_problem_file.name
    ):
        st.session_state["problem_text"] = (
            extract_text_from_file(
                uploaded_problem_file
            )
        )
        st.session_state["last_problem_file"] = (
            uploaded_problem_file.name
        )

    problem_text = st.text_area(
        "粘贴赛题原文或显示上传文件内容",
        value=st.session_state.get(
            "problem_text",
            "",
        ),
        height=160,
        placeholder=(
            "上传文件后自动显示内容，也可以直接在此处粘贴。"
        ),
        key="problem_text_area",
    )

    st.session_state["problem_text"] = problem_text

    if st.button("自动检测题型"):
        result = multi_label_classify_problem_text(
            problem_text
        )

        st.session_state["detect_result"] = result
        st.session_state["problem_type"] = (
            result["main_type"]
        )

    detect_result = st.session_state.get(
        "detect_result",
        {},
    )

    if detect_result:
        st.success(
            f"主类型：{detect_result['main_type']}"
        )

        detected_labels = detect_result.get(
            "all_detected_labels",
            [],
        )

        st.info(
            "检测到的类型："
            + (
                "、".join(detected_labels)
                if detected_labels
                else "无"
            )
        )

        if detect_result.get(
            "sub_question_context"
        ):
            with st.expander(
                "查看自动提取到的子问题片段"
            ):
                for index, snippet in enumerate(
                    detect_result[
                        "sub_question_context"
                    ],
                    1,
                ):
                    st.markdown(
                        f"**子问题 {index}：** "
                        f"{snippet}……"
                    )

        with st.expander(
            "查看题型识别分数和建模建议"
        ):
            score_table = pd.DataFrame(
                {
                    "题型类别": list(
                        detect_result.get(
                            "label_scores",
                            {},
                        ).keys()
                    ),
                    "综合得分": list(
                        detect_result.get(
                            "label_scores",
                            {},
                        ).values()
                    ),
                }
            )

            st.dataframe(
                score_table,
                use_container_width=True,
            )

            direction_map = {
                "评价类": (
                    "层次分析、TOPSIS、模糊综合评价、"
                    "灰色关联分析、主成分分析"
                ),
                "预测类": (
                    "多元回归、ARIMA、灰色预测、"
                    "神经网络、时间序列分析"
                ),
                "优化类": (
                    "线性规划、整数规划、0-1规划、"
                    "遗传算法、模拟退火"
                ),
                "机理分析类": (
                    "微分方程、动力学模型、Logistic模型、"
                    "SIR模型、稳定性分析"
                ),
                "分类类": (
                    "Logistic回归、决策树、随机森林、"
                    "SVM、聚类分析"
                ),
            }

            for label in detected_labels:
                st.write(
                    f"**{label}：** "
                    f"{direction_map.get(label, '根据题目背景确定')}"
                )

    problem_type = st.text_input(
        "赛题类型",
        value=st.session_state.get(
            "problem_type",
            "",
        ),
        placeholder="例如：预测类、评价类、优化类",
    )

    uploaded_file = st.file_uploader(
        "上传数据表",
        type=["csv", "xlsx", "xls"],
        key="data_file_uploader",
    )

    st.subheader("缺失值处理")

    missing_method = st.selectbox(
        "处理方式",
        [
            "分类型处理",
            "删除含缺失值的行",
        ],
    )

    st.subheader("异常值处理")

    outlier_method = st.selectbox(
        "识别方法",
        [
            "不处理",
            "3σ",
            "IQR",
        ],
    )

    outlier_action = st.selectbox(
        "异常值处理动作",
        [
            "仅标记，不删除",
            "删除异常行",
        ],
    )

    robust_se = st.checkbox(
        "线性回归使用HC3稳健标准误",
        value=True,
    )

    use_test_set = st.checkbox(
        "进行训练集/测试集评估",
        value=True,
    )

    test_size = st.slider(
        "测试集比例",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )


if st.session_state.get('app_mode', '数据分析') == '数据分析':
    # ============================================================
    # 十二、读取数据
    # ============================================================

    if uploaded_file is None:
        st.info("请在左侧上传 CSV 或 Excel 数据表。")
        st.stop()

    try:
        uploaded_file.seek(0)

        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

    except Exception as exc:
        st.error(f"读取文件失败：{exc}")
        st.stop()

    if raw_df is None or raw_df.empty:
        st.error("数据未成功加载，请检查文件内容。")
        st.stop()

    raw_df.columns = make_unique_columns(
        raw_df.columns
    )

    original_shape = raw_df.shape
    all_columns = list(raw_df.columns)

    if len(all_columns) < 2:
        st.error(
            "数据表至少需要包含一列因变量和一列自变量。"
        )
        st.stop()


    # ============================================================
    # 十三、数据初探
    # ============================================================

    st.subheader("1. 数据初探")

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    metric_col1.metric(
        "原始行数",
        original_shape[0],
    )

    metric_col2.metric(
        "原始列数",
        original_shape[1],
    )

    metric_col3.metric(
        "缺失单元格数",
        int(raw_df.isna().sum().sum()),
    )

    metric_col4.metric(
        "重复行数",
        int(raw_df.duplicated().sum()),
    )

    st.write("数据前20行")
    st.dataframe(
        raw_df.head(20),
        use_container_width=True,
    )

    with st.expander("查看字段基本信息"):
        info_table = pd.DataFrame(
            {
                "列名": raw_df.columns,
                "数据类型": raw_df.dtypes.astype(str).values,
                "非空数量": raw_df.notna().sum().values,
                "缺失数量": raw_df.isna().sum().values,
                "唯一值数量": raw_df.nunique(
                    dropna=True
                ).values,
                "是否疑似ID列": [
                    "是"
                    if is_suspicious_id_column(
                        raw_df[col],
                        col,
                    )
                    else "否"
                    for col in raw_df.columns
                ],
            }
        )

        st.dataframe(
            info_table,
            use_container_width=True,
        )

        dataframe_download(
            info_table,
            "字段基本信息.csv",
            key="download_info_table",
        )


    # ============================================================
    # 十四、变量选择
    # ============================================================

    st.subheader("2. 因变量、自变量与分组变量设置")

    target = st.selectbox(
        "请选择因变量",
        all_columns,
        index=0,
    )

    default_predictors = [
        col
        for col in all_columns
        if (
            col != target
            and not is_suspicious_id_column(
                raw_df[col],
                col,
            )
        )
    ]

    predictors = st.multiselect(
        "请选择自变量",
        [
            col
            for col in all_columns
            if col != target
        ],
        default=default_predictors,
    )

    if not predictors:
        st.warning("请至少选择一个自变量。")
        st.stop()

    group_candidates = ["无"]

    for col in all_columns:
        if col == target or col in predictors:
            continue

        non_missing = raw_df[col].dropna()

        if len(non_missing) == 0:
            continue

        group_count = non_missing.nunique()
        group_sizes = non_missing.value_counts()

        if (
            group_count >= 2
            and group_sizes.max() >= 2
            and group_count < len(raw_df)
        ):
            group_candidates.append(col)

    group_col = st.selectbox(
        "重复观测分组变量",
        group_candidates,
        help=(
            "如果同一对象有多次观测，请选择对应的对象编号。"
        ),
    )

    if group_col != "无" and group_col in predictors:
        predictors = [
            col
            for col in predictors
            if col != group_col
        ]

    if not predictors:
        st.error(
            "移除分组变量后没有剩余自变量。"
        )
        st.stop()


    # ============================================================
    # 十五、变量类型识别
    # ============================================================

    st.subheader("3. 变量类型识别与确认")

    initial_variable_types = {
        col: classify_variable(raw_df[col])
        for col in [target] + predictors
    }

    initial_type_table = pd.DataFrame(
        {
            "变量": list(
                initial_variable_types.keys()
            ),
            "自动识别类型": list(
                initial_variable_types.values()
            ),
            "原始数据类型": [
                str(raw_df[col].dtype)
                for col in initial_variable_types
            ],
            "唯一值数量": [
                raw_df[col].nunique(
                    dropna=True
                )
                for col in initial_variable_types
            ],
        }
    )

    st.dataframe(
        initial_type_table,
        use_container_width=True,
    )

    st.caption(
        "自动识别结果仅供参考，正式建模前请根据变量实际含义确认类型。"
    )

    variable_types = {}
    type_options = [
        "连续",
        "分类",
        "时间",
        "次数",
    ]

    for col in [target] + predictors:
        default_type = initial_variable_types[col]

        if default_type not in type_options:
            default_type = "分类"

        default_index = type_options.index(
            default_type
        )

        variable_types[col] = st.selectbox(
            f"确认变量 `{col}` 的类型",
            type_options,
            index=default_index,
            key=f"variable_type_{col}",
        )


    # 注意：
    # 页面上的变量类型已经由用户确认。
    # 这里不能再次自动覆盖，否则“次数”“分类”等人工设置会失效。
    # 自动识别结果只用于初始默认值。


    current_signature = build_analysis_signature(
        file_name=uploaded_file.name,
        target=target,
        predictors=predictors,
        variable_types=variable_types,
        group_col=group_col,
        missing_method=missing_method,
        outlier_method=outlier_method,
        outlier_action=outlier_action,
        robust_se=robust_se,
        use_test_set=use_test_set,
        test_size=test_size,
    )

    reset_model_if_signature_changed(
        current_signature
    )


    # ============================================================
    # 十六、符号表
    # ============================================================

    st.subheader("4. 数学建模符号表")

    symbol_mode = st.radio(
        "符号表类型",
        [
            "样本符号表",
            "变量符号表",
        ],
        horizontal=True,
    )

    # 注意：这里不要重新定义 target 和 predictors。
    # 它们已经在前面的建模设置部分确定。
    if symbol_mode == "样本符号表":

        sample_symbol_table = create_variable_symbol_table(
            target=target,
            predictors=predictors,
            variable_types=variable_types,
        )

        st.dataframe(
            sample_symbol_table,
            use_container_width=True,
        )

        dataframe_download(
            sample_symbol_table,
            "样本符号表.csv",
            key="download_sample_symbol_table",
        )

    else:
        variable_symbol_table = (
            create_variable_symbol_table(
                target,
                predictors,
                variable_types,
            )
        )

        st.info(
            r"可以直接编辑“变量符号”“单位”和“变量含义”。"
            r"例如：\mathrm{Year}_i"
        )

        edited_symbol_table = st.data_editor(
            variable_symbol_table,
            use_container_width=True,
            num_rows="fixed",
            disabled=[
                "原始列名",
                "变量角色",
                "变量类型",
            ],
            key="variable_symbol_editor",
        )

        st.write("LaTeX 变量符号预览")

        for _, row in edited_symbol_table.iterrows():
            symbol = str(row["变量符号"]).strip()
            original_name = str(row["原始列名"]).strip()

            if symbol:
                st.markdown(
                    f"原始变量：`{original_name}`"
                )
                st.latex(symbol)

        dataframe_download(
            edited_symbol_table,
            "变量符号表.csv",
            key="download_variable_symbol_table",
        )

    # ============================================================
    # 十七、数据清洗
    # ============================================================

    st.subheader("5. 数据清洗")

    st.info(
        "因变量缺失的样本无法用于监督建模，因此程序会直接删除；"
        "缺失值填补仅针对自变量。"
    )

    typed_df = convert_types(
        raw_df,
        variable_types,
    )

    # 自动把被读取为文本的数字列转换为数值型
    for col in typed_df.columns:
        if col == target:
            continue

        converted = pd.to_numeric(
            typed_df[col].astype(str).str.strip(),
            errors="coerce",
        )

        if converted.notna().mean() >= 0.8:
            typed_df[col] = converted


    # 自动修正被读取为文本的数字列
    for col in typed_df.columns:
        if col == target:
            continue

        converted = pd.to_numeric(
            typed_df[col].astype(str).str.strip(),
            errors="coerce",
        )

        if converted.notna().mean() >= 0.8:
            typed_df[col] = converted


    before_missing_cells = int(
        typed_df.isna().sum().sum()
    )

    target_missing_mask = typed_df[target].isna()
    deleted_target_missing_rows = int(
        target_missing_mask.sum()
    )

    typed_df = typed_df.loc[
        ~target_missing_mask
    ].copy()

    predictor_variable_types = {
        col: variable_types[col]
        for col in predictors
        if col in variable_types
    }

    (
        clean_df,
        missing_detail_table,
        deleted_missing_rows,
        imputed_cells,
    ) = fill_missing_values(
        typed_df,
        predictor_variable_types,
        missing_method,
    )

    numeric_columns_for_outlier = [
        col
        for col in [target] + predictors
        if variable_types.get(col)
        in ["连续", "次数"]
    ]

    outlier_detail_table = pd.DataFrame()
    outlier_row_count = 0
    deleted_outlier_rows = 0

    if outlier_method != "不处理":
        marked_df, outlier_detail_table = (
            detect_outliers(
                clean_df,
                numeric_columns_for_outlier,
                outlier_method,
            )
        )

        outlier_row_count = int(
            marked_df["_异常行"].sum()
        )

        if outlier_action == "删除异常行":
            deleted_outlier_rows = outlier_row_count
            clean_df = marked_df.loc[
                ~marked_df["_异常行"]
            ].copy()
        else:
            clean_df = marked_df.copy()

    else:
        clean_df["_异常行"] = False

    used_columns = [target] + predictors

    if group_col != "无":
        used_columns.append(group_col)

    if len(all_columns) > 0:
        used_columns.append(all_columns[0])

    used_columns = list(dict.fromkeys(used_columns))

    unused_columns = [
        col
        for col in clean_df.columns
        if (
            col not in used_columns
            and col != "_异常行"
        )
    ]

    clean_df = clean_df.drop(
        columns=unused_columns,
        errors="ignore",
    )

    after_missing_cells = int(
        clean_df.isna().sum().sum()
    )

    clean_df = clean_df.reset_index(
        drop=True
    )

    clean_data_for_model = clean_df.drop(
        columns=["_异常行"],
        errors="ignore",
    )

    cleaning_summary = pd.DataFrame(
        {
            "项目": [
                "处理时间",
                "赛题类型",
                "原始行数",
                "清洗后行数",
                "原始列数",
                "清洗后列数",
                "原始缺失单元格数",
                "缺失值插补数量",
                "因变量缺失删除行数",
                "自变量缺失删除行数",
                "检测到的异常行数",
                "因异常删除行数",
                "清洗后剩余缺失单元格数",
                "缺失值处理方式",
                "异常值识别方法",
                "异常值处理动作",
                "重复观测分组变量",
                "删除的无用列",
            ],
            "结果": [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                problem_type or "未填写",
                original_shape[0],
                clean_data_for_model.shape[0],
                original_shape[1],
                clean_data_for_model.shape[1],
                before_missing_cells,
                imputed_cells,
                deleted_target_missing_rows,
                deleted_missing_rows,
                outlier_row_count,
                deleted_outlier_rows,
                after_missing_cells,
                missing_method,
                outlier_method,
                outlier_action,
                group_col,
                (
                    ", ".join(unused_columns)
                    if unused_columns
                    else "无"
                ),
            ],
        }
    )

    clean_metric1, clean_metric2, clean_metric3, clean_metric4 = (
        st.columns(4)
    )

    clean_metric1.metric(
        "缺失值插补数量",
        imputed_cells,
    )

    clean_metric2.metric(
        "因缺失删除行数",
        deleted_target_missing_rows
        + deleted_missing_rows,
    )

    clean_metric3.metric(
        "检测到异常行数",
        outlier_row_count,
    )

    clean_metric4.metric(
        "清洗后样本数",
        clean_data_for_model.shape[0],
    )

    st.dataframe(
        cleaning_summary,
        use_container_width=True,
    )

    st.dataframe(
        missing_detail_table,
        use_container_width=True,
    )

    if not outlier_detail_table.empty:
        st.dataframe(
            outlier_detail_table,
            use_container_width=True,
        )

    st.dataframe(
        clean_data_for_model.head(20),
        use_container_width=True,
    )

    dataframe_download(
        cleaning_summary,
        "清洗汇总报告.csv",
        key="download_cleaning_summary",
    )

    dataframe_download(
        missing_detail_table,
        "缺失值处理明细.csv",
        key="download_missing_detail",
    )

    if not outlier_detail_table.empty:
        dataframe_download(
            outlier_detail_table,
            "异常值处理明细.csv",
            key="download_outlier_detail",
        )

    dataframe_download(
        clean_data_for_model,
        "清洗后数据.csv",
        key="download_clean_data",
    )

    cleaning_text = f"""
    本文首先对原始数据进行完整性、一致性和变量类型检查。
    原始数据共包含{original_shape[0]}条样本和{original_shape[1]}个变量。
    针对缺失数据，本文采用“{missing_method}”方法进行处理，
    共插补{imputed_cells}个缺失单元格。
    对于因变量缺失的样本，由于无法提供有效的被解释变量观测值，
    直接删除{deleted_target_missing_rows}条样本；
    另外因自变量缺失处理删除{deleted_missing_rows}条样本。
    针对异常数据，本文采用“{outlier_method}”方法进行识别，
    共检测到{outlier_row_count}条异常样本，
    其中因异常值删除{deleted_outlier_rows}条样本。
    经过数据类型转换、缺失值处理、异常值处理和无用列处理后，
    最终获得{clean_data_for_model.shape[0]}条有效样本，
    用于后续统计分析和模型建立。
    """.strip()

    st.text_area(
        "可直接用于论文的数据清洗表述",
        cleaning_text,
        height=220,
        key="cleaning_text_area",
    )


    # ============================================================
    # 十八、数据可视化
    # ============================================================

    st.subheader("6. 数据可视化")

    numeric_variables = [
        col
        for col in [target] + predictors
        if (
            variable_types.get(col)
            in ["连续", "次数"]
            and col in clean_data_for_model.columns
        )
    ]

    categorical_variables = [
        col
        for col in [target] + predictors
        if (
            variable_types.get(col) == "分类"
            and col in clean_data_for_model.columns
        )
    ]

    chart_type = st.selectbox(
        "选择图形类型",
        [
            "变量分布图",
            "变量箱线图",
            "因变量-自变量散点图",
            "因变量-自变量曲线图",
            "数值变量相关性热力图",
            "分类变量频数图",
        ],
    )

    if chart_type == "变量分布图":
        if not numeric_variables:
            st.info("当前没有连续型或次数型变量。")
        else:
            selected_col = st.selectbox(
                "选择变量",
                numeric_variables,
                key="histogram_column",
            )

            values = pd.to_numeric(
                clean_data_for_model[selected_col],
                errors="coerce",
            ).dropna()

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            sns.histplot(
                values,
                kde=True,
                ax=ax,
            )

            ax.set_title(
                f"{selected_col} 分布图"
            )
            ax.set_xlabel(selected_col)

            st.pyplot(fig)
            plt.close(fig)

    elif chart_type == "变量箱线图":
        if not numeric_variables:
            st.info("当前没有连续型或次数型变量。")
        else:
            selected_columns = st.multiselect(
                "选择变量",
                numeric_variables,
                default=numeric_variables,
                key="boxplot_columns",
            )

            if selected_columns:
                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                sns.boxplot(
                    data=clean_data_for_model[
                        selected_columns
                    ],
                    ax=ax,
                )

                ax.tick_params(
                    axis="x",
                    rotation=35,
                )

                ax.set_title("变量箱线图")

                st.pyplot(fig)
                plt.close(fig)

    elif chart_type == "因变量-自变量散点图":
        x_candidates = [
            col
            for col in numeric_variables
            if col != target
        ]

        if not x_candidates:
            st.info(
                "没有适合绘制散点图的数值型自变量。"
            )
        else:
            x_col = st.selectbox(
                "选择横轴自变量",
                x_candidates,
                key="scatter_x",
            )

            plot_df = clean_data_for_model[
                [x_col, target]
            ].copy()

            plot_df[x_col] = pd.to_numeric(
                plot_df[x_col],
                errors="coerce",
            )

            plot_df[target] = pd.to_numeric(
                plot_df[target],
                errors="coerce",
            )

            plot_df = plot_df.dropna()

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            sns.scatterplot(
                data=plot_df,
                x=x_col,
                y=target,
                ax=ax,
            )

            ax.set_title(
                f"{target} 与 {x_col} 的散点图"
            )

            st.pyplot(fig)
            plt.close(fig)

    elif chart_type == "因变量-自变量曲线图":
        x_candidates = [
            col
            for col in numeric_variables
            if col != target
        ]

        if not x_candidates:
            st.info(
                "没有适合绘制曲线图的数值型自变量。"
            )
        else:
            x_col = st.selectbox(
                "选择横轴自变量",
                x_candidates,
                key="curve_x",
            )

            plot_df = clean_data_for_model[
                [x_col, target]
            ].copy()

            plot_df[x_col] = pd.to_numeric(
                plot_df[x_col],
                errors="coerce",
            )

            plot_df[target] = pd.to_numeric(
                plot_df[target],
                errors="coerce",
            )

            plot_df = (
                plot_df.dropna()
                .sort_values(x_col)
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.plot(
                plot_df[x_col],
                plot_df[target],
                marker="o",
                linewidth=1.5,
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(target)
            ax.set_title(
                f"{target} 与 {x_col} 的变化曲线"
            )

            st.pyplot(fig)
            plt.close(fig)

    elif chart_type == "数值变量相关性热力图":
        if len(numeric_variables) < 2:
            st.info("至少需要两个数值型变量。")
        else:
            corr = clean_data_for_model[
                numeric_variables
            ].corr()

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax,
            )

            ax.set_title(
                "数值变量Pearson相关性热力图"
            )

            st.pyplot(fig)
            plt.close(fig)

    elif chart_type == "分类变量频数图":
        if not categorical_variables:
            st.info("当前没有分类变量。")
        else:
            selected_col = st.selectbox(
                "选择分类变量",
                categorical_variables,
                key="count_column",
            )

            count_table = (
                clean_data_for_model[selected_col]
                .astype("string")
                .value_counts(dropna=False)
                .rename_axis(selected_col)
                .reset_index(name="数量")
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            sns.barplot(
                data=count_table,
                x=selected_col,
                y="数量",
                ax=ax,
            )

            ax.tick_params(
                axis="x",
                rotation=35,
            )

            ax.set_title(
                f"{selected_col} 频数图"
            )

            st.pyplot(fig)
            plt.close(fig)


    # ============================================================
    # 十九、相关性分析
    # ============================================================

    st.subheader("7. 相关性分析")

    corr_table = correlation_table(
        clean_data_for_model,
        target,
        predictors,
        variable_types,
    )

    if corr_table.empty:
        st.info(
            "没有足够的连续型或次数型变量用于相关性分析。"
        )
    else:
        st.dataframe(
            corr_table,
            use_container_width=True,
        )

        dataframe_download(
            corr_table,
            "相关性分析.csv",
            key="download_correlation",
        )


    # ============================================================
    # 二十、模型构造和推荐
    # ============================================================

    st.subheader("8. 自动模型推荐与人工确认")

    try:
        y, X, groups, model_meta = build_model_data(
            clean_data_for_model,
            target,
            predictors,
            variable_types,
            group_col=group_col,
        )

        if len(y) <= X.shape[1]:
            st.error(
                "有效样本数不大于模型参数数量，无法稳定建立模型。"
            )
            st.stop()

        matrix_rank = np.linalg.matrix_rank(
            X.values
        )

        if matrix_rank < X.shape[1]:
            st.warning(
                "设计矩阵可能存在完全多重共线性，部分参数可能无法稳定估计。"
            )

        target_type_for_model = variable_types.get(target, "分类")

        # 统一变量类型名称，避免类型名称不一致导致模型推荐为“未识别”
        type_aliases = {
            "连续型": "连续",
            "次数型": "次数",
            "数值": "连续",
            "数值型": "连续",
            "类别": "分类",
            "分类变量": "分类",
            "时间型": "时间",
        }
        target_type_for_model = type_aliases.get(
            target_type_for_model,
            target_type_for_model,
        )

        recommended_info = detect_model_type(
            y,
            target_type_for_model,
            groups=groups,
        )

        # 对未识别类型提供兜底推荐。
        # 模型名称必须与 MODEL_OPTIONS 完全一致。
        if recommended_info.get("model_type") == "未识别":
            if target_type_for_model == "连续":
                recommended_info = {
                    "model_type": "多元线性回归",
                    "reason": "目标变量被设置为连续型变量。",
                }

            elif target_type_for_model == "次数":
                recommended_info = {
                    "model_type": "Poisson回归",
                    "reason": "目标变量被设置为次数型变量。",
                }

            elif target_type_for_model == "分类":
                unique_count = len(np.unique(y))

                if unique_count <= 2:
                    model_name = "二项Logistic回归"
                else:
                    model_name = "多项Logistic回归"

                recommended_info = {
                    "model_type": model_name,
                    "reason": "目标变量被设置为分类变量。",
                }

        recommended_model = recommended_info[
            "model_type"
        ]

        st.info(
            f"程序推荐模型：**{recommended_model}**\n\n"
            f"判断依据：{recommended_info['reason']}"
        )

        recommended_index = (
            MODEL_OPTIONS.index(
                recommended_model
            )
            if recommended_model in MODEL_OPTIONS
            else 0
        )

        final_model_type = st.selectbox(
            "请选择最终拟合模型",
            MODEL_OPTIONS,
            index=recommended_index,
            key="final_model_selector",
        )

        # 重要修复：
        # 统一使用 final_model_type，不再使用未定义的 model_type。
        model_type = final_model_type

        previous_model_type = st.session_state.get(
            "selected_model_type"
        )

        if (
            previous_model_type is not None
            and previous_model_type != final_model_type
        ):
            clear_model_session_state()

        st.session_state["selected_model_type"] = (
            final_model_type
        )

        if (
            final_model_type
            in [
                "线性混合效应模型",
                "比例型混合效应模型",
            ]
            and group_col == "无"
        ):
            st.error(
                "当前模型需要分组变量，请先选择重复观测分组变量。"
            )
            st.stop()

        vif_table = calculate_vif(X)

        st.write("多重共线性诊断")

        if vif_table.empty:
            st.info(
                "当前没有足够的自变量计算 VIF。"
            )
        else:
            st.dataframe(
                vif_table,
                use_container_width=True,
            )

            dataframe_download(
                vif_table,
                "VIF多重共线性诊断.csv",
                key="download_vif",
            )

        fit_button = st.button(
            "开始拟合最终模型",
            type="primary",
            key="fit_final_model",
        )

        if fit_button:
            is_valid, validation_message = (
                validate_model_selection(
                    y=y,
                    target_type=variable_types[target],
                    model_type=final_model_type,
                    groups=groups,
                )
            )

            if not is_valid:
                st.error(validation_message)
                st.stop()

            try:
                new_fitted_result = fit_model(
                    y,
                    X,
                    groups,
                    final_model_type,
                    robust_se=robust_se,
                )

                st.session_state[
                    "fitted_result"
                ] = new_fitted_result

                st.session_state[
                    "fitted_model"
                ] = new_fitted_result["model"]

                st.session_state[
                    "final_model_type"
                ] = final_model_type

                st.session_state[
                    "model_meta"
                ] = model_meta

                st.session_state[
                    "X_for_assumption"
                ] = X

                st.session_state[
                    "vif_table"
                ] = vif_table

                st.session_state[
                    "target_mapping"
                ] = model_meta.get(
                    "target_mapping"
                )

                st.success(
                    "模型拟合完成，结果已经保存。"
                )

            except Exception as exc:
                st.error(
                    f"模型拟合失败：{exc}"
                )

        # 重要修复：
        # 每次页面重新运行时从 session_state 恢复模型结果。
        fitted_result = st.session_state.get(
            "fitted_result"
        )

        fitted_model = st.session_state.get(
            "fitted_model"
        )

        stored_model_type = st.session_state.get(
            "final_model_type",
            final_model_type,
        )

        # --------------------------------------------------------
        # 模型结果
        # --------------------------------------------------------

        if fitted_result is None:
            st.info(
                '请点击“开始拟合最终模型”后查看模型结果。'
            )

        else:
            st.subheader("9. 模型结果")

            mapping = st.session_state.get(
                "target_mapping"
            )

            if mapping:
                st.write("分类因变量编码映射")

                mapping_table = pd.DataFrame(
                    {
                        "原始类别": list(
                            mapping.keys()
                        ),
                        "模型编码": list(
                            mapping.values()
                        ),
                    }
                )

                st.dataframe(
                    mapping_table,
                    use_container_width=True,
                )

            if model_is_converged(fitted_model):
                st.success(
                    "模型已收敛或未检测到明显收敛问题。"
                )
            else:
                st.warning(
                    "模型未收敛，当前系数、P值和预测结果不建议直接用于论文。"
                )

            metric_table = make_metric_table(
                fitted_result
            )

            st.write("模型评价指标")
            st.dataframe(
                metric_table,
                use_container_width=True,
            )

            dataframe_download(
                metric_table,
                "模型评价指标.csv",
                key="download_model_metrics",
            )

            result_table = make_result_table(
                fitted_model,
                stored_model_type,
            )

            st.write("模型系数结果")
            st.dataframe(
                result_table,
                use_container_width=True,
            )

            dataframe_download(
                result_table,
                "模型系数结果.csv",
                key="download_model_coefficients",
            )

            prediction_table = create_prediction_table(
                fitted_result
            )

            st.write(
                "实际值、预测值与残差"
            )

            st.dataframe(
                prediction_table.head(100),
                use_container_width=True,
            )

            dataframe_download(
                prediction_table,
                "实际值预测值残差.csv",
                key="download_prediction_table",
            )

            # ----------------------------------------------------
            # 训练集 / 测试集评估
            # ----------------------------------------------------

            st.subheader(
                "训练集/测试集评估"
            )

            if not use_test_set:
                st.info(
                    "当前已关闭训练集/测试集评估。"
                    "上方模型评价指标为样本内指标。"
                )

            elif stored_model_type in [
                "线性混合效应模型",
                "比例型混合效应模型",
            ]:
                st.info(
                    "混合效应模型暂未采用普通随机划分，"
                    "建议按照分组变量进行分组交叉验证。"
                )

            else:
                try:
                    if len(y) < 10:
                        st.warning(
                            "样本量少于10，训练集/测试集划分结果可能不稳定。"
                        )

                    indices = np.arange(len(y))

                    train_index, test_index = (
                        train_test_split(
                            indices,
                            test_size=test_size,
                            random_state=42,
                        )
                    )

                    y_train = y.iloc[train_index]
                    y_test = y.iloc[test_index]

                    X_train = X.iloc[train_index]
                    X_test = X.iloc[test_index]

                    test_model_result = fit_model(
                        y_train,
                        X_train,
                        None,
                        stored_model_type,
                        robust_se=robust_se,
                    )

                    test_model = test_model_result[
                        "model"
                    ]

                    if stored_model_type == (
                        "二项Logistic回归"
                    ):
                        test_probability = np.asarray(
                            test_model.predict(X_test),
                            dtype=float,
                        )

                        test_class = (
                            test_probability >= 0.5
                        ).astype(int)

                        test_metrics = [
                            "准确率",
                            "平衡准确率",
                            "精确率",
                            "召回率",
                            "F1值",
                            "Log Loss",
                        ]

                        test_values = [
                            accuracy_score(
                                y_test.astype(int),
                                test_class,
                            ),
                            balanced_accuracy_score(
                                y_test.astype(int),
                                test_class,
                            ),
                            precision_score(
                                y_test.astype(int),
                                test_class,
                                zero_division=0,
                            ),
                            recall_score(
                                y_test.astype(int),
                                test_class,
                                zero_division=0,
                            ),
                            f1_score(
                                y_test.astype(int),
                                test_class,
                                zero_division=0,
                            ),
                            log_loss(
                                y_test.astype(int),
                                test_probability,
                            ),
                        ]

                        if len(
                            np.unique(
                                y_test
                            )
                        ) == 2:
                            test_metrics.append(
                                "ROC-AUC"
                            )
                            test_values.append(
                                roc_auc_score(
                                    y_test,
                                    test_probability,
                                )
                            )

                    elif stored_model_type == (
                        "多项Logistic回归"
                    ):
                        probability = np.asarray(
                            test_model.predict(X_test)
                        )

                        test_class = (
                            probability.argmax(axis=1)
                        )

                        test_metrics = [
                            "准确率",
                            "平衡准确率",
                            "宏平均F1",
                        ]

                        test_values = [
                            accuracy_score(
                                y_test.astype(int),
                                test_class,
                            ),
                            balanced_accuracy_score(
                                y_test.astype(int),
                                test_class,
                            ),
                            f1_score(
                                y_test.astype(int),
                                test_class,
                                average="macro",
                                zero_division=0,
                            ),
                        ]

                    else:
                        test_prediction = np.asarray(
                            test_model.predict(X_test),
                            dtype=float,
                        )

                        test_metrics = [
                            "RMSE",
                            "MAE",
                            "R²",
                        ]

                        test_values = [
                            np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    test_prediction,
                                )
                            ),
                            mean_absolute_error(
                                y_test,
                                test_prediction,
                            ),
                            r2_score(
                                y_test,
                                test_prediction,
                            ),
                        ]

                    test_metric_table = pd.DataFrame(
                        {
                            "测试集指标": test_metrics,
                            "数值": test_values,
                        }
                    )

                    st.dataframe(
                        test_metric_table,
                        use_container_width=True,
                    )

                    dataframe_download(
                        test_metric_table,
                        "测试集评价指标.csv",
                        key="download_test_metrics",
                    )

                except Exception as exc:
                    st.warning(
                        f"测试集评估失败：{exc}"
                    )

            # ----------------------------------------------------
            # 线性模型诊断
            # ----------------------------------------------------

            if stored_model_type in [
                "多元线性回归",
                "Logit变换线性回归",
            ]:
                st.subheader("线性模型诊断")

                diagnostic_table = (
                    make_diagnostic_table(
                        fitted_model
                    )
                )

                st.dataframe(
                    diagnostic_table,
                    use_container_width=True,
                )

                dataframe_download(
                    diagnostic_table,
                    "线性模型诊断.csv",
                    key="download_diagnostic_table",
                )

                residuals = np.asarray(
                    fitted_model.resid,
                    dtype=float,
                )

                fitted_values = np.asarray(
                    fitted_model.fittedvalues,
                    dtype=float,
                )

                diag_col1, diag_col2 = st.columns(2)

                with diag_col1:
                    fig_residual, ax_residual = (
                        plt.subplots(
                            figsize=(7, 5)
                        )
                    )

                    sns.scatterplot(
                        x=fitted_values,
                        y=residuals,
                        ax=ax_residual,
                    )

                    ax_residual.axhline(
                        0,
                        color="red",
                        linestyle="--",
                    )

                    ax_residual.set_xlabel(
                        "拟合值"
                    )
                    ax_residual.set_ylabel(
                        "残差"
                    )
                    ax_residual.set_title(
                        "残差-拟合值图"
                    )

                    st.pyplot(fig_residual)
                    plt.close(fig_residual)

                with diag_col2:
                    fig_qq, ax_qq = plt.subplots(
                        figsize=(7, 5)
                    )

                    probplot(
                        residuals,
                        dist="norm",
                        plot=ax_qq,
                    )

                    ax_qq.set_title(
                        "残差正态QQ图"
                    )

                    st.pyplot(fig_qq)
                    plt.close(fig_qq)

                try:
                    influence = OLSInfluence(
                        fitted_model
                    )

                    cooks_distance = (
                        influence.cooks_distance[0]
                    )

                    cooks_table = pd.DataFrame(
                        {
                            "样本序号": np.arange(
                                len(cooks_distance)
                            ),
                            "Cook距离": cooks_distance,
                        }
                    ).sort_values(
                        "Cook距离",
                        ascending=False,
                    )

                    st.write(
                        "Cook距离较大的观测"
                    )

                    st.dataframe(
                        cooks_table.head(20),
                        use_container_width=True,
                    )

                    dataframe_download(
                        cooks_table,
                        "Cook距离.csv",
                        key="download_cooks_distance",
                    )

                except Exception as exc:
                    st.info(
                        f"Cook距离计算失败：{exc}"
                    )

            # ----------------------------------------------------
            # 分类模型诊断
            # ----------------------------------------------------

            if stored_model_type == (
                "二项Logistic回归"
            ):
                st.subheader(
                    "二分类模型诊断"
                )

                cm = confusion_matrix(
                    prediction_table[
                        "实际值"
                    ].astype(int),
                    prediction_table[
                        "预测类别"
                    ].astype(int),
                )

                cm_index = [
                    f"真实{i}"
                    for i in range(cm.shape[0])
                ]

                cm_columns = [
                    f"预测{i}"
                    for i in range(cm.shape[1])
                ]

                cm_table = pd.DataFrame(
                    cm,
                    index=cm_index,
                    columns=cm_columns,
                )

                st.write("混淆矩阵")
                st.dataframe(
                    cm_table,
                    use_container_width=True,
                )

                dataframe_download(
                    cm_table.reset_index(),
                    "二分类混淆矩阵.csv",
                    key="download_confusion_matrix",
                )

            # ----------------------------------------------------
            # 模型假设
            # ----------------------------------------------------

            st.subheader("10. 模型假设")

            assumptions = generate_assumptions(
                target,
                predictors,
                variable_types,
                stored_model_type,
                vif_table=st.session_state.get(
                    "vif_table",
                    vif_table,
                ),
                group_col=group_col,
            )

            assumptions_text = "\n".join(
                [
                    f"{index + 1}. {text}"
                    for index, text in enumerate(
                        assumptions
                    )
                ]
            )

            st.text_area(
                "可直接复制到论文的模型假设",
                assumptions_text,
                height=350,
                key="assumptions_text_area",
            )

            # ----------------------------------------------------
            # 论文表述草稿
            # ----------------------------------------------------

            st.subheader("11. 论文表述草稿")

            paper_text = f"""
    本文首先对原始数据进行完整性、一致性和变量类型检查。
    原始数据包含{original_shape[0]}条样本和{original_shape[1]}个变量。
    针对缺失数据，本文采用“{missing_method}”方法进行处理，
    共插补{imputed_cells}个缺失单元格，
    并因缺失值删除{deleted_target_missing_rows + deleted_missing_rows}条样本。
    针对异常数据，本文采用“{outlier_method}”方法进行识别，
    共检测到{outlier_row_count}条异常样本，
    并根据处理策略删除{deleted_outlier_rows}条异常样本。
    本文将“{target}”作为因变量，
    将{", ".join(predictors)}作为解释变量。
    """.strip()

            if group_col != "无":
                paper_text += (
                    f'考虑到同一“{group_col}”下可能存在多次观测，'
                    "本文将其作为重复观测的分组变量。"
                )

            paper_text += (
                f"在综合考虑因变量类型、数据取值范围和观测结构后，"
                f'最终采用“{stored_model_type}”进行建模。'
                f"清洗后最终保留{clean_data_for_model.shape[0]}条有效样本，"
                "并据此开展后续统计分析和模型估计。"
            )

            st.text_area(
                "论文表述草稿",
                paper_text,
                height=300,
                key="paper_text_area",
            )

    except Exception as exc:
        st.error(
            f"模型数据构造或分析过程中发生错误：{exc}"
        )


    # ============================================================
    # 二十一、使用注意事项
    # ============================================================

    with st.expander(
        "使用和解释模型时的注意事项"
    ):
        st.markdown(
            """
    ### 1. 自动推荐不是最终结论

    程序根据变量类型和数据结构进行初步推荐，
    最终模型应结合题目目标、变量含义和模型诊断结果确定。

    ### 2. 相关性不等于因果关系

    Pearson/Spearman 相关系数和回归系数只能反映统计关联，
    不能单独证明因果关系。

    ### 3. 异常值不一定是错误数据

    异常值可能是真实存在的极端情况。
    正式论文中删除异常数据时，应说明识别方法和删除依据。

    ### 4. R²不能单独判断模型好坏

    应结合调整R²、AIC/BIC、残差图、测试集误差、
    显著性检验和实际解释意义共同评价。

    ### 5. 分类模型不要只看准确率

    如果类别分布不平衡，还应重点查看平衡准确率、
    精确率、召回率、F1值和ROC-AUC。

    ### 6. 混合效应模型需要真实的重复观测结构

    不能仅因为某列取值重复就直接使用混合效应模型。
    分组变量应确实代表同一对象、地区、企业或其他层级单位的重复观测。

    ### 7. 测试集划分需要结合数据结构

    如果数据具有时间顺序，建议使用时间序列划分；
    如果数据具有分组结构，建议按照组进行训练集和测试集划分，
    避免同一对象同时出现在训练集和测试集中。
    """
        )


else:  # 优化求解模块
    import numpy as np
    from scipy.optimize import linprog, milp, minimize, LinearConstraint, Bounds

    st.header("📊 优化问题求解器")

    if "opt_uploaded_file" not in st.session_state:
        st.session_state.opt_uploaded_file = None
    opt_upload = st.sidebar.file_uploader("上传数据表（优化用）", type=["csv", "xlsx"], key="opt_data_upload")
    if opt_upload is not None:
        st.session_state.opt_uploaded_file = opt_upload

    if st.session_state.opt_uploaded_file is None:
        st.info("请先在侧边栏上传数据表，然后使用优化求解功能。")
        st.stop()

    try:
        uploaded_file = st.session_state.opt_uploaded_file
        uploaded_file.seek(0)
        if uploaded_file.name.endswith(".csv"):
            opt_df = pd.read_csv(uploaded_file)
        else:
            opt_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        st.stop()

    opt_df.columns = make_unique_columns(opt_df.columns)
    st.success(f"已加载数据：{opt_df.shape[0]} 行, {opt_df.shape[1]} 列")

    # 优化类型
    opt_type = st.selectbox("选择优化类型", ["线性规划 (LP)", "整数线性规划 (ILP)", "0-1 规划"])

    # 决策变量
    # 自动找出所有“可以转为数字”的列（只要超过一半值能转成数字就算）
    col_list = []
    for col in opt_df.columns:
        try:
            numeric_part = pd.to_numeric(opt_df[col], errors='coerce')
            if numeric_part.notna().mean() > 0.5:  # 超过50%的值能转成数字
                col_list.append(col)
        except Exception:
            pass

    if not col_list:
        st.error("数据表中没有可转换为数值的列，请检查数据格式。")
        st.stop()

    var_cols = st.multiselect("选择决策变量所在的列（可多选）", col_list)
    if not var_cols:
        st.warning("请至少选择一个决策变量列。")
        st.stop()
    n_vars = len(var_cols)

    # 目标函数系数：使用数据列的第一行作为默认系数，也可手动修改
    st.subheader("目标函数系数")
    obj_coeffs = []
    for c in var_cols:
        # 安全读取第一个有效数值
        num_series = pd.to_numeric(opt_df[c], errors='coerce').dropna()
        default_val = float(num_series.iloc[0]) if len(num_series) > 0 else 0.0
        coeff = st.number_input(f"系数 {c}", value=default_val, format="%.4f")
        obj_coeffs.append(coeff)
    maximize = st.checkbox("最大化目标")

    # 约束条件
    st.subheader("约束条件")
    if "cons_list" not in st.session_state:
        st.session_state.cons_list = []
    with st.form("add_constraint"):
        coeff_str = st.text_input("系数（逗号分隔）", "1,1")
        sign = st.selectbox("关系", ["<=", "=", ">="])
        rhs = st.number_input("右侧常数", value=1.0)
        if st.form_submit_button("添加约束"):
            try:
                coeffs = [float(x) for x in coeff_str.split(",")]
                if len(coeffs) != n_vars:
                    st.error(f"系数个数应为 {n_vars}")
                else:
                    st.session_state.cons_list.append((coeffs, sign, rhs))
            except:
                st.error("系数格式错误")
    # 显示已添加约束
    for i, (coeffs, sign, rhs) in enumerate(st.session_state.cons_list):
        expr = " + ".join([f"{c}*{var_cols[j]}" for j, c in enumerate(coeffs)])
        st.write(f"约束 {i+1}: {expr} {sign} {rhs}")
    if st.button("清空所有约束"):
        st.session_state.cons_list = []

    # 变量边界
    st.subheader("变量边界")
    use_bounds = st.checkbox("自定义边界（默认 >=0）")
    bounds = [(0, None) for _ in range(n_vars)]
    if use_bounds:
        for i, col in enumerate(var_cols):
            c1, c2 = st.columns(2)
            lo = c1.number_input(f"{col} 下界", value=0.0)
            hi = c2.number_input(f"{col} 上界", value=100.0)
            bounds[i] = (lo, hi)

    # 整数约束
    integrality = None
    if "整数" in opt_type:
        int_vars = st.multiselect("整数变量", var_cols)
        integrality = [1 if col in int_vars else 0 for col in var_cols]
    if opt_type == "0-1 规划":
        integrality = [2] * n_vars

    # 求解
    if st.button("🚀 求解", type="primary"):
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for coeffs, sign, rhs in st.session_state.cons_list:
            if sign == "<=":
                A_ub.append(coeffs)
                b_ub.append(rhs)
            elif sign == ">=":
                A_ub.append([-c for c in coeffs])
                b_ub.append(-rhs)
            else:
                A_eq.append(coeffs)
                b_eq.append(rhs)

        c = np.array(obj_coeffs)
        if maximize:
            c = -c

        # 转换为 numpy 数组
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None

        try:
            if integrality and any(i > 0 for i in integrality):
                from scipy.optimize import milp, LinearConstraint, Bounds
                cons = []
                if A_ub is not None:
                    cons.append(LinearConstraint(A_ub, -np.inf, b_ub))
                if A_eq is not None:
                    cons.append(LinearConstraint(A_eq, b_eq, b_eq))
                lb = [b[0] if b[0] is not None else -np.inf for b in bounds]
                ub = [b[1] if b[1] is not None else np.inf for b in bounds]
                res = milp(c=c, constraints=cons, bounds=Bounds(lb, ub), integrality=integrality)
            else:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if res.success:
                opt_x = res.x
                opt_val = -res.fun if maximize else res.fun
                st.success("✅ 求解成功！")
                st.write("**最优解：**")
                st.json({var_cols[i]: float(opt_x[i]) for i in range(n_vars)})
                st.write(f"**最优值：** {opt_val:.6f}")

                # 简单敏感性分析
                st.subheader("影子价格近似")
                for idx, (coeffs, sign, rhs) in enumerate(st.session_state.cons_list):
                    delta = 0.01 * max(abs(rhs), 1.0)
                    # 重新求解扰动后问题
                    t_A_ub, t_b_ub, t_A_eq, t_b_eq = [], [], [], []
                    for j, (coeffs_j, sign_j, rhs_j) in enumerate(st.session_state.cons_list):
                        if j == idx:
                            rhs_j += delta
                        if sign_j == "<=":
                            t_A_ub.append(coeffs_j); t_b_ub.append(rhs_j)
                        elif sign_j == ">=":
                            t_A_ub.append([-c for c in coeffs_j]); t_b_ub.append(-rhs_j)
                        else:
                            t_A_eq.append(coeffs_j); t_b_eq.append(rhs_j)
                    try:
                        t_A_ub = np.array(t_A_ub) if t_A_ub else None
                        t_b_ub = np.array(t_b_ub) if t_b_ub else None
                        t_A_eq = np.array(t_A_eq) if t_A_eq else None
                        t_b_eq = np.array(t_b_eq) if t_b_eq else None
                        res2 = linprog(c, A_ub=t_A_ub, b_ub=t_b_ub, A_eq=t_A_eq, b_eq=t_b_eq, bounds=bounds, method='highs')
                        if res2.success:
                            pval = -res2.fun if maximize else res2.fun
                            shadow = (pval - opt_val) / delta
                            st.write(f"约束 {idx+1}: 影子价格 ≈ {shadow:.6f}")
                    except:
                        pass
            else:
                st.error(f"求解失败：{res.message}")
        except Exception as e:
            st.error(f"求解错误：{e}")

    # 以下优化代码结束


# ==================== 建模彩蛋模块 ====================
# 本模块只负责显示趣味内容，不参与任何数据处理、模型拟合和结果计算。
# 如果不想显示，直接删除本模块即可。

try:
    import hashlib

    EASTER_EGG_QUOTES = [
        "模型不会自动变好，但数据清洗后通常会更诚实。",
        "相关性可以一起散步，但不代表它们互相导致。",
        "先看数据，再选模型；不要让模型替题目做决定。",
        "一个好的评价体系，应该既能排序，也能解释为什么这样排序。",
        "当结果特别完美时，先检查数据泄漏。",
        "异常值不一定是错误，也可能是数据正在认真讲故事。",
        "R²很高值得开心，但测试集表现更值得相信。",
        "数学建模的第一步不是写公式，而是确认变量到底是什么意思。",
        "如果所有指标权重都一样，至少要诚实地说这是等权重。",
        "稳定的结论，比漂亮的结论更有价值。",
    ]

    EASTER_EGG_SIGNS = [
        "今天适合检查一遍变量含义。",
        "今天适合看看残差图。",
        "今天适合重新确认正向指标和负向指标。",
        "今天适合做一次敏感性分析。",
        "今天适合给模型加一句合理的解释。",
        "今天适合怀疑一下过高的准确率。",
        "今天适合保存一份中间结果。",
        "今天适合把代码注释写清楚。",
    ]

    now = datetime.now()

    # 使用日期生成稳定索引。
    # 同一天刷新页面时内容不变，不会影响正常运行。
    date_code = now.strftime("%Y-%m-%d")
    hash_value = int(
        hashlib.md5(
            date_code.encode("utf-8")
        ).hexdigest(),
        16
    )

    quote = EASTER_EGG_QUOTES[
        hash_value % len(EASTER_EGG_QUOTES)
    ]

    today_sign = EASTER_EGG_SIGNS[
        hash_value % len(EASTER_EGG_SIGNS)
    ]

    with st.expander(
        "🔐 发现一个隐藏的建模彩蛋",
        expanded=False
    ):
        st.markdown(
            f"""
            <div style="
                padding: 16px;
                border-radius: 12px;
                background: linear-gradient(
                    135deg,
                    rgba(70,130,180,0.12),
                    rgba(120,200,160,0.12)
                );
                border: 1px solid rgba(100,140,180,0.25);
                margin-bottom: 10px;
            ">
                <h4>📐 今日建模语录</h4>
                <p style="font-size: 18px;">
                    “{quote}”
                </p>
                <p style="color: #777;">
                    当前时间：{now.strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🎲 抽取今日建模签",
            key="easter_egg_sign_button"
        ):
            st.success(
                f"今日建模签：{today_sign}"
            )

        st.caption(
            "这个彩蛋不会修改数据、模型参数或分析结果。"
        )

except Exception:
    # 彩蛋出现任何问题时静默跳过，
    # 确保不会影响主程序正常运行。
    pass

# ==================== 建模彩蛋模块结束 ====================


