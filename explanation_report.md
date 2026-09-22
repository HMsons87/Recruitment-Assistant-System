# تقرير الشرح الشامل والمفصل لنوت بوك مشروع نظام التوظيف الذكي
## (Smart Recruitment Assistant System - Notebook Walkthrough)

> **ملاحظة تمهيدية:**  
> تم إعداد هذا التقرير ليشرح ملف النوت بوك (`notebooks/smart_recruitment_system.ipynb`) خلية بخلية من البداية وحتى النهاية، بالعامية المصرية المبسطة وبأسلوب أكاديمي وعملي سلس، ليجعلك فاهماً لكل سطر كود، وكل رقم طلع في المخرجات (Outputs)، وكل قرار تصميمي اتخذناه، بحيث تكون جاهزاً تماماً بنسبة 100% لأي سؤال يسأله البشمهندس أو لجنة المناقشة.

---

## فهرس المحتويات
1. [نظرة عامة على المشروع وفكرة النوت بوك](#نظرة-عامة-على-المشروع-وفكرة-النوت-بوك)
2. [شرح الخلايا بالتفصيل (خلية بخلية)](#شرح-الخلايا-بالتفصيل)
   - [الخلية 1 و 2: استيراد المكتبات الأساسية (Import Libraries)](#الخلية-1-و-2-استيراد-المكتبات-الأساسية)
   - [الخلية 3 و 4: تحميل واستكشاف البيانات وربط المسارات (Load & Explore Dataset)](#الخلية-3-و-4-تحميل-واستكشاف-البيانات-وربط-المسارات)
   - [الخلية 5 و 6: فحص القيم المفقودة وتوزيع الهدف (Missing Values & Target Distribution)](#الخلية-5-و-6-فحص-القيم-المفقودة-وتوزيع-الهدف)
   - [الخلية 7 و 8: المعالجة وهندسة الميزات (Data Preprocessing & Feature Engineering)](#الخلية-7-و-8-المعالجة-وهندسة-الميزات)
   - [الخلية 9 و 10: التحليل الاستكشافي للميزات المهندسة (EDA)](#الخلية-9-و-10-التحليل-الاستكشافي-للميزات-المهندسة)
   - [الخلية 11 و 12: التحقق من أبعاد الميزات والهدف (Define Features & Target)](#الخلية-11-و-12-التحقق-من-أبعاد-الميزات-والهدف)
   - [الخلية 13 و 14: تقسيم البيانات تدريب واختبار (Train-Test Split)](#الخلية-13-و-14-تقسيم-البيانات-تدريب-واختبار)
   - [الخلية 15 و 16: تدريب نموذج الانحدار اللوجستي (Logistic Regression Pipeline)](#الخلية-15-و-16-تدريب-نموذج-الانحدار-اللوجستي)
   - [الخلية 17 و 18: مصفوفة الارتباك للانحدار اللوجستي (Confusion Matrix Heatmap)](#الخلية-17-و-18-مصفوفة-الارتباك-للانحدار-اللوجستي)
   - [الخلية 19 و 20: تدريب نموذج الغابة العشوائية (Random Forest Classifier)](#الخلية-19-و-20-تدريب-نموذج-الغابة-العشوائية)
   - [الخلية 21 و 22: مصفوفة الارتباك للغابة العشوائية (Confusion Matrix Heatmap)](#الخلية-21-و-22-مصفوفة-الارتباك-للغابة-العشوائية)
   - [الخلية 23 و 24: تقييم النماذج بمقاييس التصنيف (Evaluate Classification Reports)](#الخلية-23-و-24-تقييم-النماذج-بمقاييس-التصنيف)
   - [الخلية 25 و 26: مقارنة منحنيات الـ ROC (Compare ROC Curves)](#الخلية-25-و-26-مقارنة-منحنيات-الـ-roc)
   - [الخلية 27 و 28: أهمية الميزات في الغابة العشوائية (Feature Importance)](#الخلية-27-و-28-أهمية-الميزات-في-الغابة-العشوائية)
   - [الخلية 29 و 30: ملخص ومقارنة النماذج مع خط الأساس (Models Benchmark Summary)](#الخلية-29-و-30-ملخص-ومقارنة-النماذج-مع-خط-الأساس)
   - [الخلية 31 و 32: ترتيب أفضل 10 مرشحين (Top 10 Candidate Ranking)](#الخلية-31-و-32-ترتيب-أفضل-10-مرشحين)
   - [الخلية 33 و 34: حفظ النماذج ومخرجات المشروع (Save Model Artifacts)](#الخلية-33-و-34-حفظ-النماذج-ومخرجات-المشروع)
3. [الملخص الشامل وأهم أسئلة المناقشة المتوقعة (Interview & Defense Cheatsheet)](#الملخص-الشامل-وأهم-أسئلة-المناقشة-المتوقعة)

---

## نظرة عامة على المشروع وفكرة النوت بوك

### ما هي المشكلة التي يحلها المشروع؟
شركات التوظيف والشركات التقنية الكبيرة بتصرف مبالغ ضخمة ووقت كبير في تدريب موظفين أو البحث عن كفاءات. المشكلة هنا: **مين من المرشحين (Candidates) اللي بياخدوا كورسات تدريبية عنده نية حقيقية يغير وظيفته وينضم للشركة (Looking for a job change = 1)، ومين بياخد الكورس لمجرد الاستفادة ومش ناوي يسيب شغله الحالي (Not looking = 0)؟**

### دور النوت بوك في منظومة المشروع الكلية:
هذا النوت بوك هو **قلب الـ Machine Learning** في المشروع:
1. بيقرأ البيانات الخام من `data/raw/aug_train.csv`.
2. بيستدعي دوال المعالجة المتقدمة وهندسة الميزات من ملف بايثون خارجي منظم (`src/preprocessing.py`).
3. بيعمل تحليل استكشافي بصري (EDA).
4. بيبني خطوط معالجة وتدريب كاملة (Pipelines) باستخدام `src/modeling.py`.
5. بيدرب نموذجين مختلفين: **Logistic Regression** (كنموذج خطي وسريع وقابل للتفسير) و **Random Forest** (كنموذج تجميعي قوي وغير خطي Ensembled Trees).
6. بيقارن بينهم بمقاييس دقيقة تراعي عدم توازن البيانات (Imbalance) زي ROC-AUC و F1-Score والـ Baseline.
7. بيرتب المرشحين بحسب احتمالية رغبتهم في الانتقال لمساعدة مسؤولي الـ HR على اتخاذ قرارات سريعة ومبنية على بيانات.
8. بيحفظ النماذج والملفات المنظفة في مجلدات `models/` و `outputs/` و `data/processed/` عشان الـ Backend (FastAPI) والـ Frontend (Streamlit) يقدرو يستخدموها مباشرة.

---

## شرح الخلايا بالتفصيل

---

### الخلية 1 و 2: استيراد المكتبات الأساسية
#### الكود:
```python
import os
import sys
import json
import joblib
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Plotting style
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `import os`: استيراد مكتبة التعامل مع نظام التشغيل (مسارات، بيئة العمل).
2. `import sys`: استيراد مكتبة النظام في بايثون للتحكم في مسارات البحث عن الحزم (`sys.path`).
3. `import json`: مكتبة لقراءة وكتابة ملفات JSON (هنحتاجها لحفظ بيانات الموديل الوصفية Metadata).
4. `import joblib`: مكتبة متخصصة وسريعة جداً في حفظ واسترجاع موديلات الـ Machine Learning وسلاسل الـ Pipelines من وإلى القرص الصلب بصيغة `.pkl`.
5. `from pathlib import Path`: كائن حديث ومحترم للتعامل مع مسارات الملفات بطريقة Cross-platform (تشتغل على ويندوز ولينكس وماك بدون مشاكل الـ slashes).
6. `import pandas as pd`: مكتبة الجداول والبيانات الأساسية لتحميل وتعديل البيانات في هيئة DataFrames.
7. `import numpy as np`: مكتبة العمليات الحسابية والمصفوفات والتعامل مع القيم الرقمية والـ NaN.
8. `import matplotlib.pyplot as plt`: مكتبة الرسم البياني الأساسية لرسم المنحنيات والمخططات وتخصيصها.
9. `import seaborn as sns`: مكتبة رسم إحصائي مبنية فوق matplotlib بتدي رسومات احترافية وجذابة زي الـ Heatmaps والـ KDE plots.
10. `from sklearn.model_selection import train_test_split`: دالة لتقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار.
11. `from sklearn.linear_model import LogisticRegression`: استيراد كلاس نموذج الانحدار اللوجستي للتصنيف الثنائي.
12. `from sklearn.ensemble import RandomForestClassifier`: استيراد كلاس نموذج الغابة العشوائية المبني على أشجار القرار المتعددة.
13. `from sklearn.metrics import (...)`: استيراد مقاييس تقييم الأداء:
    - `classification_report`: تقرير شامل يضم الدقة، الاستدعاء، و F1-Score لكل كلاس.
    - `confusion_matrix`: مصفوفة المقارنة بين التوقعات الفعلية والتوقعات المتنبأ بها.
    - `roc_auc_score`: حساب المساحة تحت منحنى خصائص التشغيل للمستقبل.
    - `roc_curve`: إحداثيات رسم منحنى الـ ROC (معدل الإيجابيات الحقيقية مقابل الكاذبة).
    - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`: دوال حساب كل مقياس على حدة.
14. `sns.set_theme(style="whitegrid")`: تظبيط شكل وخلفية الرسومات البيانية لتكون بيضاء مخططة بشبكة رمادية واضحة ومريحة للعين.
15. `RANDOM_STATE = 42`: تثبيت رقم عشوائي عام لضمان تكرارية النتائج (Reproducibility) في كل مرة يشتغل فيها الكود.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- تجميع كل الـ dependencies في مكان واحد في أول النوت بوك هو Clean Code Best Practice.
- تثبيت `RANDOM_STATE = 42` يضمن إن لو البشمهندس شغل النوت بوك عنده تطلعله نفس الأرقام بالظبط بدون أي اختلاف.

#### ج) الاستفادة العملية:
- تجهيز بيئة العمل بكل الأدوات الرياضية والإحصائية ومكتبات النمذجة والرسم لحفظ الوقت وتجنب استدعاء مكتبات في منتصف الكود.

#### د) المخرجات (Output):
- لا يوجد مخرج مرئي (خلية تهيئة وتجهيز).

#### هـ) الربط الخارجي:
- مرتبطة بمكتبات بيئة بايثون الافتراضية، وبتجهز المتغيرات العامة لمسار المشروع لاحقاً.

---

### الخلية 3 و 4: تحميل واستكشاف البيانات وربط المسارات
#### الكود:
```python
CURRENT_DIR = Path.cwd()
PROJECT_ROOT = (
    CURRENT_DIR.parent
    if CURRENT_DIR.name.lower() == "notebooks"
    else CURRENT_DIR
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

for folder in [PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

possible_paths = [
    PROJECT_ROOT / "data" / "raw" / "aug_train.csv",
    PROJECT_ROOT / "data" / "aug_train.csv",
    PROJECT_ROOT / "aug_train.csv",
    CURRENT_DIR / "aug_train.csv",
]

DATA_PATH = None
for path in possible_paths:
    if path.exists():
        DATA_PATH = path
        break

if DATA_PATH is None:
    found_files = list(PROJECT_ROOT.rglob("aug_train.csv"))
    if found_files:
        DATA_PATH = found_files[0]

if DATA_PATH is None or not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find 'aug_train.csv' inside: {PROJECT_ROOT}\n"
        f"Make sure the file is located inside data/raw/."
    )

print(f"Found dataset at: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Add project root to sys.path to import from src
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import add_engineered_features, MODEL_EXCLUDED_COLUMNS
from src.modeling import RANDOM_STATE, build_models, calculate_metrics

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.head()
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `CURRENT_DIR = Path.cwd()`: بيحدد المسار الحالي اللي شغال منه الكود دلوقتي (لو فاتح النوت بوك من جوه فولدر notebooks).
2. `PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.lower() == "notebooks" else CURRENT_DIR`: حركة ذكية جداً؛ لو المسار الحالي هو فولدر `notebooks` بيرجع خطوة لورا للمجلد الرئيسي للمشروع، ولو شغال من الـ Root يفضل في الـ Root.
3. تعريف مسارات الحفظ الرئيسية:
   - `PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"` (للبيانات المنظفة).
   - `MODELS_DIR = PROJECT_ROOT / "models"` (للموديلات المدربة).
   - `FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"` (للرسومات والمخططات).
   - `RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"` (للملفات الإحصائية والجداول).
4. `for folder in [...]: folder.mkdir(parents=True, exist_ok=True)`: بيلف على الفولدرات دي، ولو مش موجودة بينشئها تلقائياً بدون ما يرمي Error.
5. قائمة `possible_paths`: لستة بأشهر الأماكن المحتملة لملف البيانات `aug_train.csv` لتفادي مشاكل اختلاف بيئات التشغيل.
6. حلقة `for path in possible_paths`: بتدور على الملف، وأول ما تلاقيه بتخزنه في `DATA_PATH` وتعمل `break`.
7. `if DATA_PATH is None: ... rglob`: بحث عميق احتياطي في كامل المجلد إذا ما كانش في المسارات المباشرة.
8. رفع خطأ `FileNotFoundError` لو الملف مش موجود خالص مع رسالة واضحة للمستخدم.
9. `print(f"Found dataset at: {DATA_PATH}")`: طباعة مسار الملف اللي تم العثور عليه.
10. `df = pd.read_csv(DATA_PATH)`: قراءة ملف الـ CSV وتخزينه في جدول بيانات `df`.
11. `sys.path.insert(0, str(PROJECT_ROOT))`: إضافة مسار المشروع لقائمة مسارات بايثون عشان نقدر نعمل `import` من فولدر `src`.
12. `from src.preprocessing import add_engineered_features, MODEL_EXCLUDED_COLUMNS`: استيراد دالة هندسة الميزات وقائمة الأعمدة المستبعدة أخلاقياً وفنياً من ملف المعالجة.
13. `from src.modeling import RANDOM_STATE, build_models, calculate_metrics`: استيراد دالة بناء النماذج ودالة حساب المقاييس من ملف الموديلنج.
14. `print("Dataset Shape:", df.shape)`: طباعة أبعاد الداتاسيت (عدد الصفوف والأعمدة).
15. `print("Columns:", df.columns.tolist())`: طباعة قائمة أسماء كل الأعمدة.
16. `df.head()`: عرض أول 5 صفوف من الداتاسيت كمعاينة بصرية.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- **Dynamic Path Resolution**: تجنب كتابة مسارات ثابتة (Hardcoded Paths مثل `C:\Users\...`) عشان الكود يشتغل على أي جهاز تفتحه عليه بدون أي تعديل.
- **Modular Code**: فصل منطق المعالجة في موديولات خارجية (`src/`) واستدعاؤها في النوت بوك يثبت لبشمهندس المناقشة إن الكود مكتوب بأسلوب Software Engineering منظم ومش كود نوت بوك عشوائي.

#### ج) الاستفادة العملية:
- التأكد من إنشاء بنية الفولدرات، تحميل الداتاسيت الأصلية بأمان، والتأكد من إمكانية استدعاء كود المشروع من مجلد `src`.

#### د) تفسير الـ Output:
- **المسار المطبوع**:
  `Found dataset at: ...\data\raw\aug_train.csv` (تأكيد على إيجاد الداتاسيت الأصلية).
- **أبعاد الداتاسيت (Dataset Shape)**:
  `(19158, 14)`:
  - رقم **19,158**: ده إجمالي عدد المرشحين (صفوف/عينات التدريب).
  - رقم **14**: عدد الأعمدة الأصلية (13 ميزة + عمود الهدف `target`).
- **قائمة الأعمدة الـ 14**:
  `enrollee_id`, `city`, `city_development_index`, `gender`, `relevent_experience`, `enrolled_university`, `education_level`, `major_discipline`, `experience`, `company_size`, `company_type`, `last_new_job`, `training_hours`, `target`.
- **معاينة `df.head()`**:
  أول 5 صفوف بتوضح وجود قيم مفقودة واضحة (زي `NaN` في أعمدة `gender`, `company_size`, `company_type`)، ونصوص متباينة، وقيم فئوية محتاجة معالجة لاحقة.

#### هـ) الربط الخارجي:
- **الملف المقروء**: `data/raw/aug_train.csv`.
- **الفولدرات المنشأة**: `models/`, `outputs/figures/`, `outputs/results/`, `data/processed/`.
- **الملفات المستوردة**: `src/preprocessing.py` و `src/modeling.py`.

---

### الخلية 5 و 6: فحص القيم المفقودة وتوزيع الهدف
#### الكود:
```python
# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Check target distribution
print("\nTarget Distribution:")
print(df["target"].value_counts())
print("\nTarget Percentage (%):")
print((df["target"].value_counts(normalize=True) * 100).round(2))
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `print("Missing values in each column:")`: طباعة عنوان توضيحي لفحص القيم المفقودة.
2. `print(df.isnull().sum())`:
   - `df.isnull()`: بتفحص كل خلية، وبترجع `True` لو القيمة مفقودة و `False` لو موجودة.
   - `.sum()`: بتجمع الـ `True` لكل عمود، فيظهر عدد القيم المفقودة في كل ميزة.
3. `print("\nTarget Distribution:")`: طباعة عنوان لتوزيع عمود الهدف.
4. `print(df["target"].value_counts())`: حساب تكرار كل فئة من فئات الهدف (كم واحد `0` وكم واحد `1`).
5. `print("\nTarget Percentage (%):")`: طباعة عنوان للنسبة المئوية للفئات.
6. `print((df["target"].value_counts(normalize=True) * 100).round(2))`:
   - `normalize=True`: بتحسب النسبة الكسرية لكل فئة منسوبة للمجموع الكلي.
   - `* 100`: تحويلها لنسبة مئوية.
   - `.round(2)`: تقريب الناتج لرقمين عشريين.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- فحص سلامة البيانات واكتشاف أمرين حاسمين في علم البيانات:
  1. **Missing Data Mechanism**: تحديد الأعمدة اللي فيها نواقص عشان نحدد استراتيجية التعويض (Imputation) المناسبة.
  2. **Class Imbalance**: معرفة هل الفئات متوازنة أم غير متوازنة، لأن عدم توازن البيانات يؤثر جذرياً على اختيار الخوارزميات ومقاييس التقييم.

#### ج) الاستفادة العملية:
- اكتشفنا إن معلومات الشركات فيها فاقد عالي جداً (أكثر من 30% مفقود في حجم ونوع الشركة)، وإن الداتاسيت غير متوازنة بنسبة 3 إلى 1 (75% لا يبحثون مقابل 25% يبحثون)، وده اللي خلانا في الموديل نستخدم `class_weight='balanced'` ونركز على F1-Score و ROC-AUC بدل الدقة الساذجة (Accuracy).

#### د) تفسير الـ Output:
1. **جدول القيم المفقودة**:
   - `enrollee_id`: 0 (كل مرشح له رقم تعريفي).
   - `city`: 0 (المدينة معروفة للكل).
   - `city_development_index`: 0 (مؤشر التنمية موجود للجميع وهو رقم بين 0 و 1).
   - `gender`: **4,508** قيمة مفقودة (~23.5% مفقود).
   - `relevent_experience`: 0 (الخبرة ذات الصلة مكتملة).
   - `enrolled_university`: **386** قيمة مفقودة.
   - `education_level`: **460** قيمة مفقودة.
   - `major_discipline`: **2,813** قيمة مفقودة.
   - `experience`: **65** قيمة مفقودة.
   - `company_size`: **5,938** قيمة مفقودة (~31% مفقود).
   - `company_type`: **6,140** قيمة مفقودة (~32% مفقود).
   - `last_new_job`: **423** قيمة مفقودة.
   - `training_hours`: 0 (ساعات التدريب مكتملة).
   - `target`: 0 (كل العينات لها تصنيف معروف).
2. **توزيع الهدف (Target Distribution)**:
   - الفئة `0.0` (مش بيدور على وظيفة): **14,381** مرشح بنسبة **75.07%**.
   - الفئة `1.0` (بيدور على وظيفة جديدة): **4,777** مرشح بنسبة **24.93%**.
   - **الاستنتاج الحاسم**: لو عملنا موديل غبي بيقول دايماً "0"، دقته هتكون 75.07% لكنه موديل فاشل عملياً لأنه مش هيصطاد ولا مرشح بيبحث عن شغل! عشان كده دقة الـ Accuracy لوحدها مضللة هنا.

#### هـ) الربط الخارجي:
- بيتعامل مباشرة مع الداتاسيت المحملة في الذاكرة لتوجيه خطة المعالجة في `src/preprocessing.py`.

---

### الخلية 7 و 8: المعالجة وهندسة الميزات
#### الكود:
```python
# Separate raw features and target
X_raw = df.drop(columns=["target"]).copy()
y = df["target"].astype(int).copy()

# Apply domain-specific feature engineering from src/preprocessing.py
X_features = add_engineered_features(X_raw)

# Drop non-predictive and sensitive columns (enrollee_id, city, gender)
X = X_features.drop(columns=MODEL_EXCLUDED_COLUMNS, errors="ignore").copy()

print("Engineered features created successfully.")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("\nFeature Columns (18 total):")
print(X.columns.tolist())
X.head()
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `X_raw = df.drop(columns=["target"]).copy()`: عزل ميزات الإدخال في متغير `X_raw` بحذف عمود الهدف `target`. استخدام `.copy()` يمنع ظهور تحذيرات `SettingWithCopyWarning` في البانداز.
2. `y = df["target"].astype(int).copy()`: استخراج عمود الهدف وتحويله من `float` (0.0 و 1.0) إلى عدد صحيح `int` (0 و 1).
3. `X_features = add_engineered_features(X_raw)`: تطبيق دالة هندسة الميزات المكتوبة في `src/preprocessing.py` لإنشاء ميزات مشتقة قوية.
4. `X = X_features.drop(columns=MODEL_EXCLUDED_COLUMNS, errors="ignore").copy()`: حذف الأعمدة غير التنبؤية والحساسة المحددة في `MODEL_EXCLUDED_COLUMNS` وهي:
   - `enrollee_id`: مجرد رقم تسلسلي عشوائي للمرشح لا يحمل أي نمط تنبؤي وحفظه يسبب Overfitting.
   - `city`: كود المدينة (زي city_103) فيه أكثر من 120 مدينة مختلفة، مؤشر التنمية `city_development_index` يعبر عن قوتها الاقتصادية بأفضل شكل دون الحاجة لمئات الـ Dummy Columns.
   - `gender`: استبعاد الجنس للامتثال لمبادئ **الذكاء الاصطناعي الأخلاقي (Fairness & Ethical AI)** لمنع التحيز الجنسي في قرارات التوظيف.
5. طباعة تأكيد النجاح وأشكال المصفوفات وأسماء الـ 18 ميزة النهائية وعرض `X.head()`.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- **Feature Engineering**: الخوارزميات لا تفهم الدلالات الواقعية بمفردها؛ نحن من نخبرها بعلاقات المجال (HR Domain Knowledge).
- الميزات التي تم إنشاؤها في `src/preprocessing.py` هي:
  1. `experience_years`: تحويل نصوص سنوات الخبرة لأرقام (`<1` أصبحت `0.0`، و `>20` أصبحت `21.0`).
  2. `training_hours_log`: تحويل لوغاريتمي `np.log1p(training_hours)` لتقليل التواء التوزيع اليميني الشديد (Right Skewness) وجعله قريباً من التوزيع الطبيعي.
  3. `training_intensity`: حاصل قسمة `training_hours / (experience_years + 1.0)`. بتقيس كثافة التدريب مقارنة بالخبرة؛ الخريج الجديد اللي بياخد 100 ساعة تدريب شغوف جداً ومختلف عن ذو الـ 20 سنة خبرة اللي بياخد نفس الساعات.
  4. `is_company_missing` و `is_company_type_missing`: ميزتان ثنائيتان (0 أو 1) تشيران لغياب بيانات الشركة؛ الشخص الذي لا يكتب شركته الحالية غالباً عاطل أو يبحث بشدة عن وظيفة!
  5. `cdi_x_experience`: تفاعل مؤشر المدينة مع سنوات الخبرة `city_development_index * (experience_years + 1)`. المرشح ذو الخبرة العالية في مدينة نامية ضعيفة يختلف سلوكه في الانتقال عن نفس المرشح في عاصمة كبرى.
  6. `experience_job_ratio`: نسبة استقرار العمل `(experience_years + 1) / (last_job_num + 1)`. بتقيس معدل التبديل الوظيفي بالنسبة لعمره المهني.
  7. `is_low_cdi`: ميزة ثنائية توضح إذا كان مؤشر تنمية المدينة منخفضاً (`< 0.65`)؛ البيانات أثبتت أن المدن الأقل تنمية يهاجر منها المرشحون للبحث عن فرص عمل بشكل مضاعف.

#### ج) الاستفادة العملية:
- زيادة القوة التنبؤية للنماذج بشكل ملحوظ (الميزات المهندسة احتلت أعلى مراتب الأهمية في الـ Random Forest لاحقاً).
- حماية النظام من التحيز (Bias) والامتثال لمعايير الـ AI Governance بحذف `gender`.

#### د) تفسير الـ Output:
- **شكل الميزات (Features shape)**: `(19158, 18)` -> نفس عدد المرشحين (19,158) بعدد 18 ميزة نظيفة ومنتقاة.
- **شكل الهدف (Target shape)**: `(19158,)` -> مصفوفة أحادية الأبعاد للفئات (0 و 1).
- **قائمة الـ 18 عموداً**: تضم 10 ميزات رقمية + 4 ميزات فئوية اسمية + 3 ميزات فئوية ترتيبية + مؤشرات الفقدان والتفاعل.

#### هـ) الربط الخارجي:
- **الملف المصدري المعتمد**: استدعاء مباشر لدالة `add_engineered_features` والمتغير `MODEL_EXCLUDED_COLUMNS` من ملف `src/preprocessing.py`.

---

### الخلية 9 و 10: التحليل الاستكشافي للميزات المهندسة (EDA)
#### الكود:
```python
# Exploratory Analysis for Engineered Features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Comparison of raw training hours vs log-transformed training hours
sns.kdeplot(data=X_features, x="training_hours", label="Raw Hours", ax=axes[0], color="blue")
sns.kdeplot(data=X_features, x="training_hours_log", label="Log1p Hours", ax=axes[0], color="orange")
axes[0].set_title("Training Hours Distribution (Raw vs Log)")
axes[0].legend()

# 2. Boxplot of training intensity grouped by target
temp_eda_df = X_features.copy()
temp_eda_df["target"] = y
sns.boxplot(data=temp_eda_df, x="target", y="training_intensity", ax=axes[1], showfliers=False)
axes[1].set_title("Training Intensity vs Target")
axes[1].set_xlabel("Target (0: Not Looking, 1: Job Change)")
axes[1].set_ylabel("Training Intensity")

# 3. Bar chart showing job-change rate when company info is missing vs present
company_missing_target = pd.crosstab(
    X_features["is_company_missing"], y, normalize="index"
) * 100
company_missing_target.plot(kind="bar", stacked=True, ax=axes[2], color=["#4c72b0", "#c44e52"])
axes[2].set_title("Job-Change Rate by Missing Company Info")
axes[2].set_xlabel("Company Missing (0: Present, 1: Missing)")
axes[2].set_ylabel("Percentage (%)")
axes[2].legend(["Stay (0)", "Job Change (1)"])
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "engineered_features_distribution.png", dpi=200, bbox_inches="tight")
plt.show()
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `fig, axes = plt.subplots(1, 3, figsize=(18, 5))`: إنشاء مساحة رسم تحتوي على 3 رسومات جنب بعض (صف واحد و 3 أعمدة) بعرض 18 بوصة وارتفاع 5 بوصة.
2. **الرسم الأول (أقصى اليسار)**:
   - `sns.kdeplot(..., x="training_hours", color="blue")`: رسم منحنى كثافة التوزيع الاحتمالي لساعات التدريب الخام باللون الأزرق.
   - `sns.kdeplot(..., x="training_hours_log", color="orange")`: رسم نفس التوزيع بعد التحويل اللوغاريتمي باللون البرتقالي.
   - وضع عنوان وإظهار دليل الألوان (Legend).
3. **الرسم الثاني (في المنتصف)**:
   - إنشاء داتافريم مؤقتة `temp_eda_df` وإضافة عمود `target` إليها لربط الميزات بالفئات.
   - `sns.boxplot(..., x="target", y="training_intensity", showfliers=False)`: رسم صندوق التوزيع (Boxplot) لميزة كثافة التدريب لكل فئة من فئات الهدف مع إخفاء القيم الشاذة المتطرفة (`showfliers=False`) لجعل الصندوق واضحاً.
   - تسمية المحاور والعنوان.
4. **الرسم الثالث (أقصى اليمين)**:
   - `pd.crosstab(..., normalize="index") * 100`: جدول تقاطع إحصائي لحساب النسبة المئوية للمرشحين الذين يبحثون عن وظيفة عندما تكون بيانات الشركة مفقودة (1) مقابل موجودة (0).
   - `company_missing_target.plot(kind="bar", stacked=True, ...)`: رسم بياني شريطي تراكمي (Stacked Bar) ملون بالأزرق (الباقين) والأحمر (المغادرين).
5. `plt.tight_layout()`: ضبط المسافات بين الرسومات الثلاثة تلقائياً لمنع تداخل النصوص.
6. `plt.savefig(...)`: حفظ الصورة الناتجة بدقة عالية (200 DPI) داخل مجلد `outputs/figures/`.
7. `plt.show()`: عرض الرسم البياني داخل النوت بوك.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- تقديم إثبات إحصائي بصري للبشمهندس يبرهن على أن هندسة الميزات لم تكن مجرد كود عشوائي، بل غيرت في شكل البيانات وكشفت عن أنماط قوية مرتبطة بالهدف مباشرة.

#### ج) الاستفادة العملية:
- التأكد من فاعلية التحويل اللوغاريتمي (تقليل التشتت ومساعدة النماذج الخطية)، وإثبات أن غياب بيانات الشركة مؤشر قوي جداً على رغبة المرشح في الانتقال الوظيفي.

#### د) تفسير الـ Output ورسمته:
1. **الرسم الأول (Raw vs Log Training Hours)**:
   - المنحنى الأزرق (Raw Hours) ممتد لليمين مع قمة حادة جداً عند الصفر وتلاشي طويل حتى 300+ ساعة (Positive Skewness).
   - المنحنى البرتقالي (Log1p Hours) أصبح منحنى ناعماً ومحكوماً بين 0 و 6، وده بيساعد الانحدار اللوجستي جداً في تقارب الأوزان والـ Gradient Descent.
2. **الرسم الثاني (Training Intensity vs Target)**:
   - المرشحون الذين يريدون تغيير وظائفهم (`target = 1`) عندهم وسيط كثافة تدريب أعلى ومدى أوسع، مما يثبت أن الشخص المتفرغ أو المتحمس لتغيير مجاله يستثمر ساعات تدريب أكبر مقارنة بخبرته الحالية.
3. **الرسم الثالث (Job-Change Rate by Missing Company Info)**:
   - العمود (0) [بيانات الشركة موجودة]: نسبة الرغبة في التغيير تمثل تقريباً **15-17%** فقط من هذه الفئة.
   - العمود (1) [بيانات الشركة مفقودة]: نسبة الرغبة في التغيير تقفز لأكثر من **40%**! ده إثبات صارخ على أن الـ Missingness في هذا العمود ليس عشوائياً (Missing Not at Random - MNAR) ويحمل إشارة تنبؤية فائقة الأهمية.

#### هـ) الربط الخارجي:
- **الملف المحفوظ**: تم حفظ الرسم التوضيحي بالكامل كصورة عالية الجودة في المسار:
  `outputs/figures/engineered_features_distribution.png`.

---

### الخلية 11 و 12: التحقق من أبعاد الميزات والهدف
#### الكود:
```python
# Verify defined features and target variable
print("Features shape (X):", X.shape)
print("Target shape (y):", y.shape)
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `print("Features shape (X):", X.shape)`: طباعة أبعاد مصفوفة المدخلات `X`.
2. `print("Target shape (y):", y.shape)`: طباعة طول متجه المخرجات `y`.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- إجراء فحص تأكيدي (Sanity Check) قبل الشروع في تقسيم البيانات وتدريب الموديل للتأكد من عدم حدوث أي خطأ في عدد الأسطر أو اختفاء بيانات أثناء عمليات الـ Drop السابقة.

#### ج) الاستفادة العملية:
- حماية الكود من خطأ عدم توافق الأبعاد (`ValueError: Found input variables with inconsistent numbers of samples`).

#### د) تفسير الـ Output:
- `Features shape (X): (19158, 18)`
- `Target shape (y): (19158,)`
- عدد الصفوف في `X` هو 19,158 وهو مطابق تماماً لعدد العناصر في `y`، وعدد الأعمدة 18 عموداً جاهزاً للتقسيم.

#### هـ) الربط الخارجي:
- مصفوفات في الذاكرة مهيأة للخلية التالية.

---

### الخلية 13 و 14: تقسيم البيانات تدريب واختبار (Train-Test Split)
#### الكود:
```python
# Split the data into 80% training and 20% testing sets
# stratify=y preserves the 75:25 class ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True).round(4))
print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).round(4))
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. استدعاء دالة `train_test_split`:
   - `X, y`: الميزات والهدف المراد تقسيمهم.
   - `test_size=0.20`: تخصيص 20% من البيانات للاختبار النهائي و 80% لتدريب النماذج (وهي النسبة المعيارية في أبحاث تعلم الآلة).
   - `random_state=RANDOM_STATE`: تثبيت بذرة العشوائية (42) لتكرار نفس التقسيم دائماً.
   - `stratify=y`: **أهم معلمة في السطر**؛ تضمن الحفاظ على نفس نسبة الفئات (75% للصنف 0 و 25% للصنف 1) في كل من بيانات التدريب وبيانات الاختبار بالتساوي.
2. طباعة شكل مصفوفة التدريب وشكل مصفوفة الاختبار.
3. حساب وطباعة التوزيع النسبي للفئات في التدريب والاختبار للتأكد من عمل الـ Stratification بدقة 4 أرقام عشرية.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- منع ظاهرة الـ **Data Leakage**؛ التقسيم يتم أولاً قبل أي معالجة بالوسيط أو المتوسط أو التحجيم (Scaling).
- استخدام `stratify=y` ضرورة قصوى لأن البيانات غير متوازنة؛ وبدونها قد تسقط عينات فئة الأقلية في الاختبار صدفة وتختفي من التدريب.

#### ج) الاستفادة العملية:
- تجهيز مجموعتي تدريب واختبار متكافئتين إحصائياً لقياس كفاءة النماذج على بيانات حقيقية لم ترها من قبل.

#### د) تفسير الـ Output:
- **شكل مصفوفة التدريب (`X_train`)**: `(15326, 18)` -> تم أخذ 15,326 مرشح للتدريب.
- **شكل مصفوفة الاختبار (`X_test`)**: `(3832, 18)` -> تم عزل 3,832 مرشح للاختبار النهائي (مجموعهم 19,158 عينة).
- **توزيع الفئات في التدريب (`Train target distribution`)**:
  - فئة 0: `0.7506` (75.06%)
  - فئة 1: `0.2494` (24.94%)
- **توزيع الفئات في الاختبار (`Test target distribution`)**:
  - فئة 0: `0.7508` (75.08%)
  - فئة 1: `0.2492` (24.92%)
- **الملاحظة**: النسب متطابقة بنسبة 100% تقريباً حتى خانة الآلاف، مما يؤكد نجاح التقسيم الطبقي المتقن.

#### هـ) الربط الخارجي:
- متغيرات في الذاكرة لتغذية الـ Pipelines في الخلية القادمة.

---

### الخلية 15 و 16: تدريب نموذج الانحدار اللوجستي (Logistic Regression Pipeline)
#### الكود:
```python
# Build complete ML pipelines (handling imputation, encoding, and scaling inside)
lr_pipeline, rf_pipeline = build_models()

# Train Logistic Regression Pipeline on training data
lr_pipeline.fit(X_train, y_train)

# Make predictions and calculate probabilities on test data
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("Logistic Regression Training Complete.")
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `lr_pipeline, rf_pipeline = build_models()`: استدعاء دالة بناء النماذج من `src/modeling.py`، والتي تُنشئ سلسلتي معالجة كاملتين (Pipelines):
   - الأولى للانحدار اللوجستي (`lr_pipeline`).
   - الثانية للغابة العشوائية (`rf_pipeline`).
2. `lr_pipeline.fit(X_train, y_train)`: تدريب الـ Pipeline بالكامل على بيانات التدريب فقط:
   - الـ Pipeline يقوم بحساب وسيط الميزات الرقمية وتعويض النواقص (Median Imputation).
   - يقوم بتحجيم الميزات الرقمية بالـ `StandardScaler` (طرح المتوسط والقسمة على الانحراف المعياري).
   - تعويض النواقص الفئوية بالقيمة الأكثر تكراراً (Mode).
   - ترميز الميزات الترتيبية (`OrdinalEncoder`) مثل المؤهل العلمي ومستوى الشركة.
   - ترميز الميزات الاسمية (`OneHotEncoder`) مثل التخصص والخبرة.
   - تدريب نموذج الـ Logistic Regression على الميزات المعالجة بوزن فئات متوازن `class_weight='balanced'`.
3. `y_pred_lr = lr_pipeline.predict(X_test)`: التنبؤ بالفئات المباشرة (0 أو 1) لبيانات الاختبار.
4. `y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]`: استخراج الاحتمالية الرياضية الصريحة لانتماء المرشح للفئة 1 (الرغبة في تغيير الوظيفة) والتي تتراوح بين 0.0 و 1.0.
5. طباعة رسالة تؤكد انتهاء التدريب.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- **منع تسريب البيانات (Zero Data Leakage)**: لأن كل خطوة من خطوات التعويض والتحجيم تتم داخلياً عبر `fit` على الـ Train فقط، ثم تُطبق بـ `transform` على الـ Test دون أن ترى بيانات الاختبار مسبقاً.
- معالجة عدم التوازن بـ `class_weight='balanced'`؛ بتعاقب الموديل بوزن أكبر لو أخطأ في فئة الأقلية (الفئة 1)، فتجبره إنه يهتم بيها ويصطادها.

#### ج) الاستفادة العملية:
- تدريب نموذج معياري خطي (Baseline Model) سريع، وتجهيز مصفوفات التوقعات والاحتماليات لتقييمه في الخلايا اللاحقة.

#### د) تفسير الـ Output:
- طباعة رسالة: `"Logistic Regression Training Complete."` (تم التدريب بنجاح بدون أخطاء convergence بفضل رفع `max_iter=1000`).

#### هـ) الربط الخارجي:
- **الملف المصدري المعتمد**: استدعاء دالة `build_models` من `src/modeling.py`، والتي تحتوي على تعريف `build_preprocessor` وأعمدة `NUMERIC_FEATURES`, `ORDINAL_FEATURES`, `NOMINAL_FEATURES`.

---

### الخلية 17 و 18: مصفوفة الارتباك للانحدار اللوجستي (Confusion Matrix Heatmap)
#### الكود:
```python
# Plot Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.show()
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `cm_lr = confusion_matrix(y_test, y_pred_lr)`: حساب مصفوفة الارتباك 2×2 بمقارنة القيم الحقيقية `y_test` مع التوقعات `y_pred_lr`.
2. `plt.figure(figsize=(6, 4))`: تجهيز مساحة رسم بمقاس 6×4 بوصة.
3. `sns.heatmap(...)`:
   - `cm_lr`: مصفوفة البيانات المراد رسمها.
   - `annot=True`: إظهار الأرقام العددية داخل المربعات.
   - `fmt="d"`: تنسيق الأرقام كأعداد صحيحة عادية بدون علامات عشرية.
   - `cmap="Blues"`: التدرج اللوني باللون الأزرق (الدرجة الداكنة للأعداد الأكبر).
4. تسمية المحور الأفقي بالتوقعات `Predicted` والمحور الرأسي بالقيم الحقيقية `Actual`.
5. وضع عنوان وتنسيق الهوامش وعرض الرسم.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- الدقة الإجمالية مش كافية لفهم أخطاء الموديل؛ مصفوفة الارتباك بتوضح بالظبط أين أخطأ النموذج: هل أخطأ في توقع المرشحين المغادرين أم الباقين؟

#### ج) الاستفادة العملية:
- تحديد عدد الـ True Positives والـ False Positives والـ False Negatives لحساب مقاييس الـ Precision والـ Recall بالورقة والقلم للتأكد من فهم البشمهندس لها.

#### د) تفسير الـ Output بالتفصيل (الأرقام ودلالتها):
المصفوفة تحتوي على 4 خلايا بمجموع 3,832 عينة اختبار:
1. **أعلى اليسار (True Negative - TN) = 2,074**:
   - مرشحون حقيقيون مش عايزين يغيروا وظيفتهم (0) والموديل توقع بنجاح إنهم مش هيغيروا (0).
2. **أعلى اليمين (False Positive - FP) = 803**:
   - مرشحون مش عايزين يغيروا (0)، لكن الموديل اعتقد خطأً إنهم هيمشوا (1).
3. **أسفل اليسار (False Negative - FN) = 192**:
   - مرشحون في الحقيقة عايزين يمشوا (1)، لكن الموديل غفل عنهم وتوقع إنهم مستقرين (0).
4. **أسفل اليمين (True Positive - TP) = 763**:
   - مرشحون في الحقيقة بيبحثوا عن وظيفة (1)، والموديل نجح ببراعة في اصطيادهم وتوقع إنهم هيمشوا (1).
- **الاستنتاج الحسابي العملي**:
  - من أصل 955 مرشح بيبحثوا عن عمل في الاختبار (`192 + 763`)، الموديل اصطاد **763** بنسبة استدعاء (Recall) قوية جداً = `763 / 955 = 79.90%`!
  - الدقة (Accuracy) = `(2074 + 763) / 3832 = 74.03%`.

#### هـ) الربط الخارجي:
- رسم تفاعلي داخل النوت بوك مشتق من نواتج تنبؤات النموذج في الذاكرة.

---

### الخلية 19 و 20: تدريب نموذج الغابة العشوائية (Random Forest Classifier)
#### الكود:
```python
# Train Random Forest Pipeline on training data
rf_pipeline.fit(X_train, y_train)

# Make predictions and calculate probabilities on test data
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("Random Forest Training Complete.")
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `rf_pipeline.fit(X_train, y_train)`: تدريب سلسلة الغابة العشوائية على بيانات التدريب.
   - الموديل يحتوي على 200 شجرة قرار (`n_estimators=200`).
   - تم تقييد أقصى عمق لكل شجرة بـ 10 مستويات (`max_depth=10`) لمنع الـ Overfitting.
   - تم تفعيل وزن الفئات المتوازن `class_weight='balanced'`.
   - تم تفعيل المعالجة المتوازية على كل أنوية المعالج `n_jobs=-1` لسرعة التدريب.
   - الـ Pipeline المخصص للغابة العشوائية لا يحتاج لتحجيم رقمي (`scale_numeric=False`) لأن أشجار القرار لا تتأثر بمدى الميزات (Scale-Invariant).
2. `y_pred_rf = rf_pipeline.predict(X_test)`: التنبؤ بالفئات لبيانات الاختبار.
3. `y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]`: استخراج الاحتماليات الصريحة للفئة 1 من الغابة العشوائية.
4. طباعة رسالة تفيد باكتمال التدريب.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- تقديم نموذج غير خطي وتجميعي قوي (Ensemble Learning) قادر على استيعاب العلاقات المعقدة والتفاعلات بين الميزات (Non-linear interactions) التي قد يعجز عنها النموذج الخطي.

#### ج) الاستفادة العملية:
- رفع دقة التنبؤ ومقاييس الأداء التنافسية مقارنة بالانحدار اللوجستي.

#### د) تفسير الـ Output:
- رسالة: `"Random Forest Training Complete."` (انتهاء تدريب الـ 200 شجرة بنجاح تام).

#### هـ) الربط الخارجي:
- استدعاء بنية الموديل من `src/modeling.py` (كلاس `RandomForestClassifier`).

---

### الخلية 21 و 22: مصفوفة الارتباك للغابة العشوائية (Confusion Matrix Heatmap)
#### الكود:
```python
# Plot Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.show()
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `cm_rf = confusion_matrix(y_test, y_pred_rf)`: حساب مصفوفة الارتباك لنتائج الغابة العشوائية.
2. تجهيز مساحة رسم بمقاس 6×4 بوصة.
3. رسم الـ Heatmap باستخدام اللون الأخضر (`cmap="Greens"`) لتمييزه عن الانحدار اللوجستي.
4. كتابة أسماء المحاور والعنوان وعرض الرسم.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- إجراء مقارنة بصرية سريعة بين أخطاء الغابة العشوائية والانحدار اللوجستي.

#### ج) الاستفادة العملية:
- كشف التحسن الملموس في تقليل الإنذارات الكاذبة (False Positives).

#### د) تفسير الـ Output بالتفصيل (الأرقام ودلالتها):
المصفوفة تحتوي على القيم التالية:
1. **(True Negative - TN) = 2,199**:
   - زيادة ملحوظة عن الانحدار اللوجستي (كان 2,074)؛ الغابة العشوائية تعرفت على 125 شخصاً إضافياً من المستقرين في عملهم بنجاح.
2. **(False Positive - FP) = 678**:
   - انخفاض كبير وممتاز للإنذارات الكاذبة مقارنة بـ 803 في الانحدار اللوجستي (وفرت وقت مسؤول الـ HR في التواصل مع أشخاص لا ينوون ترك عملهم).
3. **(False Negative - FN) = 201**:
   - عدد الذين فاتوا على الموديل 201 مرشح (قريب جداً من الـ 192 في الموديل السابق).
4. **(True Positive - TP) = 754**:
   - اصطياد 754 مرشحاً راغباً في الانتقال من أصل 955.
- **الخلاصة الإحصائية**: الغابة العشوائية رفعت الدقة الكلية إلى **77.06%** ورفعت دقة التحديد (Precision) إلى **52.65%** مع الحفاظ على استدعاء فائق يقارب **79%**.

#### هـ) الربط الخارجي:
- رسم تفاعلي داخل النوت بوك.

---

### الخلية 23 و 24: تقييم النماذج بمقاييس التصنيف (Evaluate Classification Reports)
#### الكود:
```python
# Evaluate Logistic Regression
print("=== Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred_lr, digits=4))
auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"Logistic Regression ROC-AUC Score: {auc_lr:.4f}")

# Evaluate Random Forest
print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred_rf, digits=4))
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"Random Forest ROC-AUC Score: {auc_rf:.4f}")
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `classification_report(y_test, y_pred_lr, digits=4)`: طباعة تقرير تصنيف مفصل للانحدار اللوجستي بدقة 4 أرقام عشرية يوضح:
   - Precision (الدقة التحديدية).
   - Recall (حساسية الاستدعاء).
   - F1-Score (المتوسط التوافقي بين الدقة والاستدعاء).
   - Support (عدد العينات الفعلي في كل فئة).
2. `auc_lr = roc_auc_score(y_test, y_prob_lr)`: حساب المساحة تحت منحنى الـ ROC باستخدام الاحتماليات الصريحة `y_prob_lr` وطباعتها.
3. تكرار نفس التقرير وحساب الـ ROC-AUC لنموذج الغابة العشوائية `auc_rf` وطباعتها.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- توفير تقييم علمي صارم متعدد الأبعاد؛ ففي قضايا الـ Imbalanced Data، الاعتماد على رقم واحد مضلل، ولابد من تقييم الفئة 1 بمفردها وتقييم قدرة التمييز الاحتمالية عبر الـ ROC-AUC.

#### ج) الاستفادة العملية:
- حسم المقارنة بين النموذجين برؤية متكاملة قبل اعتماد النموذج الفائز.

#### د) تفسير الـ Output وأرقامه بالتفصيل:
1. **تقرير الانحدار اللوجستي (Logistic Regression)**:
   - الفئة 0 (المستقرين): Precision = `0.9153`، Recall = `0.7209`، F1-Score = `0.8065` (Support = 2877).
   - الفئة 1 (المغادرين): Precision = `0.4872`، Recall = `0.7990`، F1-Score = `0.6053` (Support = 955).
   - الدقة الكلية (Accuracy) = `0.7403` (74.03%).
   - مقياس الـ **ROC-AUC Score = 0.8008**.
2. **تقرير الغابة العشوائية (Random Forest)**:
   - الفئة 0 (المستقرين): Precision = `0.9163`، Recall = `0.7643`، F1-Score = `0.8334` (Support = 2877).
   - الفئة 1 (المغادرين): Precision = `0.5265`، Recall = `0.7895`، F1-Score = `0.6318` (Support = 955).
   - الدقة الكلية (Accuracy) = `0.7706` (77.06%).
   - مقياس الـ **ROC-AUC Score = 0.8154**.
- **المقارنة والتقييم الفني**:
  - الغابة العشوائية تفوقت في **F1-Score** للفئة المستهدفة (0.6318 مقابل 0.6053).
  - الغابة العشوائية تفوقت في **ROC-AUC** (0.8154 مقابل 0.8008) مما يعني أنها أفضل قدرة على فرز وترتيب المرشحين باحتماليات متدرجة.
  - الغابة العشوائية رفعت الـ Precision من 48.7% إلى 52.6% دون التضحية بالاستدعاء العالي (~79%).

#### هـ) الربط الخارجي:
- استدعاء مباشر لدوال `classification_report` و `roc_auc_score` من مكتبة `sklearn.metrics`.

---

### الخلية 25 و 26: مقارنة منحنيات الـ ROC (Compare ROC Curves)
#### الكود:
```python
# Calculate ROC curve values
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

# Plot ROC curves comparison
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline (AUC = 0.500)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curve_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)`: حساب نقاط معدل الإيجابيات الكاذبة (FPR) ومعدل الإيجابيات الحقيقية (TPR) لموديل الانحدار اللوجستي عند كل عتبات القرار الممكنة (Thresholds).
2. `fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)`: حساب نفس النقاط لموديل الغابة العشوائية.
3. `plt.figure(figsize=(8, 6))`: مساحة رسم بأبعاد 8×6 بوصة.
4. `plt.plot(fpr_lr, tpr_lr, ...)`: رسم منحنى الانحدار اللوجستي وتوضيح قيمة المساحة تحت المنحنى (AUC = 0.801).
5. `plt.plot(fpr_rf, tpr_rf, ...)`: رسم منحنى الغابة العشوائية وتوضيح قيمة (AUC = 0.815).
6. `plt.plot([0, 1], [0, 1], linestyle="--", color="gray", ...)`: رسم خط التخمين العشوائي القطري المتقطع (AUC = 0.500) كمرجع للمقارنة.
7. كتابة أسماء المحاور والعنوان وتثبيت الـ Legend أسفل اليمين وحفظ الرسم وعرضه.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- تقديم برهان بصري رسمي ومعياري يُظهر كفاءة الفصل بين الفئات بصرف النظر عن عتبة القرار (Classification Threshold = 0.5).

#### ج) الاستفادة العملية:
- إثبات أن النماذج أعلى بكثير جداً من التخمين العشوائي (0.815 مقابل 0.500)، وأن الغابة العشوائية تتفوق في معظم مسار المنحنى.

#### د) تفسير الـ Output والرسمة:
- الخط الرمادي المتقطع يمثل أداء رمي العملة العشوائي (Coin Flip) ومساحته 0.500.
- منحنى الانحدار اللوجستي يرتفع بقوة لأعلى جهة اليسار بمساحة **0.801**.
- منحنى الغابة العشوائية يعلو منحنى الانحدار اللوجستي في أغلب أجزاء المنحنى محققاً مساحة **0.815**.
- **المعنى العملي**: لو اخترنا مرشحاً عشوائياً يبحث عن عمل ومرشحاً آخر لا يبحث، فإن فرصة أن يعطي نموذج الغابة العشوائية احتمالية أعلى للمرشح الأول هي **81.5%**، وهي قدرة تمييزية ممتازة جداً في مشاكل الموارد البشرية المعقدة.

#### هـ) الربط الخارجي:
- **الملف المحفوظ**: تم حفظ الرسم بصيغة PNG داخل:
  `outputs/figures/roc_curve_comparison.png`.

---

### الخلية 27 و 28: أهمية الميزات في الغابة العشوائية (Feature Importance)
#### الكود:
```python
# Extract feature importances from Random Forest model inside the pipeline
rf_model = rf_pipeline.named_steps["model"]
rf_preprocessor = rf_pipeline.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("Top 10 Predictive Features:")
print(feature_importance.head(10))

# Save feature importance to results
feature_importance.to_csv(RESULTS_DIR / "top_predictive_features.csv", index=False)

# Plot horizontal bar chart
top_features = feature_importance.head(10).sort_values("Importance", ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"], color="teal")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "top_predictive_features.png", dpi=200, bbox_inches="tight")
plt.show()
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `rf_model = rf_pipeline.named_steps["model"]`: استخراج كائن موديل الغابة العشوائية المدرب من داخل خط الـ Pipeline.
2. `rf_preprocessor = rf_pipeline.named_steps["preprocessor"]`: استخراج كائن محول الميزات من داخل الـ Pipeline.
3. `feature_names = rf_preprocessor.get_feature_names_out()`: استخراج الأسماء الدقيقة لكل الميزات بعد المعالجة وفك الـ One-Hot Encoding.
4. بناء جدول `feature_importance`: تجميع كل ميزة مع درجة أهميتها المحسوبة عبر خوارزمية الـ Gini Impurity (`rf_model.feature_importances_`) وترتيبها تنازلياً.
5. طباعة أول 10 ميزات في الطرفية.
6. `feature_importance.to_csv(...)`: حفظ الجدول كاملاً في ملف CSV داخل مجلد النتائج.
7. رسم مخطط شريطي أفقي (`plt.barh`) لأفضل 10 ميزات بلون تركواز أنيق (`teal`).
8. حفظ الرسم بصيغة PNG وعرضه على الشاشة.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- **قابلية التفسير (Explainable AI - XAI)**: حتى لا يكون الموديل عبارة عن صندوق أسود (Black Box)؛ نريد أن نفهم "لماذا" يتوقع النموذج أن هذا المرشح سيترك عمله؟

#### ج) الاستفادة العملية:
- إعطاء رؤية استراتيجية لمسؤولي التوظيف والـ HR عن العوامل الحقيقية التي تدفع الكفاءات للبحث عن عمل جديد.

#### د) تفسير الـ Output وأرقام الجدول:
أهم 10 ميزات خرجت بالترتيب ونسبها المئوية من التأثير الإجمالي:
1. `city_development_index` (**21.94%**): الميزة رقم 1 بلا منازع؛ مؤشر تطور المدينة هو العامل الأكبر الذي يحدد رغبة المرشح في الهجرة الوظيفية.
2. `is_low_cdi` (**14.54%**): ميزتنا المهندسة للتحقق من انخفاض مؤشر المدينة عن 0.65 جاءت في المركز الثاني مباشرة!
3. `is_company_missing` (**11.49%**): ميزتنا المهندسة عن غياب بيانات الشركة جاءت في المركز الثالث؛ إثبات قاطع على فاعلية تفكيرنا الهندسي.
4. `cdi_x_experience` (**8.02%**): ميزتنا المهندسة لتفاعل المدينة مع الخبرة حلت رابعاً.
5. `is_company_type_missing` (**7.16%**): مؤشر غياب نوع الشركة جاء خامساً.
6. `education_level` (**4.09%**): المؤهل الأكاديمي.
7. `training_intensity` (**4.02%**): ميزتنا المهندسة لكثافة ساعات التدريب لكل سنة خبرة.
8. `company_size` (**3.81%**): حجم الشركة السابقة.
9. `training_hours_log` (**3.59%**): الميزة المهندسة لساعات التدريب اللوغاريتمية.
10. `experience_job_ratio` (**3.58%**): الميزة المهندسة لنسبة استقرار الوظائف.
- **ملاحظة مبهرة للمناقشة**: من بين أفضل 10 ميزات يعتمد عليها النموذج، هناك **7 ميزات مهندسة من ابتكارنا** من ملف `src/preprocessing.py`! وده دليل قاطع على القيمة المضافة لعملنا في هندسة الميزات.

#### هـ) الربط الخارجي:
- **الملف المكتوب**: `outputs/results/top_predictive_features.csv`.
- **الرسم المحفوظ**: `outputs/figures/top_predictive_features.png`.

---

### الخلية 29 و 30: ملخص ومقارنة النماذج مع خط الأساس (Models Benchmark Summary)
#### الكود:
```python
# Summary Table comparing both models
metrics_summary = pd.DataFrame({
    "Logistic Regression": calculate_metrics(y_test, y_pred_lr, y_prob_lr),
    "Random Forest": calculate_metrics(y_test, y_pred_rf, y_prob_rf),
}).T

metrics_summary.index.name = "Model"
metrics_summary = metrics_summary.round(4)

print("=== Models Benchmark Metrics Summary ===\n")
print(metrics_summary)

# Majority-class baseline accuracy
majority_class = int(y_train.mode()[0])
baseline_pred = np.full(len(y_test), majority_class)
baseline_acc = accuracy_score(y_test, baseline_pred)
print(f"\nMajority-class baseline accuracy: {baseline_acc:.4f}")
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `calculate_metrics(...)`: استدعاء دالة حساب المقاييس من `src/modeling.py` للنموذجين وتجميع النتائج في قاموسين.
2. `pd.DataFrame({...}).T`: تجميع النتائج في داتافريم وعمل Transpose (`.T`) لتكون أسماء النماذج في الصفوف والمقاييس في الأعمدة.
3. `metrics_summary.index.name = "Model"`: تسمية عمود الفهرس باسم Model.
4. `.round(4)`: تقريب كل النتائج إلى 4 خانات عشرية.
5. `majority_class = int(y_train.mode()[0])`: تحديد الفئة الغالبة في بيانات التدريب (وهي الفئة 0).
6. `baseline_pred = np.full(len(y_test), majority_class)`: إنشاء مصفوفة تنبؤ ساذجة تفترض أن كل المرشحين لن يغيروا وظائفهم (تخمين الفئة الغالبة دائماً).
7. `baseline_acc = accuracy_score(...)`: حساب دقة خط الأساس الساذج ومقارنتها بالنماذج.
8. طباعة الجدول النهائي ودقة خط الأساس في الشاشة.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- تقديم جدول مقارنة موحد وشامل (Executive Summary Table) ومقارنته مع خط الأساس الساذج (Majority Baseline)، وهو المطلب الأكاديمي والمهني الأساسي لأي مشروع تعلم آلة حقيقي.

#### ج) الاستفادة العملية:
- اتخاذ قرار نهائي وموثق باختيار الغابة العشوائية كنموذج إنتاجي للمشروع.

#### د) تفسير الـ Output والجدول:
```
=== Models Benchmark Metrics Summary ===

                     Accuracy  Precision  Recall  F1 Score  ROC-AUC
Model                                                              
Logistic Regression    0.7403     0.4872  0.7990    0.6053   0.8008
Random Forest          0.7706     0.5265  0.7895    0.6318   0.8154

Majority-class baseline accuracy: 0.7508
```
- **تحليل الأرقام مقارنة بخط الأساس (0.7508)**:
  - خط الأساس يقول: لو خمنا دائماً 0 سنحصل على دقة 75.08% لكن F1 سيكون 0.0 و Recall سيكون 0.0!
  - الانحدار اللوجستي حقق دقة 74.03% (أقل من خط الأساس الساذج كدقة كلية)، لكنه اصطاد 79.9% من المرشحين وحقق F1 = 0.6053 و AUC = 0.8008.
  - الغابة العشوائية تفوقت على خط الأساس في الدقة (77.06% > 75.08%)، وتفوقت في كل شيء: Precision أعلى (52.65%)، F1 أعلى (0.6318)، و ROC-AUC أعلى (0.8154).

#### هـ) الربط الخارجي:
- دالة `calculate_metrics` مستوردة من `src/modeling.py`.

---

### الخلية 31 و 32: ترتيب أفضل 10 مرشحين (Top 10 Candidate Ranking)
#### الكود:
```python
# Rank test candidates by predicted probability of job-change intention
ranking_columns = [
    "enrollee_id",
    "education_level",
    "experience",
    "company_type",
    "training_hours",
    "city_development_index",
]
available_cols = [c for c in ranking_columns if c in df.columns]

ranking_df = df.loc[X_test.index, available_cols].copy()
ranking_df["Job_Change_Probability"] = y_prob_rf
ranking_df["Job_Change_Percentage"] = (y_prob_rf * 100).round(2)

# Sort candidates descending by probability
ranking_df = ranking_df.sort_values("Job_Change_Probability", ascending=False).reset_index(drop=True)
ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

# Save top 10 recommended candidates
top_10 = ranking_df.head(10).copy()
top_10.to_csv(RESULTS_DIR / "top_10_recommended_candidates.csv", index=False)

print("Top 10 Candidates by Predicted Job-Change Intention:")
print(top_10)
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `ranking_columns`: تحديد الأعمدة الأساسية اللي مسؤول التوظيف (HR Recruiter) محتاج يشوفها عشان يفهم ملف المرشح (رقم المرشح، مؤهله، خبرته، نوع شركته، ساعات تدريبه، مؤشر مدينته).
2. `available_cols`: فلترة الأعمدة للتأكد من وجودها في الداتاسيت بأمان.
3. `ranking_df = df.loc[X_test.index, available_cols].copy()`: استخراج بيانات عينات الاختبار الأصلية المقابلة لمؤشرات `X_test`.
4. `ranking_df["Job_Change_Probability"] = y_prob_rf`: إضافة عمود احتمالية الانتقال المحسوبة من موديل الغابة العشوائية.
5. `ranking_df["Job_Change_Percentage"] = (y_prob_rf * 100).round(2)`: تحويل الاحتمالية لنسبة مئوية واضحة وسهلة القراءة لغير التقنيين.
6. `ranking_df.sort_values(..., ascending=False)`: ترتيب المرشحين تنازلياً من الأعلى احتمالاً إلى الأقل.
7. `ranking_df.insert(0, "Rank", ...)`: إدراج عمود الترتيب (الرتبة 1، 2، 3...) في أول الجدول.
8. `top_10.to_csv(...)`: استخراج أفضل 10 مرشحين وحفظهم في ملف CSV.
9. طباعة جدول أفضل 10 مرشحين في الطرفية.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- **تحويل الـ Machine Learning إلى منتج بيزنس ملموس (Actionable Business Value)**: مسؤولو الـ HR لا يتعاملون مع مصفوفات 0 و 1، بل يحتاجون إلى قائمة مرتبة بالأشخاص الأكثر جاهزية للمقابلة الفورية للبدء في الاتصال بهم هاتفياً.

#### ج) الاستفادة العملية:
- تزويد لوحة التحكم (Streamlit Dashboard) وواجهة الـ API بقائمة جاهزة للمرشحين الأكثر رغبة في الانتقال.

#### د) تفسير الـ Output وأرقام الجدول:
جدول أفضل 10 مرشحين أظهر نتائج مذهلة تؤكد صحة تعلم النموذج:
- **الرتبة 1**: المرشح رقم `19759`، خريج، سنة واحدة خبرة، شركته مجهولة (NaN)، مؤشر مدينته منخفض جداً (`0.624`)، باحتمالية انتقال **88.26%**.
- **المرشحون من 2 إلى 10**: تتراوح احتمالية انتقالهم بين **88.16%** و **86.18%**.
- **النمط المشترك الملحوظ**: جميعهم بلا استثناء يعيشون في مدينة مؤشرها **0.624**، وخبراتهم صغيرة إلى متوسطة (بين أقل من سنة و 3 سنوات)؛ وهذا يؤكد أن النموذج التقط النمط الواقعي لكفاءات المدن النامية الراغبة بقوة في الانتقال وتغيير حياتها المهنية.

#### هـ) الربط الخارجي:
- **الملف المكتوب**: `outputs/results/top_10_recommended_candidates.csv`.

---

### الخلية 33 و 34: حفظ النماذج ومخرجات المشروع (Save Model Artifacts)
#### الكود:
```python
# 1. Save benchmark metrics for Dashboard and FastAPI
metrics_summary.to_csv(RESULTS_DIR / "models_benchmark_metrics.csv")

# 2. Select and save the best pipeline (Random Forest)
best_model_name = metrics_summary.sort_values(["F1 Score", "ROC-AUC"], ascending=False).index[0]
best_pipeline = rf_pipeline if best_model_name == "Random Forest" else lr_pipeline

joblib.dump(lr_pipeline, MODELS_DIR / "recruitment_logistic_regression.pkl")
joblib.dump(rf_pipeline, MODELS_DIR / "recruitment_random_forest.pkl")
joblib.dump(best_pipeline, MODELS_DIR / "best_recruitment_pipeline.pkl")
joblib.dump(best_pipeline.named_steps["preprocessor"], MODELS_DIR / "preprocessor_pipeline.pkl")

# 3. Save cleaned candidates data for dashboard display
cleaned_candidates = df.copy()

# معالجة الأعمدة الرقمية
for col in ["city_development_index", "training_hours"]:
    median_val = X_train[col].median() if col in X_train.columns else df[col].median()
    cleaned_candidates[col] = cleaned_candidates[col].fillna(median_val)

# معالجة الأعمدة الفئوية (بما فيها gender من الداتاسيت الأصلية بأمان)
cat_columns = [
    "gender", "enrolled_university", "education_level", "major_discipline",
    "experience", "company_size", "company_type", "last_new_job"
]

for col in cat_columns:
    if col in X_train.columns:
        mode_series = X_train[col].mode(dropna=True)
    elif col in df.columns:
        mode_series = df[col].mode(dropna=True)
    else:
        mode_series = []
        
    mode_val = mode_series.iloc[0] if len(mode_series) else "Unknown"
    cleaned_candidates[col] = cleaned_candidates[col].fillna(mode_val)

cleaned_candidates.to_csv(PROCESSED_DIR / "cleaned_candidates.csv", index=False)

# 4. Save metadata JSON for the API /model-info endpoint
metadata = {
    "best_model": best_model_name,
    "selection_metric": "F1 Score",
    "optimal_threshold": 0.50,
    "target_definition": {
        "0": "Not looking for a job change",
        "1": "Looking for a job change"
    },
    "excluded_from_model": MODEL_EXCLUDED_COLUMNS,
    "random_state": RANDOM_STATE
}

with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("ALL PROJECT ARTIFACTS AND MODELS SAVED SUCCESSFULLY!")
print(f"Primary Model: {best_model_name}")
print(f"Models directory:    {MODELS_DIR}")
print(f"Results directory:   {RESULTS_DIR}")
print(f"Figures directory:   {FIGURES_DIR}")
print(f"Processed directory: {PROCESSED_DIR}")
```

#### أ) شرح الكود سطر بسطر حرفياً:
1. `metrics_summary.to_csv(...)`: حفظ جدول مقارنة المقاييس كاملاً في ملف CSV للداشبورد والـ API.
2. اختيار الموديل الفائز آلياً:
   - `best_model_name = metrics_summary.sort_values(["F1 Score", "ROC-AUC"], ascending=False).index[0]`: فحص الجدول واختيار صاحب أعلى F1 Score و ROC-AUC (والذي اختار Random Forest تلقائياً).
3. حفظ الموديلات عبر مكتبة `joblib`:
   - `recruitment_logistic_regression.pkl`: حفظ بايبلاين الانحدار اللوجستي.
   - `recruitment_random_forest.pkl`: حفظ بايبلاين الغابة العشوائية.
   - `best_recruitment_pipeline.pkl`: حفظ بايبلاين الموديل الفائز ليكون هو الموديل الإنتاجي في الـ API.
   - `preprocessor_pipeline.pkl`: حفظ الـ Preprocessor بمفرده للاستخدامات المنفصلة.
4. تجهيز وحفظ داتاسيت منظفة كاملة (`cleaned_candidates.csv`):
   - نسخ البيانات الأصلية.
   - تعويض الأعمدة الرقمية باستخدام وسيط التدريب (`median`).
   - تعويض الأعمدة الفئوية (بما فيها `gender` المحفوظ للعرض فقط في الداشبورد للمسؤول) بالقيمة الأكثر تكراراً (`mode`).
   - حفظ الناتج في مجلد `data/processed/`.
5. حفظ ملف البيانات الوصفية (`model_metadata.json`):
   - يضم اسم أفضل موديل، المقياس المعتمد، عتبة القرار (0.50)، تعريف الفئات، الأعمدة المستبعدة، والـ random state.
   - حفظ الملف في مجلد `models/` ليقرأه مسار `/model-info` في FastAPI.
6. طباعة رسالة تؤكد نجاح حفظ جميع ملفات ونماذج المشروع ومساراتها.

#### ب) الهدف العام وليه اتعملت بالطريقة دي:
- **MLOps & Deployment Readiness**: النوت بوك مش مجرد تجربة معزولة بتخلص بانتهاء التشغيل؛ هذه الخلية هي جسر الربط الكامل الذي يُغذي تطبيقات الـ Full-Stack (الـ FastAPI في مجلد `backend/` والـ Streamlit في مجلد `frontend/`).

#### ج) الاستفادة العملية:
- إنتاج أصول (Artifacts) متكاملة تتيح تشغيل السيرفر وعرض الداشبورد فوراً دون الحاجة لإعادة تدريب النماذج في كل مرة.

#### د) تفسير الـ Output:
```
ALL PROJECT ARTIFACTS AND MODELS SAVED SUCCESSFULLY!
Primary Model: Random Forest
Models directory:    ...\models
Results directory:   ...\outputs\results
Figures directory:   ...\outputs\figures
Processed directory: ...\data\processed
```
- تأكيد رسمي على نجاح حفظ كل الملفات وأن الموديل الأساسي المعتمد للمشروع هو **Random Forest**.

#### هـ) الربط الخارجي الكامل:
- **الملفات المنشأة والمحفوظة في هذه الخلية**:
  1. `outputs/results/models_benchmark_metrics.csv`
  2. `models/recruitment_logistic_regression.pkl`
  3. `models/recruitment_random_forest.pkl`
  4. `models/best_recruitment_pipeline.pkl`
  5. `models/preprocessor_pipeline.pkl`
  6. `data/processed/cleaned_candidates.csv`
  7. `models/model_metadata.json`

---

## الملخص الشامل وأهم أسئلة المناقشة المتوقعة
### (Interview & Defense Cheatsheet)

لو سألك البشمهندس في المناقشة، هذه هي إجاباتك النموذجية الواثقة:

---

### س 1: إيه هو هدف المشروع؟ وما هي الداتاسيت المستخدمة؟
- **الإجابة**: المشروع هو **نظام مساعد توظيف ذكي (Smart Recruitment Assistant)**. الهدف منه هو التنبؤ باحتمالية رغبة المرشحين التقنيين (الذين التحقوا بكورسات تدريبية في علوم البيانات والذكاء الاصطناعي) في تغيير وظائفهم والانضمام لشركتنا (`target = 1`) مقابل من لا يرغبون في الانتقال (`target = 0`). الداتاسيت حجمها **19,158 صفاً و 14 عموداً** وتضم بيانات ديموغرافية، تعليمية، خبرات سابقة، وحجم ونوع الشركات وساعات التدريب.

---

### س 2: ليه استبعدتوا أعمدة `enrollee_id`, `city`, `gender` من الموديل؟
- **الإجابة**:
  1. `enrollee_id`: مجرد معرّف تسلسلي عشوائي، حفظه يسبب Overfitting ولا يحمل أي قيمة تنبؤية.
  2. `city`: تضم أكثر من 120 مدينة مختلفة؛ استخدامها سيزيد الأبعاد جداً، واستعضنا عنها بمؤشر تنمية المدينة `city_development_index` الذي يمثل القوة الاقتصادية للمدينة بشكل رقمي متصل.
  3. `gender`: استبعدناه تطبيقاً لمبادئ **الذكاء الاصطناعي العادل والأخلاقي (Fairness & Ethical AI)** لمنع تحيز النموذج ضد أي جنس في قرارات التوظيف الحساسة.

---

### س 3: البيانات غير متوازنة (Imbalanced)، اتعاملتوا مع ده إزاي؟
- **الإجابة**:
  - نسبة الفئات هي **75.07% للفئة (0)** مقابل **24.93% للفئة (1)**.
  - تعاملنا مع الأمر بـ 3 طرق هندسية:
    1. في التقسيم: استخدمنا `stratify=y` لضمان نفس النسبة 75:25 في التدريب والاختبار.
    2. في الخوارزميات: فعلنا `class_weight='balanced'` لرفع عقوبة الخطأ في فئة الأقلية وإجبار الموديل على اصطيادها.
    3. في التقييم: لم نعتمد على الـ Accuracy (لأن خط الأساس الساذج يحقق 75.08%)، بل اعتمدنا على **F1-Score** و **ROC-AUC** و **Recall**.

---

### س 4: إيه هي أهم الميزات المهندسة (Feature Engineering) اللي ضيفتوها للمشروع؟
- **الإجابة**: قمنا بابتكار 7 ميزات نوعية في `src/preprocessing.py`:
  1. `training_hours_log`: تحويل لوغاريتمي `log1p` لإزالة الالتواء اليميني الشديد لساعات التدريب.
  2. `training_intensity`: ساعات التدريب مقسومة على سنوات الخبرة.
  3. `is_company_missing` و `is_company_type_missing`: ميزات ثنائية تشير إلى غياب بيانات الشركة السابقة، وأثبت التحليل أن غيابها يرفع احتمالية الرغبة في التغيير لأكثر من 40%.
  4. `cdi_x_experience`: تفاعل مؤشر المدينة مع الخبرة.
  5. `experience_job_ratio`: نسبة استقرار الوظائف.
  6. `is_low_cdi`: مؤشر المدن منخفضة التنمية (< 0.65).
  - **الدليل على نجاحها**: احتلت 7 ميزات منها قائمة أفضل 10 ميزات في خوارزمية الغابة العشوائية!

---

### س 5: ليه استخدمتوا Scikit-Learn Pipelines؟
- **الإجابة**:
  1. **منع تسريب البيانات (Data Leakage)**: حسابات الـ Imputation والـ Scaling تتم فقط على بيانات التدريب ثم تطبق على الاختبار دون تلوث.
  2. **سهولة النشر (Deployment)**: نقوم بحفظ الـ Pipeline بالكامل في ملف `.pkl` واحد، والـ API يأخذ البيانات الخام ويطبق عليها المعالجة والتنبؤ في خطوة واحدة `pipeline.predict(input_data)`.

---

### س 6: ليه اخترتوا Random Forest كنموذج نهائي بدلاً من Logistic Regression؟
- **الإجابة**:
  - بالرغم من أن كلاهما ممتاز وحقق حساسية استدعاء (Recall) تقارب 80%، إلا أن **Random Forest** تفوقت في:
    - **F1-Score**: حققت `0.6318` مقابل `0.6053`.
    - **ROC-AUC**: حققت `0.8154` مقابل `0.8008`.
    - **Precision**: رفعت الدقة التحديدية إلى `52.65%` مقابل `48.72%`، وقللت الإنذارات الكاذبة (False Positives) من 803 إلى 678.
    - **الدقة الإجمالية (Accuracy)**: حققت `77.06%` وهي أعلى من خط الأساس الساذج (`75.08%`).

---

### س 7: إيه هي المخرجات (Artifacts) اللي بتنتج عن النوت بوك ده ومين بيستفيد منها؟
- **الإجابة**: النوت بوك ينتج بنية إنتاجية كاملة:
  1. موديل الـ Pipeline الأساسي (`models/best_recruitment_pipeline.pkl`) لموديول الـ API (`backend/`).
  2. ملف الميتاداتا (`models/model_metadata.json`) لمسار معلومات النموذج في الـ FastAPI.
  3. ملف المرشحين المنظف (`data/processed/cleaned_candidates.csv`) لعرض المرشحين في داشبورد الـ Streamlit (`frontend/`).
  4. قائمة أفضل 10 مرشحين (`outputs/results/top_10_recommended_candidates.csv`) لتقارير الإدارة ومسؤولي الـ HR.
  5. رسومات ومنحنيات الأداء وأهمية الميزات في `outputs/figures/` للعرض التوثيقي.

---
**تم إعداد التقرير بنجاح وبأعلى درجات الدقة والاحترافية لتغطية كامل تفاصيل النوت بوك.**
