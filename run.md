# MoniTor 复现运行指南（时间异常检测）

本指南面向 **已准备 Normal/Anomaly 视频** 的复现流程，覆盖依赖、路径配置、输入/输出格式与各步骤脚本用法。

---

## 1. 运行流程总览
按顺序执行以下步骤（UCF‑Crime 与 XD‑Violence 结构一致）：

1. **抽帧**：从视频生成帧与 `annotations` 列表  
   - 程序：[`src/preprocessing/extract_frames.py`](src/preprocessing/extract_frames.py)
2. **生成帧字幕**（VLM）  
   - 脚本：[`slurm/ucf_crime/01_caption.sh`](slurm/ucf_crime/01_caption.sh) / [`slurm/xd_violence/01_caption.sh`](slurm/xd_violence/01_caption.sh)
3. **构建字幕索引**（FAISS）  
   - 脚本：[`slurm/ucf_crime/02_create_index.sh`](slurm/ucf_crime/02_create_index.sh) / [`slurm/xd_violence/02_create_index.sh`](slurm/xd_violence/02_create_index.sh)
4. **字幕清洗**（检索一致性）  
   - 脚本：[`slurm/ucf_crime/03_clean_captions.sh`](slurm/ucf_crime/03_clean_captions.sh) / [`slurm/xd_violence/03_clean_captions.sh`](slurm/xd_violence/03_clean_captions.sh)
5. **LLM 在线摘要与打分**（MoniTor 核心）  
   - 脚本：[`slurm/ucf_crime/04_summary.sh`](slurm/ucf_crime/04_summary.sh) / [`slurm/xd_violence/04_summary.sh`](slurm/xd_violence/04_summary.sh)
6. **摘要索引**  
   - 脚本：[`slurm/ucf_crime/05_create_summary_index.sh`](slurm/ucf_crime/05_create_summary_index.sh) / [`slurm/xd_violence/05_create_summary_index.sh`](slurm/xd_violence/05_create_summary_index.sh)
7. **分数精修**  
   - 脚本：[`slurm/ucf_crime/06_refine_anomaly_scores.sh`](slurm/ucf_crime/06_refine_anomaly_scores.sh) / [`slurm/xd_violence/06_refine_anomaly_scores.sh`](slurm/xd_violence/06_refine_anomaly_scores.sh)
8. **评估**  
   - 脚本：[`slurm/ucf_crime/07_neweval.sh`](slurm/ucf_crime/07_neweval.sh) / [`slurm/xd_violence/07_neweval.sh`](slurm/xd_violence/07_neweval.sh)

---

## 2. 依赖与环境配置

### 2.1 依赖安装
- Python 3.10
- 安装依赖：[`requirement.txt`](requirement.txt)

```bash
pip install -r requirement.txt
```

### 2.2 环境变量
- **Qwen3‑VL‑4B 本地路径**
```bash
export MODEL_PATH=/absolute/path/to/Qwen3-VL-4B
```

### 2.3 路径配置
在每个 `slurm/*/*.sh` 中设置：
- `*_dir`：数据集根目录
- `root_path`：帧目录
- `annotationfile_path`：视频列表
- `output_*`：输出目录

---

## 3. 输入数据组织格式

### 3.1 原始视频目录
```
<dataset>/
  videos/
    Normal_xxx.mp4
    Anomaly_xxx.mp4
```

### 3.2 抽帧输出目录
```
<dataset>/
  frames/
    VideoName/
      000001.jpg
      000002.jpg
      ...
```

### 3.3 注释文件格式（测试列表）
以 [ucf/annotations/test.txt](ucf/annotations/test.txt) 结构为例：
```
<video_name> <start> <end> <label>
Normal_Videos_010_x264 0 1052 0
```

### 3.4 时间段异常标注格式
以 [ucf/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt](ucf/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt) 为例：
```
<video>.mp4 <class> <start1> <end1> <start2> <end2>
```

---

## 4. 各步骤脚本指令与参数

### 4.1 抽帧
程序：[`src/preprocessing/extract_frames.py`](src/preprocessing/extract_frames.py)

```bash
python -m src.preprocessing.extract_frames \
  --videos_dir /path/to/videos \
  --frames_dir /path/to/frames \
  --annotations_file /path/to/annotations/test.txt
```

### 4.2 生成帧字幕
脚本：[`slurm/ucf_crime/01_caption.sh`](slurm/ucf_crime/01_caption.sh)  
关键参数：`root_path`、`annotationfile_path`、`pretrained_model_name`、`batch_size`、`frame_interval`

```bash
bash slurm/ucf_crime/01_caption.sh
```

### 4.3 构建字幕索引
脚本：[`slurm/ucf_crime/02_create_index.sh`](slurm/ucf_crime/02_create_index.sh)  
关键参数：`captions_dirs`、`index_dim`

```bash
bash slurm/ucf_crime/02_create_index.sh
```

### 4.4 字幕清洗
脚本：[`slurm/ucf_crime/03_clean_captions.sh`](slurm/ucf_crime/03_clean_captions.sh)  
关键参数：`captions_dir_template`、`index_dir`、`num_neighbors`、`clip_duration`、`fps`

```bash
bash slurm/ucf_crime/03_clean_captions.sh
```

### 4.5 LLM 摘要与评分
脚本：[`slurm/ucf_crime/04_summary.sh`](slurm/ucf_crime/04_summary.sh)  
关键参数：`context_prompt`、`format_prompt`、`captions_dir`、`output_scores_dir`

```bash
bash slurm/ucf_crime/04_summary.sh
```

### 4.6 摘要索引
脚本：[`slurm/ucf_crime/05_create_summary_index.sh`](slurm/ucf_crime/05_create_summary_index.sh)

```bash
bash slurm/ucf_crime/05_create_summary_index.sh
```

### 4.7 分数精修
脚本：[`slurm/ucf_crime/06_refine_anomaly_scores.sh`](slurm/ucf_crime/06_refine_anomaly_scores.sh)  
关键参数：`index_dir`、`scores_dir`、`output_scores_dir`

```bash
bash slurm/ucf_crime/06_refine_anomaly_scores.sh
```

### 4.8 评估
脚本：[`slurm/ucf_crime/07_neweval.sh`](slurm/ucf_crime/07_neweval.sh)

```bash
bash slurm/ucf_crime/07_neweval.sh
```

---

## 5. 输出结果组织格式

### 5.1 字幕与摘要
```
captions/
  raw/<model>/<video>.json
  clean/<index>/<video>.json
  summary/<llm>/<index>/<video>.json
  clean_summary/<llm>/<index>/<video>.json
```

### 5.2 索引
```
index/
  <index_name>/index_flat_ip/<video>.bin
  <index_name>/index_flat_ip/<video>.json
index/summary/<llm>/<index>/index_flat_ip/<video>.bin
```

### 5.3 分数
```
scores/
  raw/<llm>/<index>/<dir>/<video>.json
  refined/<llm>/<index>/<dir>/<video>.json
```

### 5.4 评估指标
```
scores/raw/<llm>/<index>/<dir>/
  roc_auc_nn_<k>.txt
  pr_auc_nn_<k>.txt
```

---

## 6. 关键程序入口（索引）
- 抽帧：[`src/preprocessing/extract_frames.py`](src/preprocessing/extract_frames.py)
- 字幕：[`src/model/image_captioner.py`](src/model/image_captioner.py)
- 字幕索引：[`src/model/create_index.py`](src/model/create_index.py)
- 清洗：[`src/model/image_text_caption_cleaner.py`](src/model/image_text_caption_cleaner.py)
- 在线评分（UCF）：[`src/model/04new.py`](src/model/04new.py)
- 在线评分（XD）：[`src/model/xd_04.py`](src/model/xd_04.py)
- 摘要索引：[`src/model/create_summary_index.py`](src/model/create_summary_index.py)
- 分数精修：[`src/model/video_text_score_refiner.py`](src/model/video_text_score_refiner.py)
- 评估（UCF）：[`src/ucf_eval.py`](src/ucf_eval.py)
- 评估（XD）：[`src/xd_eval.py`](src/xd_eval.py)