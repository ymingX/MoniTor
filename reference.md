# MoniTor API Reference

本文件为项目内所有程序与脚本的 API 参考文档（按经典大型项目 Reference 风格），覆盖命令行入口、核心类与函数、参数与返回值、核心原理与输出产物。

## 约定
- 模块路径使用链接形式表示。
- 符号名使用 `ClassName`/`function_name()` 标注。
- CLI 参数以名称、类型、含义、默认值进行说明。

---

## Slurm 脚本（实验流水线入口）

### 1) UCF‑Crime

#### 脚本：[slurm/ucf_crime/01_caption.sh](slurm/ucf_crime/01_caption.sh)
- **作用**：调用 BLIP2 生成帧级字幕（多模型 array）。
- **调用程序**：`python -m src.models.image_captioner`
- **核心变量**：
  - `root_path`：帧目录根路径。
  - `annotationfile_path`：视频列表。
  - `pretrained_model_name`：BLIP2 模型名（array）。
  - `batch_size`/`frame_interval`：批大小/帧间隔。
- **输出**：`captions/raw/<model>/<video>.json`

#### 脚本：[slurm/ucf_crime/02_create_index.sh](slurm/ucf_crime/02_create_index.sh)
- **作用**：对多模型字幕建立 FAISS 文本索引。
- **调用程序**：`python -m src.models.create_index`
- **核心变量**：
  - `captions_dirs`：多模型字幕目录列表。
  - `index_dim`：向量维度（默认 1024）。
- **输出**：`index/<names>/index_flat_ip/<video>.bin|.json`

#### 脚本：[slurm/ucf_crime/03_clean_captions.sh](slurm/ucf_crime/03_clean_captions.sh)
- **作用**：检索一致性清洗字幕。
- **调用程序**：`python -m src.models.image_text_caption_cleaner`
- **核心变量**：
  - `index_dir`：索引目录。
  - `captions_dir_template`：原始字幕模板。
  - `fps`/`clip_duration`/`num_samples`/`num_neighbors`：检索与采样参数。
- **输出**：`captions/clean/<index>/<video>.json`

#### 脚本：[slurm/ucf_crime/04_summary.sh](slurm/ucf_crime/04_summary.sh)
- **作用**：LLM 在线摘要 + 异常评分。
- **调用程序**：`torchrun -m src.models.04new`
- **核心变量**：
  - `context_prompt`/`format_prompt`：评分提示。
  - `captions_dir`：清洗字幕目录。
  - `output_scores_dir`：raw score 输出目录。
- **输出**：`scores/raw/<llm>/<index>/<dir>/<video>.json`

#### 脚本：[slurm/ucf_crime/05_create_summary_index.sh](slurm/ucf_crime/05_create_summary_index.sh)
- **作用**：对 LLM 摘要建立索引。
- **调用程序**：`python -m src.models.create_summary_index`
- **输出**：`index/summary/<llm>/<index>/index_flat_ip/<video>.bin|.json`

#### 脚本：[slurm/ucf_crime/06_refine_anomaly_scores.sh](slurm/ucf_crime/06_refine_anomaly_scores.sh)
- **作用**：检索增强的分数精修。
- **调用程序**：`python -m src.models.video_text_score_refiner`
- **输出**：
  - refined scores：`scores/refined/<llm>/<index>/<dir>/<video>.json`
  - clean_summary / similarity / filenames 等中间结果。

#### 脚本：[slurm/ucf_crime/07_neweval.sh](slurm/ucf_crime/07_neweval.sh)
- **作用**：UCF‑Crime 指标计算（ROC/PR AUC）。
- **调用程序**：`python -m src.ucf_eval`
- **输出**：`roc_auc_nn_<k>.txt`、`pr_auc_nn_<k>.txt`

### 2) XD‑Violence

#### 脚本：[slurm/xd_violence/01_caption.sh](slurm/xd_violence/01_caption.sh)
- **作用/调用/输出**：同 UCF 01_caption。

#### 脚本：[slurm/xd_violence/02_create_index.sh](slurm/xd_violence/02_create_index.sh)
- **作用/调用/输出**：同 UCF 02_create_index。

#### 脚本：[slurm/xd_violence/03_clean_captions.sh](slurm/xd_violence/03_clean_captions.sh)
- **作用/调用/输出**：同 UCF 03_clean_captions。

#### 脚本：[slurm/xd_violence/04_summary.sh](slurm/xd_violence/04_summary.sh)
- **作用**：XD‑Violence 在线摘要 + 异常评分。
- **调用程序**：`python -u .../src/models/xd_04.py`
- **输出**：`scores/raw/<llm>/<index>/<dir>/<video>.json`

#### 脚本：[slurm/xd_violence/05_create_summary_index.sh](slurm/xd_violence/05_create_summary_index.sh)
- **作用/调用/输出**：同 UCF 05_create_summary_index。

#### 脚本：[slurm/xd_violence/06_refine_anomaly_scores.sh](slurm/xd_violence/06_refine_anomaly_scores.sh)
- **作用/调用/输出**：同 UCF 06_refine_anomaly_scores。

#### 脚本：[slurm/xd_violence/07_neweval.sh](slurm/xd_violence/07_neweval.sh)
- **作用**：XD‑Violence 指标计算（ROC/PR AUC）。
- **调用程序**：`python -m src.xd_eval`
- **输出**：`roc_auc_nn_<k>.txt`、`pr_auc_nn_<k>.txt`

---

## Python 模块 API

### 预处理

#### 模块：[src/preprocessing/extract_frames.py](src/preprocessing/extract_frames.py)
- **功能**：滑窗抽帧并生成 annotations。
- **函数**
  - `extract_frames_with_sliding_window(video_path, frames_dir, window_size, step_size)`
    - **参数**：
      - `video_path`：视频路径。
      - `frames_dir`：输出帧目录。
      - `window_size`：窗口大小（帧数）。
      - `step_size`：窗口步长（帧数）。
    - **返回**：`(video_name, frame_count)`。
    - **原理**：滑窗读取并保存帧为 `{:06d}.jpg`。
  - `main(videos_dir, frames_dir, annotations_file, window_size=16, step_size=8)`
    - **输出**：`annotations_file`，每行 `video_name start end label`。
- **CLI**
  - `--videos_dir`、`--frames_dir`、`--annotations_file`。

### 数据结构

#### 模块：[src/data/video_record.py](src/data/video_record.py)
- **类**：`VideoRecord`
  - **作用**：封装视频样本元数据。
  - **属性**：
    - `path`：帧目录路径。
    - `num_frames`：帧数（含起止）。
    - `start_frame` / `end_frame`。
    - `label`：标签列表。

### 模型与核心算法

#### 模块：[src/model/image_captioner.py](src/model/image_captioner.py)
- **类**：`ImageCaptioner`
  - `__init__(batch_size, frame_interval, imagefile_template, pretrained_model_name, dtype_str, output_dir)`
    - **作用**：初始化 BLIP2 与处理器。
  - `process_video(video)`
    - **作用**：批量推理生成帧字幕 JSON。
    - **输出**：`captions/raw/<model>/<video>.json`。
- **函数**：
  - `run(...)`：批量处理视频列表。
  - `parse_args()`：CLI 解析。
- **CLI 参数**：`--root_path`、`--annotationfile_path`、`--batch_size`、`--frame_interval`、`--imagefile_template`、`--pretrained_model_name`、`--output_dir`、`--dtype`、`--resume`、`--pathname`。

#### 模块：[src/model/create_index.py](src/model/create_index.py)
- **功能**：多模型字幕向量化 + FAISS 索引。
- **函数**：
  - `parse_args()`：读取 `--index_dim`、`--captions_dirs` 等。
  - `load_video_records(annotationfile_path, root_path)`
  - `process_video(video, model, device, index_dim, batch_size, frame_interval, captions_dirs, output_dir)`
    - **原理**：对唯一字幕集合进行 ImageBind 文本编码，写入 `IndexFlatIP`。
  - `init_faiss_index(index_dim)`
  - `load_video_captions(captions_dirs, video_name)`
  - `build_caption_to_frame_index(video_captions)`
  - `extract_text_list(...)`
  - `update_faiss_index(model, device, index, text_list)`
  - `build_file_names(...)`
  - `save_results(index, file_names, output_dir, video_name)`
  - `main(...)`：入口逻辑。

#### 模块：[src/model/image_text_caption_cleaner.py](src/model/image_text_caption_cleaner.py)
- **类**：`ImageTextCaptionCleaner`
  - **作用**：基于 ImageBind 视觉检索清洗字幕。
  - **核心方法**：
    - `process_video(video)`：对每个 clip 检索并重构字幕。
    - `_prepare_frame_data()`：窗口采样与帧路径构造。
    - `_load_and_transform_data()`：加载视频帧并变换为模型输入。
    - `_retrieve_captions()`：按检索结果回填字幕。
- **函数**：`run(...)`、`parse_args()`。
- **CLI 参数**：`--captions_dir_template`、`--index_dir`、`--fps`、`--clip_duration`、`--num_samples`、`--num_neighbors`、`--num_jobs`、`--job_id` 等。

#### 模块：[src/model/04new.py](src/model/04new.py)
- **类**：`GLMAnomalyScorer`
  - **功能**：UCF‑Crime 在线摘要与异常评分。
  - **核心方法**：
    - `_parse_score(response)`：解析并量化评分。
    - `_update_score_queue(frame_idx, score)`：维护评分队列。
    - `_lstm_summarize(texts)`：LLM 摘要。
    - `_calculate_similarity(text1, text2)`：TF‑IDF 相似度。
    - `_prepare_memory_summaries(captions, current_idx)`：长/短期记忆摘要。
    - `_prepare_score_prompts(captions)`：构建评分队列提示。
    - `_prepare_dialogs(captions, batch_frame_idxs)`：构造评分对话。
    - `_score_temporal_summaries(video, temporal_captions)`：调用 LLM 评分并平滑。
    - `process_video(video)`：输出单视频 JSON 结果。
- **函数**：`run(...)`、`parse_args()`。
- **CLI 参数**：`--context_prompt`、`--format_prompt`、`--captions_dir`、`--output_scores_dir`、`--api_key`、`--num_jobs` 等。

#### 模块：[src/model/xd_04.py](src/model/xd_04.py)
- **作用**：XD‑Violence 在线摘要与评分，结构与 [src/model/04new.py](src/model/04new.py) 一致。

#### 模块：[src/model/create_summary_index.py](src/model/create_summary_index.py)
- **功能**：LLM 摘要向量化并建立 FAISS 索引。
- **核心函数**：
  - `initialize_faiss_index()` / `add_text_to_index()` / `save_results()` / `process_video()`。
  - `main(...)` / `parse_args()`。

#### 模块：[src/model/video_text_score_refiner.py](src/model/video_text_score_refiner.py)
- **类**：`VideoTextScoreRefiner`
  - **作用**：检索增强的分数精修。
  - **核心方法**：
    - `retrieve_nn(video)`：检索近邻摘要并保存 similarity/indices/filenames。
    - `refine_scores(video)`：依据检索邻居融合 raw scores。
    - `_prepare_frame_data()` / `_load_and_transform_data()` / `_calculate_search_vectors()`。
- **函数**：`run(...)`、`parse_args()`。
- **CLI 参数**：`--output_scores_dir`、`--output_summary_dir`、`--output_similarity_dir`、`--output_indices_dir`、`--output_filenames_dir`、`--scores_dir`、`--index_dir`、`--num_neighbors` 等。

### 评估

#### 模块：[src/ucf_eval.py](src/ucf_eval.py)
- **功能**：UCF‑Crime 指标计算与可视化。
- **核心函数**：
  - `temporal_testing_annotations(...)`：读取时间段标注。
  - `replace_negative_scores_with_previous(...)`：替换 -1。
  - `get_video_labels(...)`：生成帧级标签。
  - `calculate_weighted_scores(...)`：窗口加权分数。
  - `main(...)`：计算 ROC/PR AUC。
- **CLI 参数**：`--temporal_annotation_file`、`--raw_scores_dir`、`--captions_dir`、`--num_neighbors`、`--video_fps` 等。

#### 模块：[src/xd_eval.py](src/xd_eval.py)
- **功能**：XD‑Violence 指标计算，与 [src/ucf_eval.py](src/ucf_eval.py) 对称。

### 工具函数

#### 模块：[src/utils/image_utiles.py](src/utils/image_utils.py)
- `load_image_from_path(img_path)`：读取单张图像为 PIL。
- `load_images_from_paths(img_paths)`：批量读取图像。

#### 模块：[src/utils/plot_utiles.py](src/utils/plot_utils.py)
- `plot_scores(scores, labels, video_name, save_dir, normal_id=7)`
  - **作用**：绘制异常分数曲线并覆盖异常区间。

#### 模块：[src/utils/sample_utiles.py](src/utils/sample_utils.py)
- `uniform_temporal_subsample(clip_frame_paths, num_samples)`
  - **作用**：等距采样帧路径。

#### 模块：[src/utils/torch_utiles.py](src/utils/torch_utils.py)
- `initialize_vlm_model_and_device()`
  - **作用**：加载 ImageBind 模型并返回 `model, device`。

#### 模块：[src/utils/path_utiles.py](src/utils/path_utils.py)
- `find_unprocessed_videos(video_list, output_dir, pathname)`
  - **作用**：断点续跑，跳过已处理样本。
- `find_last_processed_video_index(video_list, last_processed_video_path)`

#### 模块：[src/utils/visual.py](src/utils/visual.py)
- `visual_lys_2_func(video_name, scores, labels)`
  - **作用**：可视化异常曲线与区间并保存图片。

---

## 入口与调用关系速览
1. 抽帧 → [src/preprocessing/extract_frames.py](src/preprocessing/extract_frames.py)
2. 字幕生成 → [src/model/image_captioner.py](src/model/image_captioner.py)
3. 索引构建 → [src/model/create_index.py](src/model/create_index.py)
4. 字幕清洗 → [src/model/image_text_caption_cleaner.py](src/model/image_text_caption_cleaner.py)
5. LLM 摘要与评分 → [src/model/04new.py](src/model/04new.py) / [src/model/xd_04.py](src/model/xd_04.py)
6. 摘要索引 → [src/model/create_summary_index.py](src/model/create_summary_index.py)
7. 分数精修 → [src/model/video_text_score_refiner.py](src/model/video_text_score_refiner.py)
8. 评估 → [src/ucf_eval.py](src/ucf_eval.py) / [src/xd_eval.py](src/xd_eval.py)
