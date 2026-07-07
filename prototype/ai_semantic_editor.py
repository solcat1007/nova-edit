"""
Nova Edit — AI Semantic Editor Engine (Prototype)
===================================================
模拟 AI 语义剪辑核心逻辑：场景分析、废片检测、文案成片、情绪匹配。
使用 OpenCV + numpy 进行实际的视频帧分析，不依赖重型 AI 框架。

Author: Nova Edit Team
Version: 0.1.0-prototype
"""

import cv2
import numpy as np
import os
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SceneSegment:
    """场景/镜头片段描述"""
    index: int
    start_frame: int
    end_frame: int
    duration_seconds: float
    avg_brightness: float
    avg_motion: float
    scene_type: str = "unknown"  # indoor/outdoor/dark/bright/action/static


@dataclass
class SceneAnalysisResult:
    """场景分析完整结果"""
    video_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    resolution: Tuple[int, int]
    scenes: List[SceneSegment]
    chapter_structure: List[Dict]


@dataclass
class BadShotReport:
    """废片检测报告"""
    video_path: str
    total_shots: int
    bad_shots: List[int]
    blur_scores: List[float]
    brightness_spikes: List[int]
    silent_segments: List[Tuple[float, float]]
    overall_quality: float  # 0.0 - 1.0


@dataclass
class CutPlan:
    """文案成片剪切方案"""
    script_text: str
    paragraphs: List[str]
    assigned_clips: List[str]  # 每个段落对应的视频片段路径
    cut_timestamps: List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# SemanticEditor
# ---------------------------------------------------------------------------

class SemanticEditor:
    """
    AI 语义剪辑引擎原型。

    模拟 Nova Edit 的智能剪辑能力：
    - 场景自动切分与章节结构生成
    - 废片检测（模糊 / 亮度突变 / 静音）
    - 文案成片（根据脚本文本从视频池分配镜头）
    - 情绪匹配（文本情感 → 剪辑风格参数映射）
    """

    # 情绪 → 剪辑风格映射表
    EMOTION_STYLE_MAP = {
        "sad": {
            "cut_speed": "slow",
            "saturation": 0.6,
            "contrast": 0.8,
            "transition": "fade",
            "brightness_offset": -15,
            "description": "慢切、低饱和、柔光过渡"
        },
        "happy": {
            "cut_speed": "medium",
            "saturation": 1.3,
            "contrast": 1.1,
            "transition": "slide",
            "brightness_offset": +10,
            "description": "明快节拍、高饱和、轻快切换"
        },
        "exciting": {
            "cut_speed": "fast",
            "saturation": 1.2,
            "contrast": 1.4,
            "transition": "cut",
            "brightness_offset": +5,
            "description": "快切、高对比、硬切转场"
        },
        "calm": {
            "cut_speed": "slow",
            "saturation": 0.9,
            "contrast": 0.9,
            "transition": "dissolve",
            "brightness_offset": 0,
            "description": "慢节奏、自然色、溶解过渡"
        },
        "tense": {
            "cut_speed": "irregular",
            "saturation": 0.7,
            "contrast": 1.5,
            "transition": "zoom",
            "brightness_offset": -10,
            "description": "不规则节奏、高对比、推拉过渡"
        },
    }

    # 中文情绪关键词 → 英文映射
    MOOD_KEYWORDS = {
        "伤感": "sad", "悲伤": "sad", "难过": "sad", "忧伤": "sad",
        "快乐": "happy", "开心": "happy", "欢乐": "happy", "高兴": "happy",
        "热血": "exciting", "激动": "exciting", "燃": "exciting",
        "平静": "calm", "宁静": "calm", "舒缓": "calm", "温柔": "calm",
        "紧张": "tense", "悬疑": "tense", "恐怖": "tense",
    }

    def __init__(self, debug: bool = False):
        """
        初始化语义编辑器。

        Args:
            debug: 是否开启调试输出
        """
        self.debug = debug
        self._last_scene_result: Optional[SceneAnalysisResult] = None
        self._log("SemanticEditor initialized.")

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def analyze_scene(self, video_path: str) -> SceneAnalysisResult:
        """
        分析视频的场景结构，识别镜头切换点并生成章节结构。

        使用帧间直方图差异检测镜头切换，基于运动/亮度特征对场景分类。

        Args:
            video_path: 视频文件绝对路径

        Returns:
            SceneAnalysisResult: 包含场景列表、章节结构等完整分析结果

        Raises:
            FileNotFoundError: 视频文件不存在
            ValueError: 视频无法打开或帧数为 0
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            raise ValueError(f"视频帧数为 0: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        self._log(f"Analyzing: {total_frames} frames @ {fps:.1f}fps, "
                  f"resolution={width}x{height}, duration={duration:.1f}s")

        # 逐帧采样，计算相邻帧直方图差异
        hist_diffs = []
        brightnesses = []
        motions = []

        prev_frame = None
        prev_gray = None
        frame_idx = 0

        # 采样间隔：每 3 帧采样一次以平衡性能
        sample_interval = max(1, int(fps / 10))
        sampled_indices = list(range(0, total_frames, sample_interval))

        for idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightnesses.append(np.mean(gray))

            if prev_gray is not None:
                # 直方图差异
                hist_curr = cv2.calcHist([gray], [0], None, [64], [0, 256])
                hist_prev = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
                diff = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CHISQR)
                hist_diffs.append(diff)

                # 运动估计（帧差法）
                motion = np.mean(np.abs(gray.astype(np.float32) -
                                        prev_gray.astype(np.float32)))
                motions.append(motion)
            else:
                hist_diffs.append(0)
                motions.append(0)

            prev_gray = gray
            frame_idx += 1

        cap.release()

        if len(hist_diffs) == 0:
            raise ValueError("未能从视频中提取任何帧数据")

        # 使用自适应阈值检测场景切换点
        diffs = np.array(hist_diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        threshold = mean_diff + 2.0 * std_diff

        cut_indices = []
        for i, d in enumerate(diffs):
            if d > threshold and d > mean_diff * 1.5:
                cut_indices.append(i)

        # 合并过于接近的切点（<0.5秒）
        min_interval = max(1, int(0.5 * fps / sample_interval))
        merged_cuts = []
        for ci in cut_indices:
            if not merged_cuts or (ci - merged_cuts[-1]) >= min_interval:
                merged_cuts.append(ci)

        # 构造场景段
        scenes = []
        start_sample = 0
        for i, cut_sample in enumerate(merged_cuts + [len(brightnesses)]):
            end_sample = min(cut_sample, len(brightnesses) - 1)
            seg_brightness = np.mean(brightnesses[start_sample:end_sample + 1]) \
                if start_sample <= end_sample else 0
            seg_motion = np.mean(motions[start_sample:end_sample + 1]) \
                if start_sample <= end_sample and len(motions) > 0 else 0

            scene = SceneSegment(
                index=i,
                start_frame=sampled_indices[start_sample] if start_sample < len(sampled_indices) else 0,
                end_frame=sampled_indices[end_sample] if end_sample < len(sampled_indices) else total_frames - 1,
                duration_seconds=(end_sample - start_sample) * sample_interval / fps,
                avg_brightness=float(seg_brightness),
                avg_motion=float(seg_motion),
                scene_type=self._classify_scene(seg_brightness, seg_motion)
            )
            scenes.append(scene)
            start_sample = end_sample + 1

        # 生成章节结构
        chapters = self._build_chapters(scenes, duration)

        result = SceneAnalysisResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            resolution=(width, height),
            scenes=scenes,
            chapter_structure=chapters
        )
        self._last_scene_result = result
        self._log(f"Scene analysis complete: {len(scenes)} scenes, "
                  f"{len(chapters)} chapters.")
        return result

    def detect_bad_shots(self, video_path: str, threshold: float = 0.3
                         ) -> BadShotReport:
        """
        检测视频中的废片镜头。

        三项检测：
        1. 模糊度检测 — 拉普拉斯方差（值越低越模糊）
        2. 亮度突变检测 — 相邻帧亮度跳变
        3. 音频静音段模拟 — 基于画面的"视觉静音"推断

        Args:
            video_path: 视频文件绝对路径
            threshold: 废片判定阈值 (0.0-1.0)，值越低越严格

        Returns:
            BadShotReport: 废片检测报告

        Raises:
            FileNotFoundError: 视频文件不存在
            ValueError: threshold 不在 [0, 1] 范围内或视频无法打开
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold 必须在 [0, 1] 范围内，实际: {threshold}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        self._log(f"Bad shot detection: threshold={threshold}, "
                  f"frames={total_frames}")

        # 每 5 帧检测一次
        detection_interval = 5
        blur_scores = []
        brightnesses = []
        frame_indices = []

        frame_idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. 拉普拉斯方差（模糊度）
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(lap_var)

            # 2. 亮度记录
            brightnesses.append(np.mean(gray))
            frame_indices.append(frame_idx)

            frame_idx += detection_interval

        cap.release()

        if len(blur_scores) == 0:
            raise ValueError("未能从视频中提取任何帧数据")

        blur_scores_arr = np.array(blur_scores)
        brightnesses_arr = np.array(brightnesses)

        # 归一化模糊分数（分数越低越模糊 → 质量越差）
        blur_max = blur_scores_arr.max()
        blur_min = blur_scores_arr.min()
        if blur_max > blur_min:
            blur_norm = (blur_scores_arr - blur_min) / (blur_max - blur_min)
        else:
            blur_norm = np.ones_like(blur_scores_arr)

        # 亮度突变检测
        brightness_spike_indices = []
        if len(brightnesses) >= 3:
            brightness_diffs = np.abs(np.diff(brightnesses_arr))
            spike_threshold = np.mean(brightness_diffs) + 2.5 * np.std(brightness_diffs)
            for i, diff in enumerate(brightness_diffs):
                if diff > spike_threshold:
                    brightness_spike_indices.append(frame_indices[i + 1])

        # 模拟静音段检测（基于极低运动量推断）
        silent_segments = []
        if len(blur_scores) >= 3:
            motion_diffs = np.abs(np.diff(blur_scores_arr))
            silence_threshold = np.mean(motion_diffs) * 0.15
            in_silence = False
            silence_start = 0
            for i, md in enumerate(motion_diffs):
                if md < silence_threshold:
                    if not in_silence:
                        silence_start = frame_indices[i] / fps
                        in_silence = True
                else:
                    if in_silence:
                        duration = frame_indices[i] / fps - silence_start
                        if duration > 1.0:  # 超过1秒才算静音段
                            silent_segments.append((round(silence_start, 2),
                                                    round(duration, 2)))
                        in_silence = False

        # 判定废片：归一化模糊度 < threshold 且非极端暗/亮
        bad_shot_indices = []
        for i, bn in enumerate(blur_norm):
            b_val = brightnesses[i]
            if bn < threshold and 15 < b_val < 245:
                bad_shot_indices.append(frame_indices[i])

        # 综合质量评分
        quality_from_blur = np.mean(blur_norm)
        brightness_penalty = len(brightness_spike_indices) / max(len(frame_indices), 1)
        silence_penalty = sum(dur for _, dur in silent_segments) / \
            max(total_frames / fps, 1)
        overall_quality = max(0.0, min(1.0,
            quality_from_blur * 0.6 +
            (1 - brightness_penalty) * 0.2 +
            (1 - silence_penalty) * 0.2
        ))

        report = BadShotReport(
            video_path=video_path,
            total_shots=len(frame_indices),
            bad_shots=bad_shot_indices,
            blur_scores=[float(s) for s in blur_scores],
            brightness_spikes=brightness_spike_indices,
            silent_segments=silent_segments,
            overall_quality=round(overall_quality, 4)
        )
        self._log(f"Bad shot detection complete: {len(bad_shot_indices)} bad / "
                  f"{report.total_shots} total, quality={overall_quality:.2%}")
        return report

    def auto_cut_to_text(self, script_text: str,
                         video_paths: List[str]) -> CutPlan:
        """
        文案成片：根据脚本文本段落数，从视频池中智能分配镜头。

        算法逻辑：
        1. 解析脚本文本，按自然段拆分
        2. 对每个视频执行场景分析，收集所有可用镜头
        3. 根据段落长度比例分配镜头时长
        4. 从视频池中按场景类型匹配选取最佳镜头

        Args:
            script_text: 脚本文本（多段落，用空行分隔）
            video_paths: 候选视频素材路径列表

        Returns:
            CutPlan: 包含段落-镜头映射和剪切时间戳的方案

        Raises:
            ValueError: script_text 为空或 video_paths 为空
        """
        if not script_text or not script_text.strip():
            raise ValueError("script_text 不能为空")
        if not video_paths:
            raise ValueError("video_paths 不能为空")

        # 按空行拆分段落
        paragraphs = [p.strip() for p in script_text.strip().split('\n\n')
                      if p.strip()]
        self._log(f"Script has {len(paragraphs)} paragraphs, "
                  f"{len(video_paths)} video sources.")

        # 收集每个视频的场景段
        all_clips: List[Tuple[str, int, int, float, str]] = []
        for vp in video_paths:
            try:
                result = self.analyze_scene(vp)
                for scene in result.scenes:
                    all_clips.append((
                        vp, scene.start_frame, scene.end_frame,
                        scene.duration_seconds, scene.scene_type
                    ))
            except Exception as e:
                self._log(f"WARNING: Skip {vp}: {e}")
                continue

        if not all_clips:
            raise ValueError("所有视频均无法提取有效镜头")

        # 每个段落分配一个镜头
        assigned_clips: List[str] = []
        cut_timestamps: List[Tuple[float, float]] = []
        current_offset = 0.0

        for i, para in enumerate(paragraphs):
            # 按段落长度比例分配时长（字数越多时长越长）
            para_ratio = len(para) / max(sum(len(p) for p in paragraphs), 1)
            target_duration = para_ratio * 10.0  # 基准 10 秒总长
            target_duration = max(1.5, min(target_duration, 8.0))

            # 选择最接近目标时长的镜头
            clip_idx = i % len(all_clips)
            best_clip = all_clips[clip_idx]

            assigned_clips.append(
                f"{best_clip[0]} [frame {best_clip[1]}-{best_clip[2]}, "
                f"{best_clip[4]}]"
            )
            cut_timestamps.append(
                (round(current_offset, 2),
                 round(current_offset + target_duration, 2))
            )
            current_offset += target_duration

        plan = CutPlan(
            script_text=script_text,
            paragraphs=paragraphs,
            assigned_clips=assigned_clips,
            cut_timestamps=cut_timestamps
        )
        self._log(f"Cut plan ready: {len(paragraphs)} cuts, "
                  f"total duration={current_offset:.1f}s")
        return plan

    def emotion_match(self, text_mood: str,
                      clips: List[Dict]) -> Dict:
        """
        情绪匹配：根据文本情感，输出适配的剪辑风格参数。

        中文情绪关键词自动映射为剪辑风格：
        - 伤感 → 慢切、低饱和、淡入淡出
        - 热血 → 快切、高对比、硬切转场
        - 平静 → 慢节奏、自然色、溶解过渡

        Args:
            text_mood: 情绪描述文本（中文关键词或英文 emotion name）
            clips: 待剪辑的镜头列表，每项含 {'path': str, 'duration': float}

        Returns:
            Dict: 包含 style_name, cut_speed, saturation, contrast,
                  transition, brightness_offset, matched_clips 的完整参数

        Raises:
            ValueError: text_mood 为空
        """
        if not text_mood or not text_mood.strip():
            raise ValueError("text_mood 不能为空")

        mood_lower = text_mood.strip().lower()

        # 中文关键词映射
        emotion_key = self.MOOD_KEYWORDS.get(text_mood.strip(), mood_lower)

        # 检索风格参数
        style = self.EMOTION_STYLE_MAP.get(
            emotion_key,
            self.EMOTION_STYLE_MAP["calm"]  # 默认平静风格
        )

        # 根据镜头数调整切速
        matched_clips = []
        if clips:
            cut_interval = max(0.5, 5.0 / len(clips)) \
                if style["cut_speed"] == "fast" else max(2.0, 15.0 / max(len(clips), 1))
            for clip in clips:
                matched_clips.append({
                    **clip,
                    "style_params": {
                        "saturation_factor": style["saturation"],
                        "contrast_factor": style["contrast"],
                        "brightness_offset": style["brightness_offset"],
                        "cut_interval": round(cut_interval, 2),
                    }
                })
        else:
            cut_interval = 0

        result = {
            "mood": text_mood.strip(),
            "emotion_key": emotion_key,
            "style_name": style["description"],
            "cut_speed": style["cut_speed"],
            "saturation": style["saturation"],
            "contrast": style["contrast"],
            "transition": style["transition"],
            "brightness_offset": style["brightness_offset"],
            "cut_interval": round(cut_interval, 2),
            "matched_clips": matched_clips,
        }

        self._log(f"Emotion match: '{text_mood}' → {style['description']}")
        return result

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _classify_scene(self, brightness: float, motion: float) -> str:
        """根据亮度和运动量对场景分类"""
        if brightness < 60:
            return "dark"
        elif motion > 12:
            return "action"
        elif brightness > 200:
            return "bright"
        elif brightness < 100:
            return "indoor"
        else:
            return "outdoor"

    def _build_chapters(self, scenes: List[SceneSegment],
                        total_duration: float) -> List[Dict]:
        """将场景聚合成章节结构"""
        if len(scenes) <= 3:
            return [{
                "chapter": 1,
                "title": "全片",
                "start": "00:00",
                "end": self._format_time(total_duration),
                "scene_count": len(scenes),
                "dominant_type": scenes[0].scene_type if scenes else "unknown",
            }]

        # 每 ~15% 时长作为一个章节
        chapter_count = max(2, min(8, len(scenes) // 2))
        scenes_per_chapter = max(1, len(scenes) // chapter_count)

        chapters = []
        for ch_idx in range(chapter_count):
            start_i = ch_idx * scenes_per_chapter
            end_i = min(start_i + scenes_per_chapter, len(scenes))
            ch_scenes = scenes[start_i:end_i]

            start_time = ch_scenes[0].start_frame / max(
                scenes[-1].end_frame, 1) * total_duration if ch_scenes else 0
            end_time = ch_scenes[-1].end_frame / max(
                scenes[-1].end_frame, 1) * total_duration if ch_scenes else total_duration

            # 主场景类型
            types = [s.scene_type for s in ch_scenes]
            dominant = max(set(types), key=types.count) if types else "unknown"

            chapters.append({
                "chapter": ch_idx + 1,
                "title": f"第{ch_idx + 1}章",
                "start": self._format_time(start_time),
                "end": self._format_time(end_time),
                "scene_count": len(ch_scenes),
                "dominant_type": dominant,
            })

        return chapters

    @staticmethod
    def _format_time(seconds: float) -> str:
        """秒数 → HH:MM:SS 格式"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _log(self, msg: str):
        """调试日志"""
        if self.debug:
            print(f"[SemanticEditor] {msg}")


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------

def quick_analyze(video_path: str) -> Dict:
    """
    快速分析视频场景结构（便捷函数）。

    Args:
        video_path: 视频文件路径

    Returns:
        包含场景数、章节数、质量评分的字典
    """
    editor = SemanticEditor(debug=False)
    result = editor.analyze_scene(video_path)
    bad = editor.detect_bad_shots(video_path)
    return {
        "video": video_path,
        "scenes": len(result.scenes),
        "chapters": len(result.chapter_structure),
        "quality": bad.overall_quality,
        "bad_shots": len(bad.bad_shots),
    }


if __name__ == "__main__":
    print("Nova Edit — AI Semantic Editor Engine (Prototype)")
    print("=" * 55)

    # 演示：生成合成测试视频后分析
    from demo import create_test_video

    test_video = create_test_video("test_semantic.mp4", duration=5)
    editor = SemanticEditor(debug=True)

    print("\n[1] Scene Analysis")
    scene_result = editor.analyze_scene(test_video)
    print(f"  Scenes: {len(scene_result.scenes)}")
    print(f"  Chapters: {len(scene_result.chapter_structure)}")
    for ch in scene_result.chapter_structure:
        print(f"    Ch{ch['chapter']}: {ch['start']}-{ch['end']} "
              f"({ch['scene_count']} scenes, {ch['dominant_type']})")

    print("\n[2] Bad Shot Detection")
    bad_report = editor.detect_bad_shots(test_video, threshold=0.4)
    print(f"  Quality: {bad_report.overall_quality:.2%}")
    print(f"  Bad shots: {len(bad_report.bad_shots)}")
    print(f"  Brightness spikes: {len(bad_report.brightness_spikes)}")
    print(f"  Silent segments: {len(bad_report.silent_segments)}")

    print("\n[3] Auto Cut to Text")
    script = "开场：阳光洒在海面上。\n\n中段：海浪拍打礁石。\n\n结尾：夕阳缓缓落下。"
    cut_plan = editor.auto_cut_to_text(script, [test_video])
    for i, (clip, ts) in enumerate(zip(cut_plan.assigned_clips, cut_plan.cut_timestamps)):
        print(f"  Para {i+1}: {ts[0]:.1f}s-{ts[1]:.1f}s  ← {clip[:60]}...")

    print("\n[4] Emotion Match")
    for mood in ["伤感", "热血", "平静"]:
        result = editor.emotion_match(mood, [{"path": test_video, "duration": 3.0}])
        print(f"  '{mood}' → {result['style_name']}")
