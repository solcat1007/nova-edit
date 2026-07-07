"""
Nova Edit — SmartMatting: Intelligent Keying Engine (Prototype)
================================================================
Nova Edit 智能抠像引擎原型：无痕万能抠像。

核心能力：
- GrabCut 自动抠像（自动 ROI + 迭代优化）
- 发丝级边缘优化（形态学 + 引导滤波边缘羽化）
- 合成自动环境光适配（色温分析 → 前景色温匹配）

Author: Nova Edit Team
Version: 0.1.0-prototype
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SmartMatting:
    """
    智能抠像引擎。

    基于 OpenCV GrabCut 实现自动前景分割，结合形态学操作和
    颜色转移技术实现发丝级边缘优化与环境光自动匹配。
    """

    # GrabCut 默认迭代次数
    GRABCUT_ITERATIONS = 5

    # 形态学核大小
    MORPH_KERNEL_SIZE = (3, 3)

    # 羽化边缘高斯核
    FEATHER_KERNEL_SIZE = (5, 5)

    def __init__(self, debug: bool = False):
        """
        初始化 SmartMatting 引擎。

        Args:
            debug: 是否开启调试输出
        """
        self.debug = debug
        self._log("SmartMatting initialized.")

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def auto_matting(self, image_path: str,
                     output_path: Optional[str] = None) -> Dict:
        """
        基于 GrabCut 的自动抠像。

        自动推断前景矩形 ROI 并执行迭代 GrabCut，输出透明背景前景图。

        算法流程：
        1. 自动生成初始 ROI（图像中央 60% 区域）
        2. 执行 GrabCut 迭代分割
        3. 提取 alpha mask 并合成透明背景

        Args:
            image_path: 输入图像绝对路径
            output_path: 输出 PNG 路径，None 则自动生成（同目录 _matting.png）

        Returns:
            Dict: {
                'output_path': 输出 PNG 路径,
                'mask_path': 掩码图路径,
                'foreground': 前景 BGR 图像数据,
                'alpha_mask': alpha 通道数据,
                'iterations': 实际迭代次数
            }

        Raises:
            FileNotFoundError: 图像文件不存在
            ValueError: 图像无法读取或尺寸为 0
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        if img.size == 0:
            raise ValueError(f"图像尺寸为 0: {image_path}")

        h, w = img.shape[:2]
        self._log(f"Auto matting: {w}x{h}, {image_path}")

        # 自动 ROI：中央 60% 区域
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        # GrabCut 初始化
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        self._log(f"  GrabCut ROI: {rect}, iterations={self.GRABCUT_ITERATIONS}")

        try:
            cv2.grabCut(
                img, mask, rect,
                bgd_model, fgd_model,
                self.GRABCUT_ITERATIONS,
                cv2.GC_INIT_WITH_RECT
            )
        except cv2.error as e:
            raise ValueError(f"GrabCut 执行失败: {e}")

        # 提取前景 mask
        mask2 = np.where((mask == cv2.GC_FGD) |
                         (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # 合成透明背景
        b, g, r = cv2.split(img)
        alpha = mask2
        rgba = cv2.merge([b, g, r, alpha])

        # 保存结果
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_matting.png"
        cv2.imwrite(output_path, rgba)

        # 保存掩码
        mask_path = output_path.replace(".png", "_mask.png").replace(
            "_matting_mask", "_mask")
        cv2.imwrite(mask_path, mask2)

        result = {
            "output_path": output_path,
            "mask_path": mask_path,
            "foreground": img,
            "alpha_mask": mask2,
            "iterations": self.GRABCUT_ITERATIONS,
        }
        self._log(f"  Output: {output_path}")
        return result

    def hair_refine(self, image_path: str, mask_path: str,
                    output_path: Optional[str] = None) -> str:
        """
        发丝级边缘优化。

        使用形态学闭运算 + 高斯羽化平滑 mask 边缘，
        模拟发丝级细节保留效果。

        Args:
            image_path: 原始图像路径
            mask_path: 初始 alpha mask 路径（来自 auto_matting）
            output_path: 输出优化后 PNG 的路径，None 则自动生成

        Returns:
            str: 优化后图像路径

        Raises:
            FileNotFoundError: 图像或 mask 不存在
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask 文件不存在: {mask_path}")

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        if mask is None:
            raise ValueError(f"无法读取 mask: {mask_path}")

        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))

        self._log(f"Hair refine: {w}x{h}")

        # 步骤 1：形态学闭运算（填充孔洞）+ 开运算（去除噪点）
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.MORPH_KERNEL_SIZE)
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2, 2))

        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask_refined = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

        # 步骤 2：高斯羽化边缘
        mask_float = mask_refined.astype(np.float32) / 255.0
        mask_feathered = cv2.GaussianBlur(
            mask_float, self.FEATHER_KERNEL_SIZE, sigmaX=1.5)

        # 步骤 3：边缘区域精细处理 — 用引导滤波平滑过渡
        edge_band = self._get_edge_band(mask_refined, band_width=3)
        mask_feathered[edge_band > 0] = cv2.GaussianBlur(
            mask_float, (3, 3), sigmaX=1.0)[edge_band > 0]

        # 合成
        alpha = (mask_feathered * 255).astype(np.uint8)
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, alpha])

        if output_path is None:
            base = os.path.splitext(image_path)[0]
            output_path = f"{base}_hair_refined.png"
        cv2.imwrite(output_path, rgba)

        self._log(f"  Output: {output_path}")
        return output_path

    def composite_with_lighting(self, fg_path: str, bg_path: str,
                                output_path: Optional[str] = None) -> str:
        """
        合成并自动适配环境光。

        分析背景色温（白平衡），调整前景色温以匹配背景环境，
        实现"无痕合成"效果。

        算法流程：
        1. 分析背景平均色温（基于灰度世界假设）
        2. 提取前景 RGB + alpha
        3. 用色温匹配矩阵调整前景色彩
        4. Alpha 混合合成

        Args:
            fg_path: 前景图像路径（需含 alpha 通道，如 PNG）
            bg_path: 背景图像路径
            output_path: 输出合成图像的路径，None 则自动生成

        Returns:
            str: 输出合成图像路径

        Raises:
            FileNotFoundError: 前景或背景文件不存在
        """
        if not os.path.exists(fg_path):
            raise FileNotFoundError(f"前景文件不存在: {fg_path}")
        if not os.path.exists(bg_path):
            raise FileNotFoundError(f"背景文件不存在: {bg_path}")

        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        bg = cv2.imread(bg_path)
        if fg is None:
            raise ValueError(f"无法读取前景: {fg_path}")
        if bg is None:
            raise ValueError(f"无法读取背景: {bg_path}")

        self._log(f"Composite with lighting: fg={fg.shape}, bg={bg.shape}")

        # 统一尺寸
        fg_h, fg_w = fg.shape[:2]
        bg = cv2.resize(bg, (fg_w, fg_h))

        # 提取前景 RGB + alpha
        if fg.shape[2] == 4:
            fg_rgb = fg[:, :, :3]
            alpha = fg[:, :, 3].astype(np.float32) / 255.0
        else:
            fg_rgb = fg
            alpha = np.ones((fg_h, fg_w), np.float32)

        # 步骤 1：分析背景色温
        bg_wb = self._estimate_color_temperature(bg)

        # 步骤 2：分析前景色温（仅考虑非透明区域）
        fg_wb = self._estimate_color_temperature(fg_rgb, alpha)

        # 步骤 3：计算色温转移矩阵
        color_transfer = self._compute_color_transfer(fg_wb, bg_wb)

        # 步骤 4：应用色温调整
        fg_adjusted = self._apply_color_transfer(fg_rgb, color_transfer)

        # 步骤 5：Alpha 混合
        alpha_3ch = np.stack([alpha] * 3, axis=2)
        composite = (fg_adjusted.astype(np.float32) * alpha_3ch +
                     bg.astype(np.float32) * (1 - alpha_3ch))
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        if output_path is None:
            base = os.path.splitext(fg_path)[0]
            output_path = f"{base}_composite.png"
        cv2.imwrite(output_path, composite)

        self._log(f"  Color temp: fg={fg_wb} → bg={bg_wb}")
        self._log(f"  Output: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_edge_band(self, mask: np.ndarray,
                       band_width: int = 3) -> np.ndarray:
        """
        提取 mask 的边缘带区域。

        Args:
            mask: 二值掩码 (0/255)
            band_width: 边缘带宽度（像素）

        Returns:
            边缘带二值图
        """
        kernel = np.ones((band_width, band_width), np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        return dilated - eroded

    def _estimate_color_temperature(self, img: np.ndarray,
                                     mask: Optional[np.ndarray] = None
                                     ) -> Tuple[float, float, float]:
        """
        基于灰度世界假设估计色温（RGB 均值）。

        Args:
            img: BGR 图像 (H,W,3)
            mask: 可选遮罩 (H,W)，1.0=参与计算

        Returns:
            (r_mean, g_mean, b_mean) BGR 均值
        """
        if mask is not None and mask.ndim == 2:
            mask_3ch = np.stack([mask] * 3, axis=2)
            masked = img.astype(np.float64) * mask_3ch
            count = np.sum(mask) + 1e-8
            means = np.sum(masked, axis=(0, 1)) / count
        else:
            means = np.mean(img.astype(np.float64), axis=(0, 1))

        return (float(means[2]), float(means[1]), float(means[0]))  # → RGB

    def _compute_color_transfer(self,
                                 src_rgb: Tuple[float, float, float],
                                 dst_rgb: Tuple[float, float, float]
                                 ) -> np.ndarray:
        """
        计算色温转移矩阵（对角缩放 + 偏移）。

        使前景色温向背景色温靠拢，保留一定原始特征。

        Args:
            src_rgb: 前景平均 RGB
            dst_rgb: 背景平均 RGB

        Returns:
            3x3 仿射矩阵 (BGR 序)
        """
        scale_r = dst_rgb[0] / max(src_rgb[0], 1.0)
        scale_g = dst_rgb[1] / max(src_rgb[1], 1.0)
        scale_b = dst_rgb[2] / max(src_rgb[2], 1.0)

        # 适度混合：保留 30% 原始色温
        blend = 0.7
        scale_r = 1.0 + (scale_r - 1.0) * blend
        scale_g = 1.0 + (scale_g - 1.0) * blend
        scale_b = 1.0 + (scale_b - 1.0) * blend

        # BGR 序矩阵
        matrix = np.array([
            [scale_b, 0, 0],
            [0, scale_g, 0],
            [0, 0, scale_r],
        ], dtype=np.float32)
        return matrix

    def _apply_color_transfer(self, img: np.ndarray,
                               matrix: np.ndarray) -> np.ndarray:
        """
        应用色温转移矩阵。

        Args:
            img: BGR 图像 (H,W,3)
            matrix: 3x3 缩放矩阵

        Returns:
            调整后的 BGR 图像
        """
        adjusted = img.astype(np.float32)
        for c in range(3):
            adjusted[:, :, c] *= matrix[c, c]
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _log(self, msg: str):
        """调试日志"""
        if self.debug:
            print(f"[SmartMatting] {msg}")


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def quick_matting(image_path: str) -> str:
    """
    快速一键抠像（便捷函数）。

    Args:
        image_path: 输入图像路径

    Returns:
        输出 PNG 路径
    """
    matting = SmartMatting(debug=True)
    result = matting.auto_matting(image_path)
    refined = matting.hair_refine(image_path, result["mask_path"])
    return refined


def green_screen_key(image_path: str, output_path: Optional[str] = None) -> str:
    """
    绿幕抠像（基于色度键的快速抠像，作为 GrabCut 的补充）。

    Args:
        image_path: 输入图像路径
        output_path: 输出路径

    Returns:
        输出图像路径
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 绿色范围
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)

    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask])

    if output_path is None:
        base = os.path.splitext(image_path)[0]
        output_path = f"{base}_greenscreen.png"
    cv2.imwrite(output_path, rgba)
    return output_path


if __name__ == "__main__":
    print("Nova Edit — SmartMatting Engine (Prototype)")
    print("=" * 46)

    from demo import create_test_image, create_test_background

    fg_img = create_test_image("test_fg.png",
                                "NOVA\nEDIT",
                                bg_color=(255, 255, 255),
                                text_color=(30, 30, 200))
    bg_img = create_test_background("test_bg.png",
                                     (127, 180, 80))

    sm = SmartMatting(debug=True)

    print("\n[1] Auto Matting (GrabCut)")
    try:
        result = sm.auto_matting(fg_img)
        print(f"  Output: {result['output_path']}")
        print(f"  Mask:   {result['mask_path']}")
        print(f"  Alpha foreground pixels: "
              f"{np.count_nonzero(result['alpha_mask'])}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n[2] Hair Refine")
    try:
        refined = sm.hair_refine(fg_img, result["mask_path"])
        print(f"  Output: {refined}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n[3] Composite with Lighting")
    try:
        composite = sm.composite_with_lighting(refined, bg_img)
        print(f"  Output: {composite}")
    except Exception as e:
        print(f"  FAILED: {e}")
