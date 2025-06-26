import numpy as np
import cv2
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

def create_audio_visualization(
    wav_path, 
    output_video_path,
    width=1920,
    height=1080,
    target_fps=30,
    freq_min=50,
    freq_max=600,
    perspective_start=0.2,
    perspective_end=1.0,
    height_scale=0.7,
    decay_factor=0.95,
    line_width=2,
    blur_size=(5,5),
    n_fft=2048,
    fixed_duration=True  # 新增：是否固定视频时长
):
    # 读取WAV文件
    rate, data = wavfile.read(wav_path)
    
    # 如果是立体声，转换为单声道
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # 计算音频时长
    duration = len(data) / rate
    
    # 设置视频帧率
    fps = target_fps
    
    # 计算期望的总帧数（用于固定时长）
    desired_frames = int(duration * fps)
    
    # 计算频谱图
    if fixed_duration:
        # 方法1：调整hop_length以匹配期望帧数
        hop_length = max(1, int(len(data) / desired_frames))
        win_length = min(n_fft, hop_length * 4)  # 确保窗口大小合理
    else:
        # 使用默认参数
        hop_length = n_fft // 4
        win_length = n_fft
    
    freqs, times, Sxx = spectrogram(
        data, 
        fs=rate,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        scaling='spectrum'
    )
    
    # 转换为分贝并归一化
    Sxx = 10 * np.log10(Sxx + 1e-12)
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-12)
    
    # 只保留50-600Hz的频率范围
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs = freqs[freq_mask]
    Sxx = Sxx[freq_mask, :]
    
    # 方法2：重采样到固定帧数
    if fixed_duration and len(times) != desired_frames:
        interp_func = interp1d(times, Sxx, axis=1, kind='linear', fill_value="extrapolate")
        new_times = np.linspace(times.min(), times.max(), desired_frames)
        Sxx = interp_func(new_times)
        times = new_times
        print(f"重采样频谱图: {len(times)}帧 → {desired_frames}帧")
    
    # 创建频率索引映射到屏幕宽度
    x_positions = ((freqs - freq_min) / (freq_max - freq_min) * width).astype(int)
    
    # 创建透视效果参数
    perspective = np.linspace(perspective_start, perspective_end, len(freqs))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 初始化帧缓冲区
    frame_buffer = np.zeros((height, width), dtype=np.float32)
    
    # 生成每一帧
    for i in tqdm(range(len(times)), desc="Generating frames"):
        # 向下滚动缓冲区
        frame_buffer = np.roll(frame_buffer, -1, axis=0)
        frame_buffer[-1, :] = 0
        
        # 获取当前频谱
        spectrum = Sxx[:, i]
        
        # 应用透视效果
        spectrum_persp = spectrum * perspective
        
        # 计算Y位置
        y_values = height - (spectrum_persp * height * height_scale).astype(int)
        y_values = np.clip(y_values, 0, height - 1)
        
        # 绘制频谱线
        for j in range(len(freqs) - 1):
            x1, x2 = x_positions[j], x_positions[j + 1]
            y1, y2 = y_values[j], y_values[j + 1]
            
            if 0 <= x1 < width and 0 <= x2 < width:
                cv2.line(frame_buffer, (x1, y1), (x2, y2), 1.0, line_width)
        
        # 应用衰减
        frame_buffer *= decay_factor
        
        # 转换为图像并模糊
        frame_img = np.clip(frame_buffer * 255, 0, 255).astype(np.uint8)
        frame_img = cv2.GaussianBlur(frame_img, blur_size, 0)
        frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
        
        # 写入视频帧
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"视频生成完成: 时长={len(times)/fps:.2f}秒, 帧数={len(times)}")
    return output_video_path

if __name__ == "__main__":
    input_wav = "input.wav"
    output_video = "output_final.mp4"
    
    # 使用固定时长模式
    create_audio_visualization(
        input_wav,
        output_video,
        fixed_duration=True,
        perspective_start=0.1,  # 远处的高低
        perspective_end=1.0,    # 近处的高低
        decay_factor=0.99,      # 衰减
        height_scale=1.2       # 振幅
    )