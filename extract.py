from moviepy.editor import *
import os
import argparse
import cv2
import time
from tqdm import tqdm
import face_recognition
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract(dataset):
    input_dir_path = f'data/{dataset}/Raw'
    output_wav_dir_path = f'data/{dataset}/wav'
    output_img_dir_path = f'data/{dataset}/img'

    if not os.path.exists(output_wav_dir_path):
        os.makedirs(output_wav_dir_path)

    if not os.path.exists(output_img_dir_path):
        os.makedirs(output_img_dir_path)

    for folder in tqdm(os.listdir(input_dir_path)):
        input_folder = os.path.join(input_dir_path, folder)

        output_wav_folder = os.path.join(output_wav_dir_path, folder)
        if not os.path.exists(output_wav_folder):
            os.makedirs(output_wav_folder)

        output_img_folder = os.path.join(output_img_dir_path, folder)
        if not os.path.exists(output_img_folder):
            os.makedirs(output_img_folder)

        for filename in os.listdir(input_folder):
            if filename.split(".")[-1] != "mp4":
                continue

            input_file = os.path.join(input_folder, filename)
            output_wav_file = os.path.join(output_wav_folder, filename)
            output_img_file = os.path.join(output_img_folder, filename)

           

            video = VideoFileClip(input_file)

            extract_audio(video, output_wav_file)
            extract_image(video, output_img_file)


def extract_audio(video, output_wav_file):
    audio = video.audio
    # Set the desired sampling rate
    desired_sampling_rate = 16000  # Replace this value with your desired sampling rate
    # Resample the audio to the desired sampling rate
    resampled_audio = audio.set_fps(desired_sampling_rate)
    # Save the extracted and resampled audio to a WAV file
    output_wav_file = output_wav_file.replace(".mp4", ".wav")
    resampled_audio.write_audiofile(output_wav_file, codec='pcm_s16le', verbose=False, logger=None)


def extract_image(video, output_img_file, interval=1, threshold=0.5):
    output_img_file = output_img_file[:-4]  # 去掉文件扩展名
    if not os.path.exists(output_img_file):
        os.makedirs(output_img_file)

    faces_list = []  # 存储所有人脸编码
    video_duration = video.duration

    if video_duration < 1:
        frame = video.get_frame(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]

            # 裁剪出人脸区域
            face_image = frame[top:bottom, left:right]

            # 保存人脸区域图像
            save_path = os.path.join(output_img_file, f"{0}.jpg")
            cv2.imwrite(save_path, face_image)
    else:
        for index, t in enumerate(range(0, int(video_duration), interval)):
            # 获取视频帧
            frame = video.get_frame(t)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间

            # 提取人脸编码
            face_encoding = get_face(frame)

            # 初始化 is_similar 变量
            is_similar = False  # 赋默认值为 False，表示当前人脸默认不与已有的人脸相似

            if face_encoding is not None:
                # 对比当前人脸与列表中所有已有的人脸编码
                for stored_encoding in faces_list:
                    if stored_encoding is not None:  # 确保 stored_encoding 不是 None
                        similarity = calculate_similarity(face_encoding, stored_encoding)
                        if similarity < threshold:  # 如果与任何一个已有的人脸相似度低于阈值
                            is_similar = True
                            break

            # 只有当所有相似度都高于阈值时，才将当前人脸编码添加到列表
            if not is_similar:
                faces_list.append(face_encoding)

                # 获取人脸区域的位置
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]

                    # 裁剪出人脸区域
                    face_image = frame[top:bottom, left:right]

                    # 保存人脸区域图像
                    save_path = os.path.join(output_img_file, f"{index}.jpg")
                    cv2.imwrite(save_path, face_image)





def get_face(frame):
    # 获取人脸位置
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        # 提取第一个检测到的人脸编码
        face_encoding = face_recognition.face_encodings(frame, face_locations)
        return face_encoding[0]  # 返回人脸编码
    return None

def calculate_similarity(face_encoding1, face_encoding2):
    if face_encoding1 is None or face_encoding2 is None:
        return float('inf')  # 如果有一个是 None，返回无穷大（表示非常不相似）
    
    distance = np.linalg.norm(np.array(face_encoding1) - np.array(face_encoding2))
    return distance


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MOSI', help='dataset name')
    args = parser.parse_args()
    extract(args.dataset)
