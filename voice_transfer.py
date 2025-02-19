import faster_whisper
import subprocess
import os
import re
from tqdm import tqdm
from modelscope import snapshot_download
class WhisperTranscriber:
    @staticmethod
    def load_whisper_model():
        try:
            return faster_whisper.WhisperModel(model_size_or_path = './whisper_model/angelala00/faster-whisper-small', device="auto")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    @staticmethod
    def transcribe_audio_to_text(audio_path, model):
        if not os.path.exists(audio_path):
            print(f"Audio file does not exist: {audio_path}")
            return None
        
        segments, info = model.transcribe(audio_path)
        total_duration = getattr(info, 'duration', None)
        
        if total_duration is None:
            print("Failed to retrieve audio duration.")
        
        with tqdm(total=total_duration, desc='Transcribing', unit='s',
                 bar_format='{l_bar}{bar}| {n:.1f}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            segments_list = []
            for segment in segments:
                segments_list.append(segment)
                if total_duration is not None:
                    pbar.update(segment.end - (segments_list[-2].end if len(segments_list) > 1 else 0))
            
            result_text = " ".join(segment.text for segment in segments_list)
            return {
                "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments_list],
                "text": result_text
            }

    @staticmethod
    def save_transcript_to_txt(transcript, file_path):
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(transcript)
        except IOError as e:
            print(f"Error writing transcript to file: {str(e)}")

    @staticmethod
    def format_transcript_with_timestamps_and_numbers(transcript_result):
        formatted = ""
        for idx, segment in enumerate(transcript_result['segments'], 1):
            formatted += f"{idx}. [{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}\n\n"
        return formatted

    @staticmethod
    def get_video_duration(video_path):
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting duration: {str(e)}")
            return None

    @staticmethod
    def extract_audio_from_video(video_path, audio_output_path):
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            return
        
        duration = WhisperTranscriber.get_video_duration(video_path)
        if duration is None:
            print("Could not determine video duration.")
            return
        
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_output_path]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        time_pattern = re.compile(r'time=(\d+:\d+:\d+\.\d+)')
        
        with tqdm(total=duration, desc='Extracting', unit='s',
                 bar_format='{l_bar}{bar}| {n:.1f}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            current_time = 0
            for line in process.stdout:
                print(line.strip())  # 调试：打印每一行输出
                match = time_pattern.search(line)
                if match:
                    time_str = match.group(1)
                    parts = list(map(float, time_str.split(':')))
                    new_time = parts[0]*3600 + parts[1]*60 + parts[2]
                    if duration and new_time <= duration:
                        update_amount = new_time - current_time
                        if update_amount > 0:
                            pbar.update(update_amount)
                            current_time = new_time
            process.wait()
            if process.returncode != 0:
                print(f"FFmpeg returned non-zero exit status: {process.returncode}")

def main():
    os.makedirs("audio", exist_ok=True)
    os.makedirs("transcript", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    video_path = os.path.join("videos", "video.mp4")
    audio_path = os.path.join("audio", "audio.wav")
    transcript_path = os.path.join("transcript", "transcript.txt")

    WhisperTranscriber.extract_audio_from_video(video_path, audio_path)
    
    model = WhisperTranscriber.load_whisper_model()
    if model is None:
        print("Exiting due to model loading failure.")
        return
    
    result = WhisperTranscriber.transcribe_audio_to_text(audio_path, model)
    if result is None:
        print("Exiting due to transcription failure.")
        return
    
    formatted = WhisperTranscriber.format_transcript_with_timestamps_and_numbers(result)
    WhisperTranscriber.save_transcript_to_txt(formatted, transcript_path)
    
    print(f"Transcript saved to {transcript_path}")

if __name__ == '__main__':
    # model_dir = snapshot_download(model_id = 'angelala00/faster-whisper-small',cache_dir= './whisper_model')
    main()