import webview
import flask
import threading
from edit_video import clean_same_frames
from video_to_images import export_images
from voice_transfer import WhisperTranscriber
from clip_part import load_texts, export_md_with_keywords_and_ocr
from RAG_part import DocumentAnalyzer
import os
from flask import Flask, send_from_directory,jsonify, request
import threading
from configs import config
from modelscope import snapshot_download
import shutil
app_flask = Flask(__name__, static_folder='./UI_design')
shutdown_event = threading.Event()

def clear_files(directory):
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return
    
    if not os.path.isdir(directory):
        print(f"{directory} 不是一个有效的目录")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除 {file_path} 失败。原因: {e}")

def initialize_directories(file_paths):
    for path in file_paths.values():
        if os.path.isdir(path):  # 如果路径指向目录，则清空该目录
            clear_files(path)
        elif os.path.isfile(path):  # 如果路径指向文件，则直接删除该文件
            try:
                os.unlink(path)
                print(f"已删除文件: {path}")
            except Exception as e:
                print(f"删除 {path} 失败。原因: {e}")

@app_flask.route('/')
def serve_index():
    return app_flask.send_static_file("index.html")

# 服务于静态文件的通用路由
@app_flask.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app_flask.static_folder, path)

# 添加一个新路由以服务于config.json文件
@app_flask.route('/configs/config.json')
def serve_config():
    return send_from_directory('./configs', 'config.json')

@app_flask.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_event.set()
    func = flask.request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'
@app_flask.route('/api/create-folder', methods=['POST'])
def create_folder():
    data = request.get_json()
    folder_name = data.get('folderName')
    folder_path = './final_output'
    
    if not folder_name:
        return jsonify({"success": False, "message": "未提供文件夹名称"}), 400
    
    full_path = os.path.join(folder_path, folder_name) 
    
    try:
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return jsonify({"success": True, "message": f"文件夹 {folder_name} 在 {folder_path} 创建成功"}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
@app_flask.route('/api/save-md', methods=['POST'])
def save_md():
    data = request.get_json()
    md_content = data.get('mdContent')
    folder_name = data.get('folderName')
    file_name = data.get('fileName')
    
    if not md_content or not folder_name or not file_name:
        return jsonify({"success": False, "message": "未提供必要的参数"}), 400

    # 使用os.path.join来构建完整的文件路径
    folder_path = os.path.join('./final_output', folder_name)
    file_path = os.path.join(folder_path, f"{file_name}.md")

    # 确保文件夹存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    try:
        with open(file_path, 'w',encoding = 'utf-8') as file:
            file.write(md_content)
        return jsonify({"success": True, "message": f"MD 文件已保存到 {file_path}"}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
@app_flask.route('/api/list-folders', methods=['GET'])
def list_folders():
    try:
        folder_path = './final_output'
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return jsonify({"folders": [], "files": {}}), 404

        folders = []
        files = {}  # 文件夹对应的 MD 文件字典

        items = os.listdir(folder_path)
        for item in items:
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                folders.append(item)
                md_files = []  # 保存当前文件夹中的 .md 文件
                for filename in os.listdir(item_path):
                    if filename.endswith(".md"):
                        md_files.append(filename)
                files[item] = md_files  # 保存文件夹中的 .md 文件

        return jsonify({
            "folders": folders,
            "files": files
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500
@app_flask.route('/api/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # 确保目标目录存在
        output_dir = './videos'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存文件并重命名为video.mp4
        file_path = os.path.join(output_dir, 'video.mp4')
        file.save(file_path)
        return jsonify({'message': 'Video uploaded and renamed successfully.'}), 200
    
@app_flask.route('/api/delete-folder', methods=['POST'])
def delete_folder():
    folder_path = os.path.join('./final_output', request.json.get('folderName'))
    try:
        shutil.rmtree(folder_path)  # 删除文件夹
        return jsonify({'success': True, 'message': 'Folder deleted successfully.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
@app_flask.route('/api/delete-file', methods=['POST'])
def delete_file():
    data = request.get_json()
    folder_name = data.get('folderName')
    file_name = data.get('fileName')

    if not folder_name or not file_name:
        return jsonify({"success": False, "message": "未提供文件夹名称或文件名称"}), 400

    file_path = os.path.join('./final_output', folder_name, file_name)

    if not os.path.exists(file_path):
        return jsonify({"success": False, "message": f"文件 {file_name} 不存在"}), 404

    try:
        os.remove(file_path)
        return jsonify({"success": True, "message": f"文件 {file_name} 删除成功"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
@app_flask.route('/static/filter_images/<path:filename>')
def serve_filter_images(filename):
    return send_from_directory('./filter_images', filename)

def run_flask_app():
    app_flask.run(host='0.0.0.0', port=5000, threaded=True)
class App():
    def __init__(self):
        return
    #去除视频重复帧
    def remove_same_frames(self, video_path, output_path):
        return clean_same_frames(video_path, output_path)
    #导出图片和AI总结
    def export_images(self):
        #不需要传递任何参数
        return export_images()      
    def generate_transcript(self, video_path, audio_path, transcript_path):
        WhisperTranscriber.extract_audio_from_video(video_path, audio_path)
        # 加载模型
        model = WhisperTranscriber.load_whisper_model()
        # 转录音频为文本
        result = WhisperTranscriber.transcribe_audio_to_text(audio_path, model)
        # 格式化转录文本，添加时间戳和序号
        formatted_transcript = WhisperTranscriber.format_transcript_with_timestamps_and_numbers(result)
        # 保存文本
        WhisperTranscriber.save_transcript_to_txt(formatted_transcript, transcript_path)
    def load_texts(self, file_path):
        return load_texts(file_path)
    def export_main_md(self,image_path, texts, output_path):
        return export_md_with_keywords_and_ocr(image_path, texts, output_path)
    #RAG部分，然后db_name是数据库名，prompt是问题
    def RAG_part(self,prompt):
        # 如果./transcripts文件夹存在，则使用transcript文件
        if os.path.exists('./transcript'):
            documents_path = './transcript'
        else:
            documents_path = './final_output'
        RAG = DocumentAnalyzer(documents_path= documents_path,
                                db_path= './my_db',
                                db_name='my_database',
                                similarity_threshold=0.3,
                                chunk_count=5)
        RAG.create_and_save_index()
        response = RAG.get_model_response(prompt)
        return response
        
    def read_md_files(self,directory):
        md_contents = {}

        # 检查是否是目录
        if os.path.isdir(directory):
            # 遍历目录中的所有文件
            for filename in os.listdir(directory):
                if filename.endswith(".md"):
                    file_path = os.path.join(directory, filename)
                    with open(file_path, "r", encoding='utf-8') as file:
                        content = file.read()
                        content = content.replace('../filter_images/', '/static/filter_images/')
                        md_contents[filename] = content
            return md_contents

        # 检查是否是文件
        elif os.path.isfile(directory):
            # 检查文件扩展名是否为 .md
            if directory.endswith(".md"):
                with open(directory, "r", encoding='utf-8') as file:
                    content = file.read()
                    content = content.replace('../filter_images/', '/static/filter_images/')
                    md_contents[os.path.basename(directory)] = content
                return md_contents
            else:
                return {"error": "Invalid file type. Only .md files are supported."}

        else:
            return {"error": "Invalid path. Please provide a valid directory or .md file path."}
    def initialize(self,file_paths):
       return initialize_directories(file_paths)
    
    def get_md_files(self, folder_path):
        md_files = []
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".md"):
                    md_files.append(filename)
        return md_files

                     
def main():
    files_paths = config['file_path']
    initialize_directories(files_paths)
    print("初始化完成")
    api = App()
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Create and start the WebView window
    webview.create_window(
        "Video2MD",
        url="http://localhost:5000/",
        min_size=(800, 600),
        resizable=True,
        js_api=api
    )
    webview.start(debug=True)

if __name__ == '__main__':
    # 如果模型不存在，则下载
    if not os.path.exists('./whisper_model'):
        os.makedirs('./whisper_model')
        model_dir = snapshot_download(model_id = 'angelala00/faster-whisper-small',cache_dir= './whisper_model')
    if not os.path.exists('./embedding_model'):
        os.makedirs('./embedding_model')
        model_dir = snapshot_download(model_id = 'Jerry0/m3e-base',cache_dir= './embedding_model')
    main()
