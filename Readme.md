# 项目介绍 #
!["logo"](./logo.png)
项目名称叫做Video2MD,主要是用来将网课类的视频转化为可管理的文字，输出的文章以markdown的格式保存。

# 基本使用 #
## 首先 git clone 项目
'https://github.com/yoruniubi/Video2MD.git'
## 然后配置api_key
然后将自己的api_key填入**config.json**文件中

建议使用qwen的api_key

## 建议使用anaconda来建立虚拟环境  
'conda create -n video2md python=3.10.8'
## 安装依赖项
'pip install -r requirements.txt'
## 运行程序
'python app_ui.py'
# 温馨提示 #
如果感觉效果不好，可以替换更好的模型

修改**client.chat.completions.create**的

**model** 参数 

