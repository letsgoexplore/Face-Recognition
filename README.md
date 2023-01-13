# Face-Recognition
### 配置环境
先确认已经安装了conda，如果没有的话先pip install anaconda。而后创建conda虚拟环境并且启动，遇到选项[y/n]选择y。
```
conda create facerecog python=3.8
conda activate facerecog
```
然后下载依赖包
```
pip install -r requirment.txt
```
配置到此为止

### 数据预处理
```
python pre_processing.py
```

### 用training_set训练网络
```
python main.py
```

### 在test_pair上测试数据集
```
python predict.py
```
结果在text_result开头的txt中
