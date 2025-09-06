<img width="1034" height="561" alt="image" src="https://github.com/user-attachments/assets/3d7c6465-96c0-4132-9292-9e5cc450f4d1" />
目录扫描：personal_file_index_build.py 支持自定义 root_list，只索引你关心的文件夹（避免全盘扫描）。

文本抽取：解析 PDF / DOCX 文本，自动切分成小片段。

向量索引：使用 BAAI/bge-m3 或 bge-zh-v1.5 系列模型，将文本转为语义向量，存储在 FAISS 数据库中。

增量更新：build_add.py 支持新增文件的索引构建，而无需全量重建。

文件检索：

find_path.py：在检索基础上调用大模型 API（如硅基流动），也可本地部署7B版本大模型，取决于显卡性能，生成更自然的答案与文件定位结果。也可以纯本地语义检索，返回最相似的文件路径与片段（实际测试效果不好）；

