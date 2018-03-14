# Style Transfer for Headshot Portraits 代码实现

参考论文：[《Style Transfer for Headshot Portraits》](http://people.csail.mit.edu/yichangshih/portrait_web/)

基本步骤转 python 代码，SIFTflow 部分仍使用 matlab 代码。

## 使用

1. 下载 SIFTflow 代码，根据操作系统编译好后，将 `SIFTflow` 文件夹放到根目录的 `libs/` 目录下，与 `image_pyramids` 同级。
2. 从论文网站上下载数据集，解压到任意目录，修改 `config.py` 中的 `root` 为 `data` 的路径。
3. 配置输出目录，修改 `config.py` 中的 `output_folder`。
4. 修改 `config.py` 中的 `style_in`、`im_in_name`、`style_ex`、`im_ex_name`，选择图片。
5. 执行 `python first.py`
6. matlab 中运行 `sift_mask`
7. 执行 `python second.py`

整体跑的时间会比较久，一张图大概需要几分钟。
