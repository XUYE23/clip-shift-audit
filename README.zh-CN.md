# CLIP Shift Audit

用于 CLIP 下游分类器的开放集评估与拒识校准。核心评估只依赖 NumPy。

安装：python -m pip install .
演示：clip-shift-audit demo --output outputs/demo.json
测试：python -m unittest discover -s tests -v

打开 outputs/demo.html 查看报告。演示数据是固定随机种子的合成特征，
结果仅用于验证评估流程。真实图像结果需要运行可选 OpenCLIP 提取脚本。

输入包含类别文本特征、已知类校准特征、测试特征及标签。
输出包括 AUROC、FPR@95 TPR、已知类接受率、未知类接受率、选择性准确率。
原始测试数据与校准数据必须隔离，类别词表与温度须在校准前固定。

适合：CLIP 迁移、开放世界识别、模型部署前的拒识评估、增量类别变化研究。
详见英文 README 与 docs/protocol.md。
