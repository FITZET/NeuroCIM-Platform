# Windows 发布说明

## 发布形式

项目采用 PyInstaller one-folder 方式生成 Windows x64 免安装包。发布包包含 PyTorch、CUDA、PySide6、NeuroSim、模型 checkpoint、数据集和器件测量 CSV，运行端无需安装 Python。

最终产物位于：

```text
release/NeuroCIM-Windows-x64/
release/NeuroCIM-Windows-x64.zip
release/SHA256SUMS.txt
```

解压 ZIP 后双击 `NeuroCIM/NeuroCIM.exe` 启动。不要只复制 EXE；程序需要同级 `_internal` 目录以及发布包根目录中的 `3_models`、`data` 和 `0_data`。

## 内置资源

完整包包含 LeNet-5、BNN、ResNet-20、TinyTransformer 和 GPT-2 Small 的 checkpoint；包含 MNIST、CIFAR-10 的完整训练集和测试集；包含五组器件乘法测量 CSV。首次启动即可执行误差拟合，随后选择网络进行推理或误差感知训练。

## 构建

在 `5_NeuroCIM_opt` 目录执行：

```powershell
python -m pip install -r packaging/requirements-build.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_release.ps1
```

构建脚本使用固定的 `release/NeuroCIM-Windows-x64` 目录，不创建按版本复制的源码目录。临时文件位于 `release/.build`，构建结束后删除。

## 验收

构建过程会检查：

1. 五个网络模型及其 checkpoint、tokenizer 和辅助参数；
2. MNIST 70,000 个样本和 CIFAR-10 60,000 个样本的文件结构；
3. 至少五份器件测量 CSV；
4. 冻结版 GUI 能够启动并生成截图；
5. ZIP64 归档完整性以及 EXE、ZIP 的 SHA-256。

NeuroSim 使用 CC BY-NC 4.0。第三方模型与数据集说明保存在 `THIRD_PARTY_NOTICES.md` 和 `licenses` 目录。
