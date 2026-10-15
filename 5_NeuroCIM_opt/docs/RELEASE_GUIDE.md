# Windows 发布说明

## 用户下载与授权

用户在 [NeuroCIM Access](https://fitzet.github.io/NeuroCIM-Access/) 使用 GitHub 账号提交申请。管理员为 Issue 添加 `approved` 标签后，自动化流程会向该账号发送私有 `NeuroCIM-Distribution` 仓库的只读邀请。用户接受邀请并运行公开的 `NeuroCIM-Installer.exe` 后，引导程序完成 GitHub 设备登录、NVIDIA GPU 检测、版本选择、分卷下载、SHA-256 校验和安装。

程序按当前用户安装到 `%LOCALAPPDATA%\Programs\NeuroCIM`，不要求管理员权限。运行输出默认写入 `%LOCALAPPDATA%\NeuroCIM\outputs`。

## 发布内容

每个正式版本在私有分发仓库中同时提供：

- CPU 通用版安装包；
- NVIDIA GPU 版安装包；
- `update-manifest.json`，记录各分卷的名称、大小和 SHA-256；
- LeNet-5、BNN、ResNet-20、TinyTransformer 和 GPT-2 的 checkpoint；
- MNIST、CIFAR-10、五组器件 CSV 和 NeuroSim 运行文件。

Inno Setup 把大型安装包切分为一个 `.exe` 和若干 `.bin`，单个文件不超过 GitHub Release 的资产限制。引导程序和软件内更新器会下载同一版本的全部文件，逐个验证后再运行安装器。

## 自动发布

在源代码仓库推送形如 `v1.0.0` 的标签后，`.github/workflows/windows-release.yml` 会：

1. 分别安装 CPU 与 CUDA 版 PyTorch；
2. 从私有分发仓库取得经过认证的运行资源；
3. 执行测试、冻结检查和资源完整性检查；
4. 构建 CPU/GPU 两套当前用户安装包；
5. 合并更新清单并发布到私有分发仓库；
6. 构建公开引导程序并发布到访问门户仓库。

源代码仓库需要以下 Actions secrets：

- `NEUROCIM_GITHUB_CLIENT_ID`：启用 Device Flow 的 GitHub OAuth App 客户端 ID；
- `DISTRIBUTION_TOKEN`：仅允许写入 `NeuroCIM-Distribution` Releases/Contents 的细粒度令牌；
- `ACCESS_RELEASE_TOKEN`：仅允许写入 `NeuroCIM-Access` Releases/Contents 的细粒度令牌。

访问门户仓库还需要 `DISTRIBUTION_ADMIN_TOKEN`，其权限仅限管理 `NeuroCIM-Distribution` 的协作者。

## 本地构建

在 `5_NeuroCIM_opt` 目录执行：

```powershell
./scripts/build_installer.ps1 -Flavor GPU -Version 1.0.0 -ClientId <OAuth客户端ID>
./scripts/build_bootstrapper.ps1 -ClientId <OAuth客户端ID>
```

构建脚本会临时把公开 OAuth Client ID 写入冻结程序，并在成功或失败后恢复源文件，因此不会把本地测试配置遗留在代码树中。

NeuroSim 使用 CC BY-NC 4.0。第三方模型和数据集说明保存在 `THIRD_PARTY_NOTICES.md` 与 `licenses` 目录。
