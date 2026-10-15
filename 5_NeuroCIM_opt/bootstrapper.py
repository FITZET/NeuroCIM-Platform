from __future__ import annotations

import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

from neurocim_opt.release_config import ACCESS_PORTAL_URL
from neurocim_opt.updater import (
    CredentialStore,
    GitHubReleaseClient,
    UpdateError,
    installed_flavor,
    launch_installer,
)


class InstallerWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NeuroCIM 安装引导程序")
        self.geometry("660x420")
        self.minsize(620, 390)
        self.configure(background="#F4F7FB")
        self.client = GitHubReleaseClient()
        self.busy = False

        panel = ttk.Frame(self, padding=34)
        panel.pack(fill="both", expand=True, padx=24, pady=24)
        ttk.Label(panel, text="NeuroCIM", font=("Microsoft YaHei UI", 27, "bold")).pack(anchor="w")
        ttk.Label(
            panel,
            text="使用已批准的 GitHub 账号登录，程序将自动检测 CPU/NVIDIA GPU 并下载安装。",
            wraplength=560,
            font=("Microsoft YaHei UI", 11),
        ).pack(anchor="w", pady=(10, 20))
        self.hardware = ttk.Label(panel, text="正在检测硬件……")
        self.hardware.pack(anchor="w")
        self.status = ttk.Label(panel, text="准备就绪", wraplength=560)
        self.status.pack(anchor="w", pady=(12, 8))
        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(4, 18))
        row = ttk.Frame(panel)
        row.pack(fill="x")
        self.install_button = ttk.Button(row, text="GitHub登录并安装", command=self.start)
        self.install_button.pack(side="left")
        ttk.Button(row, text="申请访问权限", command=lambda: webbrowser.open(ACCESS_PORTAL_URL)).pack(
            side="left", padx=12
        )
        ttk.Button(row, text="退出", command=self.destroy).pack(side="right")
        self.after(50, self._show_hardware)

    def _show_hardware(self) -> None:
        flavor = installed_flavor(Path.cwd()).upper()
        self.hardware.configure(text=f"自动选择版本：{flavor}")

    def ui(self, callback, *args, **kwargs) -> None:
        self.after(0, lambda: callback(*args, **kwargs))

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def start(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.install_button.configure(state="disabled")
        self.progress.configure(value=0)
        threading.Thread(target=self._install, daemon=True).start()

    def _install(self) -> None:
        try:
            if not self.client.token:
                authorization = self.client.begin_device_login()
                self.ui(
                    self.set_status,
                    f"浏览器中输入代码 {authorization.user_code} 完成 GitHub 登录。",
                )
                webbrowser.open(authorization.verification_uri)
                self.client.finish_device_login(authorization)
            flavor = installed_flavor(Path.cwd())
            self.ui(self.set_status, "正在检查可用安装版本……")
            update = self.client.latest_update(flavor=flavor, current="0.0.0")
            if update is None:
                raise UpdateError("私有发布仓库中没有可安装版本")
            local = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NeuroCIM" / "installer" / update.version

            def report(name: str, done: int, total: int) -> None:
                percent = round(done * 100 / total) if total else 0
                self.ui(self.set_status, f"正在下载 {name}：{percent}%")
                self.ui(self.progress.configure, value=percent)

            installer = self.client.download_update_package(update, local, progress=report)
            self.ui(self.set_status, "校验完成，正在启动安装程序……")
            launch_installer(installer)
            self.ui(self.after, 600, self.destroy)
        except UpdateError as exc:
            self.ui(self._failed, str(exc))
        except Exception as exc:
            self.ui(self._failed, f"安装失败：{exc}")

    def _failed(self, message: str) -> None:
        self.busy = False
        self.install_button.configure(state="normal")
        self.set_status(message)
        if "尚未获准" in message:
            messagebox.showinfo("需要审批", message + "\n\n请先在申请网页提交注册申请。")
            webbrowser.open(ACCESS_PORTAL_URL)
        else:
            messagebox.showerror("NeuroCIM安装", message)


if __name__ == "__main__":
    InstallerWindow().mainloop()
