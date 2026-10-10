# BeyondContact 发布流程

本项目采用“组织私有仓库 + Anonymous GitHub 审稿链接”的方式。无需单独注册匿名 GitHub 账号。组织名称和真实远程地址只保留在本地 Git 配置中，不写入发布文件。

## 1. 创建组织私有仓库

使用有组织建仓权限的 GitHub 账号打开 https://github.com/new ：

- Owner：目标组织。
- Repository name：`BeyondContact`。
- Visibility：**Private**。
- 不勾选 README、.gitignore 或 License 初始化选项。
- 不使用 Fork 或从开发仓库导入历史。

本地已准备独立的 `main` 分支和匿名提交身份，未复制开发仓库的 Git 历史。

## 2. 上传独立副本

在 BeyondContact 目录检查已配置的远程地址；确认对应新建的组织私有仓库后推送：

```bash
git remote -v
git push -u origin main
```

使用拥有目标仓库写入权限的正常账号认证即可。匿名提交身份与 GitHub 登录账号是两回事；仓库本身保持 Private，审稿人不直接访问 GitHub 原始地址。

## 3. 生成匿名审稿链接

打开 https://anonymous.4open.science/ 并使用自己的 GitHub 账号连接。

1. 为目标私有仓库配置访问权限。如果当前服务提供 GitHub App 的逐仓库安装入口，选择 Only select repositories，仅选择 BeyondContact。以实际授权页面显示的权限为准。
2. 组织策略可能要求组织管理员批准应用访问。登录个人账号并不自动授权读取组织私有仓库。
3. 新建匿名项目，填写组织私有仓库的 GitHub 地址，选择发布分支（如界面提供）并使用 `BeyondContact` 作为匿名展示名称。
4. 将个人姓名、用户名、机构名、组织名、邮箱和开发仓库名称加入匿名替换词表；不要替换代码运行依赖的包名或第三方署名。
5. 完成匿名化后，使用无登录的浏览器窗口检查生成的页面和下载内容。
6. 论文和补充材料仅使用生成的匿名链接；原 GitHub 仓库继续保持 Private。

匿名服务会读取并缓存提交给它的内容，因此只连接这份已裁剪的发布仓库。匿名面向审稿人，GitHub 和服务运营方仍可知道来源；公开代码相似性也可能揭示来源。

## 4. 网页

`docs/` 中已准备静态网页。匿名阶段不要给审稿人发送包含真实组织名称的 GitHub Pages 地址。

如果匿名服务界面提供静态网页发布功能，配置入口为 `docs/index.html`，并检查 CSS、JavaScript 和页面链接均通过匿名地址加载。仓库浏览链接不等于网页已发布；需要实际验证。若服务不提供该功能，先通过匿名仓库分享网页源文件，正式公开后再启用 GitHub Pages。

## 5. 审稿结束后公开

在原组织仓库 Settings → Danger Zone → Change repository visibility 中将仓库改为 Public。恢复论文作者、机构及引用信息后，可通过 Settings → Pages → Deploy from a branch → main → /docs 发布网页。

无需搬迁仓库，本地副本仍与开发项目独立。

## 环境和资源

Python 包名保留为 `blower` 和 `blower_arm_rl`。请使用独立 Python/Conda 环境安装这份副本，避免 editable install 覆盖开发环境。具体第三方资产署名及机器人资产授权文档缺口见 THIRD_PARTY_NOTICES.md。

参考：
- https://github.com/tdurieux/anonymous_github
- https://docs.github.com/en/apps/using-github-apps/installing-a-github-app-from-a-third-party
- https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility
