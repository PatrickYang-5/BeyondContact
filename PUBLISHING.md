# BeyondContact 发布步骤

## 1. 创建匿名账号与空仓库

GitHub 仓库会公开显示所属账号；仅把仓库命名为 BeyondContact 不会隐藏原账号身份。请使用不包含姓名、机构或个人主页的独立审稿账号。公开代码、图片与已有公开项目相似，仍可能被关联；此副本不提供绝对匿名保证。

登录该账号，打开 https://github.com/new ：

- Owner：匿名账号；Repository name：`BeyondContact`。
- 给审稿人直接访问时选择 Public；也可先 Private 检查后再公开。
- 不勾选 README、.gitignore 或 license 初始化选项，本地已经有独立历史。
- 不使用 Fork 或原仓库 Import。

## 2. 推送本地副本

在这个 BeyondContact 目录运行，替换 ANONYMOUS_USER：

```bash
git remote add origin https://github.com/ANONYMOUS_USER/BeyondContact.git
git push -u origin main
```

使用匿名账号认证。如果凭据管理器自动使用旧账号，需要切换到匿名账号；不要把访问令牌写入 URL、命令文件或仓库。这个目录已设置本地提交身份 `Anonymous Authors <anonymous@example.invalid>`，不读取原项目的提交历史，也不修改全局 Git 身份。若希望关联匿名账号，可使用该账号 Settings → Emails 给出的 noreply 邮箱，在后续提交前设置本地 user.email。

## 3. 开启网页

仓库 Settings → Pages → Build and deployment：

- Source：Deploy from a branch。
- Branch：main；目录：/docs；Save。

发布后地址为 `https://ANONYMOUS_USER.github.io/BeyondContact/`。网页无原仓库链接，也不需要构建工具。

## 4. 本地隔离

当前目录是独立副本和独立 Git 仓库，没有与开发项目共享的链接或 remote。后续修改本目录不会修改开发项目。

Python 包名仍是 `blower` 和 `blower_arm_rl`；为避免 editable install 覆盖开发环境，请为发布副本使用独立 Python/Conda 环境。发布前请阅读 THIRD_PARTY_NOTICES.md 中具体的资产授权缺口。

参考：
- https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
- https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address
