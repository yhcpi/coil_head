# Git 使用文档（针对本机 hyperyolo 项目）

> 写给以后的自己：怎么把这台机器的代码推到 GitHub。

---

## 0. 三个概念先理清

| 词 | 是什么 | 在哪里 |
|----|--------|--------|
| **本地仓库** | 你现在改代码的这个文件夹 | `/home/pi/projects/hyperyolo` |
| **远程仓库** | GitHub 上的同名仓库 | `github.com/yhcpi/coil_head` |
| **origin** | 给远程仓库起的别名（git 里默认这么叫） | 写在 `.git/config` 里 |

`git push` = 把本地的 commit 同步到 GitHub。

---

## 1. 第一次配新机器（已完成）

只做一次，**这台机器已经配好**。

### 1.1 生成 SSH 钥匙

```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_coil_head -N ""
```

会生成两个文件：
- `~/.ssh/github_coil_head` ← **私钥，永远不外传**
- `~/.ssh/github_coil_head.pub` ← **公钥，要上传给 GitHub**

### 1.2 公钥上传到 GitHub

1. 打开 https://github.com/settings/keys
2. **New SSH key** → Title 随便起（如 `coil_head-pi`）
3. Key 粘贴 `.pub` 文件的完整内容
4. 保存

### 1.3 测试连接

```bash
ssh -T git@github.com
# 应该看到: Hi yhcpi! You've successfully authenticated, ...
```

### 1.4 告诉 git 用哪把钥匙（本机有别的 key 时必须）

```bash
git config core.sshCommand "ssh -i ~/.ssh/github_coil_head -o StrictHostKeyChecking=accept-new"
```

这个设置写在 `.git/config` 里，**只对本仓库生效**。

### 1.5 让 git 知道远端在哪

```bash
git remote add origin git@github.com:yhcpi/coil_head.git
```

---

## 2. 日常 push 流程（3 步）

假设你改了文件，要推上去：

### 第 1 步：看哪些文件改了

```bash
git status
```

输出示例：
```
modified:   scripts/eval.py          ← 改过的
new file:   scripts/eval_v2.py       ← 新增的
```

### 第 2 步：选文件 + 打版本

```bash
# 选所有改过的文件
git add .

# 或者只选某些文件
git add scripts/eval.py scripts/eval_v2.py

# 打版本（commit）— -m 后面是这次改了什么
git commit -m "加了一个新评估脚本"
```

`commit` 是一次**快照**，可以理解成"打一个版本号"。

### 第 3 步：推到 GitHub

```bash
git push origin main
```

- `origin`：远端别名（1.5 配的那个）
- `main`：本地分支名

**以后所有 push 就这一行**。SSH 钥匙已经配好，不会再问密码。

---

## 3. 常用命令速查

| 想做什么 | 命令 |
|----------|------|
| 看仓库状态 | `git status` |
| 看最近 5 个 commit | `git log --oneline -5` |
| 看具体某次 commit 改了什么 | `git show <commit-sha>` |
| 看所有分支 | `git branch -a` |
| 撤销**没 add** 的改动 | `git checkout -- <file>` |
| 撤销**已 add** 但没 commit | `git restore --staged <file>` |
| 撤销**最后一次 commit**（保留改动） | `git reset --soft HEAD~1` |
| 撤销**最后一次 commit**（彻底丢弃） | `git reset --hard HEAD~1` ⚠️ |
| 看谁改过这一行 | `git blame <file>` |
| 拉取 GitHub 最新代码 | `git pull origin main` |
| 查看远端地址 | `git remote -v` |

---

## 4. 本仓库的 git 配置（已写好）

```bash
git config --local --list
```

| 配置项 | 值 | 作用 |
|--------|----|----|
| `core.sshCommand` | `ssh -i ~/.ssh/github_coil_head -o StrictHostKeyChecking=accept-new` | 让 git 用专门的 SSH 钥匙 |
| `remote.origin.url` | `git@github.com:yhcpi/coil_head.git` | 远端地址 |
| `user.name` | `pi`（全局） | commit 作者名 |
| `user.email` | `13787086860@163.com`（全局） | commit 作者邮箱 |

### 4.1 .gitignore 已经排除（不会上传）

```
runs/                  ← 训练产物（可重建）
*.pt *.onnx *.engine   ← 模型权重（用 GitHub Release 分发）
data/coil/images/      ← 数据集图片（太大，单独提供）
data/new_label/        ← labelme JSON 原始标注
repos/                 ← 第三方参考仓库（自己 clone）
.claude/ .omo/         ← 个人会话状态
*.bak *.tar.gz         ← 备份/临时
```

---

## 5. ⚠️ 踩过的坑（必看）

### 5.1 HTTPS 443 在这台机器上不稳

**症状**：
```
error: RPC 失败。curl 55 Recv failure: 连接被对方重置
```
或
```
curl: (28) Connection timed out
```

**原因**：到 github.com:443 的 HTTPS 路径被中间网络干扰（GFW/企业代理）。同台机器 SSH 22 端口是通的。

**解决**：**永远走 SSH**，不要走 HTTPS。即使一时能 push 大文件也别赌。

### 5.2 不要把 token 明文塞进 remote URL

**错的**：
```bash
git remote add origin https://用户名:TOKEN@github.com/yhcpi/coil_head.git
```

**为什么错**：token 会留在 `.git/config`、commit message、错误日志、screen share、聊天截屏……任何地方**全文明文泄露**。

**正确的**：用 SSH 钥匙（5.1 的方式），token 根本不需要存在。

### 5.3 已经在 conversation 里出现过的 PAT

`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` 在 2026-07-10 的对话里**完整明文出现过**——即使没主动用它，也必须 revoke：

1. https://github.com/settings/tokens
2. 找到这个 token → **Revoke**

后续如果要再生成新 PAT：
- 用 **fine-grained**（不是 classic）
- **Repository access** 选 `Only select repositories` → 勾 `yhcpi/coil_head`
- **Permissions** 只勾 `Contents: Read and write`
- 这样即使泄露，别人也只能动这一个仓库

### 5.4 git 默认走错 SSH 钥匙

**症状**：`Permission denied (publickey)`

**原因**：本机 `~/.ssh/` 下有别的钥匙（比如 `id_rsa`），git 默认走第一个匹配的，没认新生成的。

**解决**（已在 1.4 配好）：
```bash
git config core.sshCommand "ssh -i ~/.ssh/github_coil_head -o StrictHostKeyChecking=accept-new"
```

### 5.5 大文件推送慢/失败

**症状**：跑到一半 `connection reset`，但小文件没事。

**原因**：WSL2 + GitHub HTTP/2 大包传输不稳。

**解决**：
- 已经在 `.gitignore` 排除 `runs/` `*.pt` `data/coil/images/`
- 整个仓库控制在 ~200MB 以内
- 如果还不行，用 `git bundle create /tmp/x.bundle --all` 打包，让人手动拷过去

---

## 6. 第一次 push 一个新文件 — 完整示例

假设新建了 `scripts/test.py`：

```bash
cd /home/pi/projects/hyperyolo

# 1. 看状态
git status
#   ?? scripts/test.py     ← ?? = 未追踪

# 2. 加进去 + 打版本
git add scripts/test.py
git commit -m "Add test.py"

# 3. 推
git push origin main

# 输出:
# To github.com:yhcpi/coil_head.git
#    1c7b4cb..xxxxxxx  main -> main
```

完事。

---

## 7. 第一次 push 一个新仓库（备忘）

如果以后开新项目，第一次 push 流程：

```bash
cd /path/to/new-project
git init                                          # 初始化本地仓库
git add .
git commit -m "Initial commit"
git remote add origin git@github.com:yhcpi/<repo>.git
git push -u origin main                           # -u 记住 upstream
# 之后 push 直接 git push 即可
```

---

## 8. 紧急情况

### 8.1 误 commit 了不想上传的文件

```bash
git reset --soft HEAD~1           # 撤回到 add 状态（文件还在）
git restore --staged <file>       # 从 add 状态撤回（文件还在）
# 然后 git commit -m "新的 commit" 覆盖
```

### 8.2 已经 push 上去了，但想撤回

```bash
git revert <commit-sha>           # 新建一个 commit 反向操作，不删历史
# 或
git reset --hard <commit-sha>     # 直接回退（会改写历史，危险）
git push --force origin main      # 强制推送覆盖 GitHub（很危险，确认只有你一个人在用）
```

### 8.3 完全乱了，重新来

```bash
rm -rf .git                       # 删除本地仓库
git clone git@github.com:yhcpi/coil_head.git   # 从 GitHub 拉一份干净的
# 然后把改动文件复制回来，重新 add/commit/push
```

---

## 9. 参考

- https://docs.github.com/en/authentication/connecting-to-github-with-ssh
- https://git-scm.com/book/zh/v2