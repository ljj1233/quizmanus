 
### 📋 任务指令书：爬虫引擎迁移 (Jina -\> Trafilatura)

**执行角色**：高级 Python 工程师
**任务目标**：将项目中的爬虫模块从依赖 Jina 服务迁移为本地 Trafilatura 库，并彻底清除 Jina 相关代码与配置。
**核心原则**：保持接口兼容性（Interface Compatibility），确保上层调用感知不到底层变更。

#### 📅 第一阶段：依赖管理 (Dependency)

1.  **添加新依赖**：
      - 在 `requirements.txt` 中添加 `trafilatura>=1.6.0`。
2.  **安装依赖**：
      - 执行 `pip install trafilatura` 确保环境就绪。

#### 🛠 第二阶段：核心代码重构 (Refactor)

**目标文件**：`src/graph/crawler/crawler.py`
**操作要求**：

1.  **重写 `Crawler` 类**：
      - 移除对 `jina_client` 和 `readability_extractor` 的导入。
      - 引入 `trafilatura`。
      - **关键逻辑**：使用 `trafilatura.fetch_url(url)` 获取内容，使用 `trafilatura.extract(..., output_format="markdown")` 提取正文。
2.  **保持接口一致**：
      - `crawl(self, url: str) -> Article` 的方法签名**不能变**。
      - 返回值必须是 `src/graph/crawler/article.py` 中定义的 `Article` 对象。
      - 需手动填充 `Article` 的 `title`, `url`, `content` 字段（从 Trafilatura 的 metadata 和 result 中获取）。

**代码参考逻辑**：

```python
# 伪代码提示
downloaded = trafilatura.fetch_url(url)
content = trafilatura.extract(downloaded, output_format="markdown", include_tables=True)
metadata = trafilatura.extract_metadata(downloaded)
# 组装 Article 对象返回...
```

#### 🗑 第三阶段：代码清理 (Cleanup)

**删除以下不再需要的文件**：

1.  `src/graph/crawler/jina_client.py`
2.  `src/graph/crawler/readability_extractor.py`

#### ⚙️ 第四阶段：配置清洗 (Configuration)

1.  **环境配置**：
      - 检查 `.env` 和 `.env-example`，**移除** `JINA_API_KEY` 字段。
2.  **代码配置**：
      - 全局搜索 `JINA` 关键字，确保没有残留的引用（例如在 `src/config/tools.py` 或 `src/graph/tools/crawler.py` 的注释中）。

#### ✅ 第五阶段：验证 (Verification)

1.  **创建测试脚本** `tests/test_new_crawler.py`：
      - 抓取一个简单的维基百科页面（如 `https://zh.wikipedia.org/wiki/Python`）。
      - 断言返回的 `Article` 对象 `content` 不为空，且长度大于 100 字符。
2.  **运行测试**：确保重构未破坏现有功能。

 