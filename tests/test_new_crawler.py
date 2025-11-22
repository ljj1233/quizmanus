import pytest
from src.graph.crawler.crawler import Crawler

# 这里放入你想测试的国内 URL
# 1. CSDN 博客文章（技术类）
# 2. 知乎专栏文章（相对好爬）
# 3. 百度百科（最稳定的国内测试源）
TEST_URLS = [
    "https://zhuanlan.zhihu.com/p/1893871287308366230",                 # 知乎 Python 专栏
]

@pytest.mark.parametrize("url", TEST_URLS)
def test_crawler_returns_markdown_content(url: str):
    print(f"\n🚀 正在测试抓取: {url}")
    crawler = Crawler()
    
    try:
        article = crawler.crawl(url)
    except Exception as exc:
        # 如果网络不通（比如服务器在海外连不上国内，或者国内连不上特定站点），跳过
        pytest.skip(f"网络请求失败: {exc}")

    # 断言 1: 只要标题不是 "Error"，就说明请求通了
    # (Crawler 类里如果 fetch 失败会返回 title="Error")
    assert article.title != "Error", f"❌ 爬虫被拦截或失败! 错误信息: {article.content}"
    
    # 断言 2: 内容不能为空
    assert article.content, "❌ 抓取到的内容为空"
    
    # 断言 3: 内容长度要足够（避免只抓到 '403 Forbidden' 或验证码提示）
    # 中文网页通常包含大量元数据，Trafilatura 提取后一般都会超过 100 字
    assert len(article.content) > 50, f"❌ 内容太短 ({len(article.content)} chars)，可能被反爬拦截了"

    # 打印结果看看
    print(f"✅ 成功! 标题: {article.title}")
    print(f"📄 内容预览: {article.content[:100].replace(chr(10), ' ')}...") # 打印前100个字
    
    # 断言 4: URL 应该一致
    assert article.url == url