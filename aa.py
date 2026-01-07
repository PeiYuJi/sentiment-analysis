import requests
from bs4 import BeautifulSoup
import jieba
from snownlp import SnowNLP
import pandas as pd
import re

# ======================
# 1️⃣ 目標網址
# ======================
url = "https://www.ptt.cc/bbs/Stock/M.1767791864.A.C8D.html"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Cookie": "over18=1"
}

# ======================
# 2️⃣ 爬取文章
# ======================
res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")

# ======================
# 3️⃣ 文章主內容
# ======================
main_content = soup.find(id="main-content")
text = main_content.text

# 移除 metadata
text = re.sub(r"※ 發信站:.*", "", text)
text = re.sub(r"--\n.*", "", text, flags=re.S)

print("📄 文章內容擷取完成")

# ======================
# 4️⃣ 推文擷取
# ======================
pushes = soup.find_all("div", class_="push")
push_texts = []

for p in pushes:
    tag = p.find("span", class_="push-tag").text.strip()
    content = p.find("span", class_="push-content").text.strip(": ")
    push_texts.append(f"{tag} {content}")

print(f"💬 推文數量：{len(push_texts)}")

# ======================
# 5️⃣ 合併所有文字
# ======================
all_text = text + " ".join(push_texts)

# ======================
# 6️⃣ 中文斷詞
# ======================
words = jieba.lcut(all_text)

# 移除過短詞
words = [w for w in words if len(w) > 1]

# ======================
# 7️⃣ 情緒分析
# ======================
sentiments = []
for sentence in push_texts:
    s = SnowNLP(sentence)
    sentiments.append(s.sentiments)

sentiment_df = pd.DataFrame({
    "comment": push_texts,
    "sentiment": sentiments
})

# 分類
def sentiment_label(score):
    if score > 0.6:
        return "正向"
    elif score < 0.4:
        return "負向"
    else:
        return "中立"

sentiment_df["label"] = sentiment_df["sentiment"].apply(sentiment_label)

# ======================
# 8️⃣ 統計結果
# ======================
summary = sentiment_df["label"].value_counts()

print("\n📊 輿情分析結果")
print(summary)

# ======================
# 9️⃣ 顯示前幾筆
# ======================
print("\n🔍 推文情緒樣本")
print(sentiment_df.head(10))


