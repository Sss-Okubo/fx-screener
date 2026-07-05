# FX通貨ペアスクリーナー

主要18通貨ペア(円クロス7・ドルストレート6・その他クロス5)を毎週スコアリングし、
「どのペアの買い(ロング)が魅力的か」をランキングするツール。
[stock-screener](https://github.com/Sss-Okubo/stock-screener) のFX版。

## スコアリング (キャリー・割安重視)

各指標を全ペア内で偏差値化し、重み付きで合算する:

| カテゴリ | 重み | 指標 |
|---|---:|---|
| 金利差 (キャリー) | 30% | 基軸通貨と決済通貨の政策金利差 (`config.yaml` の設定値) |
| 割安 | 20% | 5年平均レートからの乖離 (低いほど割安) |
| トレンド | 20% | 50日/200日移動平均との乖離 |
| 勢い | 15% | 3ヶ月リターン / MACDヒストグラム |
| 安定 | 15% | 年率ボラティリティ (低いほど良い) |

RSI(14) が 75 を超えたペアは過熱としてペナルティ。
スコアは「ロングの魅力度」なので、低スコアは売り方向の妙味を示す場合がある。

## 実行方法

```bash
pip install -r requirements.txt
python -m screener run [--no-notify] [--limit N] [--force-refresh]
```

- 価格データは `data/fx.db` (SQLite) にキャッシュされる
- Discord通知は環境変数 `DISCORD_WEBHOOK_URL` を設定した場合のみ送信

## 出力

- `reports/YYYY-MM-DD.md` — Markdownレポート (全ペアランキング + スコア内訳)
- `reports/index.html` — Webページ版 (GitHub Pagesで公開)

## 自動実行

GitHub Actions で毎週土曜 07:30 JST に実行し、レポートのcommit・GitHub Pagesへのデプロイ・Discord通知を行う (`.github/workflows/screen.yml`)。

## 注意

- 金利差は政策金利の設定値 (`config.yaml` の `rates.policy`) から計算した概算であり、実際のスワップポイントとは異なる。**中銀の利上げ・利下げがあったら手で更新すること**
- 本ツールは機械的なスクリーニングであり投資助言ではない。FXはレバレッジにより損失が拡大するリスクがある
