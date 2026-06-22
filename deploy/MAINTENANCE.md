# ForgeCal 月1メンテナンス手順

VPS（ConoHa, 160.251.168.249）で ChirpStack/IoT と同居している ForgeCal の
月1点検。所要 約5分。SSH接続してから実行する。

> ⚠️ ここは **ForgeCal だけ** の手順。ChirpStack/IoT 側の月1メンテは別ファイル
> （デスクトップ `マニュアル/01_m2_chirpstack_初期設定.docx` 10章）を参照。

```bash
ssh root@160.251.168.249
```

## ① コンテナ稼働確認
```bash
cd /opt/plasticfem && docker compose -f web/docker-compose.yml ps
curl -s http://127.0.0.1:8000/api/queue        # {"waiting":0,...} が返ればOK
```
- `Up` でなければ `docker compose -f web/docker-compose.yml up -d`

## ② IoT に影響していないか（同居の健全性）
```bash
free -h                                         # 空きRAM・swap
(cd /root/chirpstack-docker && docker compose ps | grep -c Up)   # 8 が正常
docker stats --no-stream forge-calc             # mem が 768MiB 以内
```

## ③ ディスクと不要データの掃除
```bash
df -h /                                          # 空き確認
docker image prune -f                            # 再ビルドで残った古いイメージ削除
docker builder prune -f                          # 古いビルドキャッシュ削除
du -sh /var/lib/docker/volumes/forge-calc_forge_jobs/_data 2>/dev/null  # ジョブ結果サイズ
```
- ジョブ結果は自動削除（直近20件＋24h）なので通常は数十MB程度。

## ④ TLS証明書（自動更新の確認）
```bash
certbot certificates | grep -A2 fem.matsumoto-works.jp
```
- `VALID: XX days` を確認。30日以下なら手動更新（cs と同じ手順、DNS一時グレー化）。

## ⑤ ログにエラーが無いか（任意）
```bash
docker compose -f web/docker-compose.yml logs --tail 50
```

✅ ①〜④が正常なら完了。異常があれば Claude に出力を共有して相談。
