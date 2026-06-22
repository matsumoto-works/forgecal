# forge-calc — VPS デプロイ手順（ChirpStack と同居・干渉ゼロ）

ConoHa VPS (160.251.168.249, Ubuntu 22.04) に、稼働中の ChirpStack/IoT に
**一切触れずに** forge-calc を追加する手順。隔離は (1) 専用 Docker プロジェクト
＋メモリ cgroup 上限、(2) localhost バインド＋nginx 別サイト、(3) CPU 制限 で担保。

## ⚠️ 触ってはいけないもの（厳守）
- `/root/chirpstack-docker/`（compose・DB・設定）
- `sensor-api.service`、毎日 3:00 の cron バックアップ
- nginx の既存 `chirpstack` サイト設定
- 使用中ポート: 80, 443, 1700/udp, 1883, 3000, 3001, 5000, 5001, 8080, 8090, 5432
  （forge-calc は **8000 を localhost のみ** で使う＝衝突なし）

## 前提（サブドメイン方式 — 新規ドメイン登録は不要）
- ホスト名は **`fem.matsumoto-works.jp`**（既存 Cloudflare ゾーンのサブドメイン）。
- Cloudflare の `matsumoto-works.jp` ゾーンに **A レコード `fem → 160.251.168.249`** を追加するだけ
  （ネームサーバ変更・浸透待ち不要）。certbot 取得時のみ一時的に **DNS only（グレー雲）** にする。
- 将来 `forge-calc.jp` 等の独立ドメインにしたくなったら server_name を足すだけで両対応可。

## 1. コードを配置（chirpstack-docker とは別ディレクトリ）
```bash
ssh root@160.251.168.249
mkdir -p /opt/plasticfem && cd /opt/plasticfem
# リポジトリを配置（git or scp）。少なくとも以下が必要:
#   plasticfem/  run_case.py  model_cases/  web/
git clone <repo-url> .        # もしくは scp で転送
```

## 2. コンテナを起動（メモリ/CPU ハード上限つき）
```bash
cd /opt/plasticfem
docker compose -f web/docker-compose.yml up -d --build
docker compose -f web/docker-compose.yml ps        # Up を確認
curl -s http://127.0.0.1:8000/api/cases            # ケース一覧が返れば起動OK
```
- `mem_limit 768m` / `cpus 1.5` がこのコンテナに効く（ChirpStack は別 cgroup で無影響）。
- メモリ確認: `docker stats --no-stream forge-calc`

## 3. TLS 証明書（既存 nginx 設定を編集しない方式）
```bash
# certonly は server ブロックを書き換えない（chirpstack 設定に触れない）
certbot certonly --nginx -d fem.matsumoto-works.jp
# 失敗する場合は webroot か standalone(一時的に80空ける)で取得
```

## 4. nginx 別サイトを追加（chirpstack サイトは不変）
```bash
cp /opt/plasticfem/deploy/nginx-forge-calc.conf \
   /etc/nginx/sites-available/forge-calc
ln -s /etc/nginx/sites-available/forge-calc /etc/nginx/sites-enabled/
nginx -t            # ★必ず通ることを確認してから reload
systemctl reload nginx
```

## 5. 公開確認
```bash
# Cloudflare の A レコードを Proxied（オレンジ雲）に戻す
curl -I https://fem.matsumoto-works.jp     # 200/301 + TLS
```
ブラウザで `https://fem.matsumoto-works.jp` を開き、工法を選んで「解析を実行」。

## 運用
- ログ: `docker compose -f web/docker-compose.yml logs -f`
- 更新: `git pull && docker compose -f web/docker-compose.yml up -d --build`
- 停止: `docker compose -f web/docker-compose.yml down`
- 結果は named volume `forge_jobs`（`/data/jobs`）に保持。**ジョブ結果は自動削除**
  （直近 MAX_JOBS=20 件＋24h、`jobs.py _prune()`）＝溜まり続けない。
- **掃除（再ビルドで溜まる古いイメージ）**: `up -d --build` を繰り返すと古い
  Docker イメージが dangling として残る。月1メンテ時に掃除推奨:
  ```bash
  docker image prune -f      # dangling イメージ削除（稼働中は消えない）
  docker builder prune -f    # 古いビルドキャッシュ削除
  ```
- セキュリティグループ（ConoHa）の変更は **不要**（8000 は localhost、80/443 は既存開放済）。

## ロールバック（IoT に影響が出た場合）
```bash
docker compose -f web/docker-compose.yml down        # コンテナ停止
rm /etc/nginx/sites-enabled/forge-calc && nginx -t && systemctl reload nginx
```
これで forge-calc は完全に消え、ChirpStack 構成は元のまま。
