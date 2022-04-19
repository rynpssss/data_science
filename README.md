 # 独学実装　データサイエンス
モデルごとに理論・実装をノートブックにまとめ、スクリプトにリファクタリングまで行っています。
### 理論・実装・リファクタリング済みモデル
- GLM
- ロジスティック回帰

### 理論・実装済みモデル
- GLMM
- ARIMA / SARIMA
- SVM
- GBDT
- ランダムフォレスト

### 理論済みモデル
- PCA
- LDA
- 因子分析
- K-means
- 決定木
  
 ## 開発環境
 JupyterLabおよびVScodeにて開発中
 ## 開発手順
 Githubの個人Gitを使用して開発中<br>
 https://github.com/rynpssss/data_science.git

個人Pushのため、ブランチを1つずつ切ってから都度commitしている状況です。<br>
そのため、コンフリクトも起きずに常に直進しています。

PRはメッセージに軽く一言。（感想とか）

 ## ビルド方法
  - ノートブック<br>
    Dockerにて開発環境を構築
    git clone後、以下を実行して自動でJupyterLabを立ち上げています。
    ```
    docker-compose build
    docker-compose up
    ```

 - スクリプト<br>
    venvにて開発環境を構築<br>
    環境へ入る時
    ```
    source ./venv/bin/activate
    ```
    出る時
    ```
    deactivate
    ```
 
 TODO:スクリプト開発環境とノートブック開発環境が異なっているため、変更急務
 ## テスト
debugやloggingを未実装（やり方が分かっていないので、追加予定）

----
## 以下は、マークダウン練習用

$a\eta$

`test = a + b`

```
test = a + b
```

```
pip install pandas
```


$e^{i\pi}$ = -1

$\frac{a+b}{2a}$

$\frac{x+y}{y^{x\pi\eta}}$

$y=x^{\frac{1}{2}}$

$\lim_{x \to \infty}f(x)$

$\lim_{x \to \infty} f(x)$

$\log_ x$