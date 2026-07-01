---
Title: "Overview of total least-squares methods"
Authors:
  - Markovsky, Ivan
  - Van Huffel, Sabine
Year: 2007
Venue: Signal Processing
Tags:
  - "total-least-squares"
  - "errors-in-variables"
  - "system-identification"
  - "singular-value-decomposition"
  - "weighted-least-squares"
  - "structured-approximation"
PDF: "[[papers/Markovsky-SigProc2007-Overview_Total_Least-Squares/main.pdf|📃]]"
Import Date: "2026-06-26"
Read Date: 2026-06-26
Executive Summary: "Total least squares (TLS) の古典的手法から weighted TLS, structured TLS への拡張を体系的に整理したサーベイ論文. 古典 TLS は SVD による閉形式解を持つが, weighted/structured TLS は解析解を持たず非凸最適化で解く必要がある. 行列低ランク近似問題としての再定式化を軸に, kernel/image/I-O 表現の 3 種のパラメタリゼーションと対応する最適化アルゴリズムを分類した. デコンボリューション, 線形予測, EIV システム同定への応用を示している."
Citekey: Markovsky-SigProc2007-Overview_Total_Least-Squares
BibTeX Key: markovsky2007overview
DOI: "10.1016/j.sigpro.2007.04.004"
Relevance: 4
Repository: none
Category: note
Template Version: v2.3
---

## Executive Summary
Total least squares (TLS) の古典的手法から weighted TLS, structured TLS への拡張を体系的に整理したサーベイ論文. 古典 TLS は SVD による閉形式解を持つが, weighted/structured TLS は解析解を持たず非凸最適化で解く必要がある. 行列低ランク近似問題としての再定式化を軸に, kernel/image/I-O 表現の 3 種のパラメタリゼーションと対応する最適化アルゴリズムを分類した. デコンボリューション, 線形予測, EIV システム同定への応用を示している.

---
## Summary

### この論文が答えた問い, あるいは解決した課題は何か?

Total least squares の古典的手法とその拡張(weighted, generalized, structured)が分散的に発展してきた状況に対し, 問題定式化・表現・アルゴリズムの 3 軸で統一的に分類・整理する枠組みを提供した. (§1 Introduction)

### 提案手法のアプローチと, その根幹をなす要素は何か?

TLS 問題を"データ行列 $C = [A \; B]$ に最も近い rank 制約付き行列 $\hat{C}$ を求める行列低ランク近似問題"(TLS2) として再定式化した. この定式化を中心に, 以下の構成要素で体系を組み立てている.

- **Eckart–Young–Mirsky の定理**: フロベニウスノルムで最も近い rank-$n$ 近似が SVD の上位 $n$ 特異値で与えられる. 古典 TLS の解法の理論的基盤. (§3.2)
- **3 種のモデル表現**: kernel 表現 ($Rc = 0$), image 表現 ($\operatorname{colspan}(P)$), input/output 表現 ($X^\top a = b$). 各表現が異なる最適化問題の定式化に対応する. (§3.1)
- **Weighted TLS の重み行列 $W$**: 行方向 $W_\ell$ と列方向 $W_r$ の重みで, 観測ごと・変数ごとのノイズ水準の違いを反映. $W = I$ が standard TLS に退化. (§4)
- **Structured TLS のアフィン構造 $S(p)$**: データ行列がパラメータ $p$ のアフィン関数で生成される制約を表現. Toeplitz, Hankel, block 構造などを統一的に扱う. (§5.3)

### 特に参考とした既存研究と, それらと比した提案手法の新規性は何か?

Golub & Van Loan (1980) の古典 TLS アルゴリズム, Van Huffel & Vandewalle (1991) の非一意解への拡張, De Moor (1993) の structured TLS と $L_2$ 近似の関係, Premoli & Rastello (2002) の element-wise weighted TLS を主要な先行研究として位置づけている.

本論文の新規性は個別手法の提案ではなく, TLS2(行列低ランク近似)を統一的な視点として, weighted/structured を含む TLS ファミリー全体を問題の構造と解法アルゴリズムの 2 次元で分類したこと. 特に structured TLS を Riemannian SVD, maximum likelihood PCA, Premoli–Rastello, weighted low rank approximation の 4 手法で整理した Table 1 (§4.2) は, 従来の個別論文では得られない横断的な比較を提供している.

### どのように訓練・最適化したのか?

N/A: 本論文はサーベイ論文であり, 新規アルゴリズムの訓練・最適化実験は含まない. 各手法の解法アルゴリズムを理論的に記述している.

- **損失関数 / 最適化目的**: 古典 TLS は $\|\Sigma_2\|_F = \sqrt{\sigma^2_{n+1} + \cdots + \sigma^2_{n+d}}$ の最小化. weighted TLS は $\|\sqrt{W_\ell} (C - \hat{C}) \sqrt{W_r}\|_F$ の最小化. structured TLS は $\|\Delta p\|$ subject to $\operatorname{rank}(S(p - \Delta p)) \leq n$ の最小化.
- **データセット**: N/A(サーベイ論文)

### どのように検証したか? 指標と結果は?

N/A: 定量的な実験評価は含まない. §5.1 でデコンボリューション, 線形予測, EIV システム同定の 3 つの応用例を理論的に示し, 各問題が structured TLS に帰着することを導出している.

### 検証結果に基づいた議論, 明らかになった課題はあるか?

(§6 Conclusions) 古典 TLS と generalized TLS は SVD で全大域最適解を分類できるが, weighted TLS と structured TLS は非凸最適化に頼らざるを得ず, 局所最適解しか保証されない. これが TLS 手法の階層における重要な分水嶺であると述べている.

(§4.2) weighted TLS の解法として Table 1 に挙げた 4 手法のうち, Riemannian SVD (De Moor) は収束性の証明がなく, maximum likelihood PCA (Wentzell et al.) は線形収束で大域最適性は保証されない. Premoli–Rastello 法はヒューリスティックであり, 最適性条件の近似解を求める.

(§5.2, §5.3) structured TLS では, アフィン構造の場合に内側最小化を解析的に解いて外側を非線形最小二乗問題 (STLS'_X) に帰着させることで計算量を削減できるが, 大域最適性は依然として保証されない.

---
## 自身の研究との関連

UR5e の慣性パラメータ同定で OLS+bias と TLS+bias を比較しており, TLS+bias で $I_6$ 列にも摂動を許してしまう問題の理論的背景がこの論文で整理されている. §4 の weighted TLS で列ごとの重み $W_r$ を設定すれば $I_6$ 列を実質的に固定でき, 理論的に正しい TLS+bias が実現できる. ただし閉形式解がなく反復法が必要になる点を踏まえると, OLS+bias で十分な現状のセットアップでは実装の動機が薄い.

[[papers/Kubus-IROS2008-Recursive_Total_Least-Squares/on-line-estimation-of-inertial-parameters-using-a-recursive-total-least-squares-approach|Kubus+ 2008]] が recursive TLS を慣性パラメータ同定に適用しており, 本論文の §2.4 で述べられた TLS の統計的性質(弱一致推定量, EIV モデルの最尤推定量との関係)がその理論的根拠にあたる.

---
## 追加議論


---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@article{markovsky2007overview,
  author    = {Markovsky, Ivan and Van Huffel, Sabine},
  title     = {Overview of total least-squares methods},
  journal   = {Signal Processing},
  volume    = {87},
  number    = {10},
  pages     = {2283--2302},
  year      = {2007},
  doi       = {10.1016/j.sigpro.2007.04.004},
  publisher = {Elsevier}
}
```
</details>
