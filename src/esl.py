"""
ESL & ESL-PSC sklearn 风格实现
=================================
ESL:     Kumar & Sharma (2021), Molecular Biology and Evolution
ESL-PSC: Allard et al. (2025), Nature Communications

提供两个核心类:
  - ESLClassifier:    标准 ESL, 基于 Sparse Group LASSO 逻辑回归
  - ESLPSCClassifier: ESL-PSC, 在 PSC 配对设计下构建 ensemble 模型

依赖: numpy, pandas, scikit-learn, group-lasso
      pip install group-lasso
"""

import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from group_lasso import LogisticGroupLasso
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================================
#  数据预处理工具
# =====================================================================

def one_hot_encode_sequences(seq_df, valid_chars=None):
    """
    对氨基酸序列 DataFrame 进行 one-hot 编码.

    Parameters
    ----------
    seq_df : pd.DataFrame
        行=物种, 列=gene_pos 格式的位置名, 元素为单字符氨基酸或 '-'.
    valid_chars : set or None
        合法字符集. 默认 20 种氨基酸 + gap.

    Returns
    -------
    X : np.ndarray, shape (n_species, n_features)
    feature_names : list[str]  — 每列对应 "gene_pos_AA"
    group_ids : np.ndarray     — 每列所属基因组编号
    position_names : list[str] — 每列对应的原始位置名
    gene_names : list[str]     — 去重后的基因名列表
    """
    if valid_chars is None:
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY-")

    features, feature_names, group_ids, position_names = [], [], [], []

    # 提取基因名: 列名 "GeneA_12" -> 基因名 "GeneA"
    gene_name_set = []
    for col in seq_df.columns:
        gname = col.rsplit("_", 1)[0] if "_" in col else col
        if gname not in gene_name_set:
            gene_name_set.append(gname)
    gene_to_id = {g: i for i, g in enumerate(gene_name_set)}

    for col in seq_df.columns:
        vals = seq_df[col].values
        unique_chars = sorted(set(c for c in vals if c in valid_chars))
        if len(unique_chars) <= 1:          # 跳过单态位点
            continue
        gname = col.rsplit("_", 1)[0] if "_" in col else col
        gid = gene_to_id[gname]
        for ch in unique_chars:
            features.append((vals == ch).astype(np.float64))
            feature_names.append(f"{col}_{ch}")
            group_ids.append(gid)
            position_names.append(col)

    X = np.column_stack(features) if features else np.empty((len(seq_df), 0))
    return X, feature_names, np.array(group_ids), position_names, gene_name_set


# =====================================================================
#  ESLClassifier
# =====================================================================

class ESLClassifier(BaseEstimator, ClassifierMixin):
    """
    ESL (Evolutionary Sparse Learning) 分类器.

    使用 Sparse Group LASSO 逻辑回归:
        L(β) = logistic_loss(β) + λ₁‖β‖₁ + λ₂ Σ_g w_g ‖β_g‖₂

    Parameters
    ----------
    lambda1 : float, default=0.01
        位点级 L1 稀疏参数.
    lambda2 : float, default=0.01
        基因组级 group-L2 稀疏参数.
    max_iter : int, default=1000
        优化最大迭代次数.
    tol : float, default=1e-5
        收敛容差.

    Attributes (fit 后可用)
    -----------------------
    coef_ : np.ndarray, shape (n_features,)
    intercept_ : float
    feature_names_ : list[str]
    group_ids_ : np.ndarray
    position_names_ : list[str]
    gene_names_ : list[str]
    classes_ : np.ndarray, [0, 1]

    Examples
    --------
    >>> clf = ESLClassifier(lambda1=0.01, lambda2=0.01)
    >>> clf.fit(X_train, y_train, group_ids, gene_names=gene_names)
    >>> y_pred = clf.predict(X_test)
    >>> proba  = clf.predict_proba(X_test)
    """

    def __init__(self, lambda1=0.01, lambda2=0.01, max_iter=1000, tol=1e-5):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, group_ids,
            feature_names=None, position_names=None, gene_names=None):
        """
        训练模型.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            one-hot 编码矩阵 (由 one_hot_encode_sequences 生成).
        y : array-like, shape (n_samples,)
            标签, 0 或 1.
        group_ids : np.ndarray, shape (n_features,)
            每个特征所属的基因组编号.
        feature_names, position_names, gene_names :
            可选元信息, 用于后续 sparsity score 计算.

        Returns
        -------
        self
        """
        self.classes_ = np.array([0, 1])
        self.feature_names_ = feature_names
        self.group_ids_ = group_ids
        self.position_names_ = position_names
        self.gene_names_ = gene_names

        y = np.asarray(y, dtype=int)

        if len(np.unique(group_ids)) > 1:
            gl = LogisticGroupLasso(
                groups=group_ids,
                group_reg=self.lambda2,
                l1_reg=self.lambda1,
                n_iter=self.max_iter,
                tol=self.tol,
                supress_warning=True,
            )
            gl.fit(X, y)
            if gl.coef_.ndim == 2:
                self.coef_ = gl.coef_[:, 1] - gl.coef_[:, 0]
            else:
                self.coef_ = gl.coef_.flatten()
            if hasattr(gl.intercept_, '__len__') and len(gl.intercept_) > 1:
                self.intercept_ = float(gl.intercept_[1] - gl.intercept_[0])
            else:
                self.intercept_ = float(gl.intercept_)
        else:
            # 只有单基因时回退到 sklearn L1 逻辑回归
            C = 1.0 / max(self.lambda1 * X.shape[0], 1e-6)
            lr = LogisticRegression(
                penalty='l1', C=C, solver='saga',
                max_iter=self.max_iter, tol=self.tol, random_state=42,
            )
            lr.fit(X, y)
            self.coef_ = lr.coef_.flatten()
            self.intercept_ = float(lr.intercept_[0])

        return self

    # ---------- 预测 ----------

    def decision_function(self, X):
        """Sequence Prediction Score (SPS) = Xβ + b"""
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """
        返回 shape (n_samples, 2) 的概率矩阵, 列 0 / 1 对应类别 0 / 1.
        SPP = sigmoid(SPS)
        """
        sps = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-sps))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        """返回 0/1 预测标签."""
        return (self.decision_function(X) > 0).astype(int)

    # ---------- Sparsity Scores ----------

    def get_PSS(self):
        """Position Sparsity Score: 每个位点的绝对系数之和."""
        pss = {}
        if self.position_names_ is None:
            return pss
        for i, pos in enumerate(self.position_names_):
            pss[pos] = pss.get(pos, 0.0) + abs(self.coef_[i])
        return dict(sorted(pss.items(), key=lambda x: -x[1]))

    def get_GSS(self):
        """Gene (Group) Sparsity Score: 每个基因的绝对系数之和."""
        gss = {}
        if self.gene_names_ is None:
            return gss
        for i, gid in enumerate(self.group_ids_):
            g = self.gene_names_[gid]
            gss[g] = gss.get(g, 0.0) + abs(self.coef_[i])
        return dict(sorted(gss.items(), key=lambda x: -x[1]))

    def get_HSS(self):
        """Hypothesis Sparsity Score: 所有 GSS 之和."""
        return sum(self.get_GSS().values())

    def get_selected_genes(self, threshold=0.0):
        """返回 GSS > threshold 的基因."""
        return {g: s for g, s in self.get_GSS().items() if s > threshold}

    def get_selected_positions(self, threshold=0.0):
        """返回 PSS > threshold 的位点."""
        return {p: s for p, s in self.get_PSS().items() if s > threshold}


# =====================================================================
#  ESL-PSC 并行训练 worker（模块级函数，供 multiprocessing pickle）
# =====================================================================

def _train_one_psc_model(task):
    """
    训练单个 ESL 模型（供 ESLPSCClassifier.fit 多进程调用）。

    Parameters
    ----------
    task : tuple
        (X_psc, y_psc, group_ids, feature_names, position_names, gene_names,
         l1, l2, max_iter, tol)

    Returns
    -------
    dict or None — {"clf": ESLClassifier, "mfs": float}
    """
    (X_psc, y_psc, group_ids, feature_names, position_names, gene_names,
     l1, l2, max_iter, tol) = task
    try:
        clf = ESLClassifier(lambda1=l1, lambda2=l2, max_iter=max_iter, tol=tol)
        clf.fit(X_psc, y_psc, group_ids,
                feature_names=feature_names,
                position_names=position_names,
                gene_names=gene_names)
        sps = clf.decision_function(X_psc)
        mfs = float(np.sqrt(np.mean(((y_psc * 2 - 1) - sps) ** 2)))
        return dict(clf=clf, mfs=mfs)
    except Exception:
        return None


# =====================================================================
#  ESLPSCClassifier
# =====================================================================

class ESLPSCClassifier(BaseEstimator, ClassifierMixin):
    """
    ESL-PSC (Paired Species Contrast) 分类器.

    在 PSC 设计下构建多个 ESL 模型的 ensemble, 用于预测趋同性状.

    Parameters
    ----------
    lambda1_list : list[float]
        待搜索的 λ₁ 值.
    lambda2_list : list[float]
        待搜索的 λ₂ 值.
    top_pct : float, default=0.2
        按 MFS 排序后保留 top 多少比例的模型做 ensemble.
    n_alternates : int, default=3
        每个 order 内最多取多少种物种组合.
    order_col : str, default="order_chinese"
        meta_df 中表示物种 order 的列名.
    label_col : str, default="label"
        meta_df 中表示标签的列名.

    Examples
    --------
    >>> clf = ESLPSCClassifier()
    >>> clf.fit(seq_df_train, meta_df_train)
    >>> pred = clf.predict_species(seq_df_full, "Bat_eco_0")
    """

    def __init__(self, lambda1_list=None, lambda2_list=None,
                 top_pct=0.2, n_alternates=3,
                 order_col="order_chinese", label_col="label",
                 n_cpu=1, verbose=False):
        self.lambda1_list = lambda1_list or [0.005, 0.01, 0.05, 0.1]
        self.lambda2_list = lambda2_list or [0.005, 0.01, 0.05, 0.1]
        self.top_pct = top_pct
        self.n_alternates = n_alternates
        self.order_col = order_col
        self.label_col = label_col
        self.n_cpu = n_cpu
        self.verbose = verbose
        self.classes_ = np.array([0, 1])

    # ---------- PSC 配对 ----------

    @staticmethod
    def _select_psc_pairs(meta_df, order_col, label_col):
        """为每个含有正/负样本的 order 选一对."""
        eco = meta_df[meta_df[label_col] == 1]
        non = meta_df[meta_df[label_col] == 0]
        pairs, used = [], set()
        for order in meta_df[order_col].unique():
            if order in used:
                continue
            e = eco[eco[order_col] == order]
            n = non[non[order_col] == order]
            if len(e) > 0 and len(n) > 0:
                pairs.append((e.index[0], n.index[0]))
                used.add(order)
        return pairs

    def _generate_psc_combinations(self, meta_df):
        eco = meta_df[meta_df[self.label_col] == 1]
        non = meta_df[meta_df[self.label_col] == 0]
        order_pairs = {}
        for order in meta_df[self.order_col].unique():
            e = eco[eco[self.order_col] == order]
            n = non[non[self.order_col] == order]
            if len(e) > 0 and len(n) > 0:
                pairs = list(product(e.index, n.index))
                order_pairs[order] = pairs[:self.n_alternates]
        if not order_pairs:
            return []
        return [list(c) for c in product(*order_pairs.values())]

    # ---------- fit ----------

    def fit(self, seq_df, meta_df, max_iter, tol):
        """
        训练 ESL-PSC ensemble.

        Parameters
        ----------
        seq_df : pd.DataFrame  — 序列数据 (行=物种, 列=gene_pos)
        meta_df : pd.DataFrame — 元数据 (行=物种, 必须含 label_col 和 order_col)

        Returns
        -------
        self
        """
        self.seq_df_ = seq_df
        self.meta_df_ = meta_df

        # 用全体训练物种做统一的 one-hot 编码, 保证维度一致
        species_list = list(seq_df.index)
        X_all, fn, gid, pn, gn = one_hot_encode_sequences(seq_df)
        self.encoding_info_ = dict(
            species_list=species_list, feature_names=fn,
            group_ids=gid, position_names=pn, gene_names=gn,
            n_feat=X_all.shape[1],
        )
        sp_to_idx = {s: i for i, s in enumerate(species_list)}

        pairs_list = self._generate_psc_combinations(meta_df)
        if not pairs_list:
            pairs = self._select_psc_pairs(meta_df, self.order_col, self.label_col)
            pairs_list = [pairs] if pairs else []

        n_lambda = len(self.lambda1_list) * len(self.lambda2_list)
        n_total = len(pairs_list) * n_lambda
        if self.verbose:
            print(f"  [ESL-PSC] {len(pairs_list)} PSC 组合 × {n_lambda} λ 对 = {n_total} 个模型待训练")

        # ---- 收集所有训练任务 ----
        tasks = []
        for pairs in pairs_list:
            psc_species = list(dict.fromkeys(
                s for pair in pairs for s in pair
            ))
            if len(psc_species) < 4:
                continue
            psc_idx = [sp_to_idx[s] for s in psc_species]
            X_psc = X_all[psc_idx].copy()  # copy 避免子进程共享大矩阵的视图问题
            y_psc = np.array([int(meta_df.loc[s, self.label_col]) for s in psc_species])

            for l1 in self.lambda1_list:
                for l2 in self.lambda2_list:
                    tasks.append((X_psc, y_psc, gid, fn, pn, gn,
                                  l1, l2, max_iter, tol))

        # ---- 并行训练所有模型 ----
        n_workers = min(self.n_cpu, len(tasks)) if len(tasks) > 0 else 1
        if self.verbose:
            print(f"  [ESL-PSC] 并行训练 {len(tasks)} 个模型 (workers={n_workers})")

        if n_workers > 1 and len(tasks) > 1:
            with Pool(processes=n_workers) as pool:
                raw_results = list(tqdm(
                    pool.imap(_train_one_psc_model, tasks),
                    total=len(tasks),
                    desc="  PSC models",
                    disable=not self.verbose,
                ))
        else:
            raw_results = []
            it = tqdm(tasks, desc="  PSC models", disable=not self.verbose) if self.verbose else tasks
            for task in it:
                raw_results.append(_train_one_psc_model(task))

        self.models_ = [r for r in raw_results if r is not None]

        # 按 MFS 保留 top models
        if self.models_:
            self.models_.sort(key=lambda m: m["mfs"])
            n = max(1, int(len(self.models_) * self.top_pct))
            self.top_models_ = self.models_[:n]
        else:
            self.top_models_ = []

        return self

    # ---------- 预测 ----------

    def _encode_species(self, seq_df, species_name):
        """
        用训练时存储的编码 schema 直接构造新物种的特征向量.
        不重新编码, 保证维度与训练时严格一致.
        """
        row = seq_df.loc[species_name]
        fn = self.encoding_info_["feature_names"]
        n_feat = self.encoding_info_["n_feat"]
        x = np.zeros(n_feat, dtype=np.float64)
        for j, fname in enumerate(fn):
            # fname 格式: "GeneA_12_A" -> position="GeneA_12", char="A"
            parts = fname.rsplit("_", 1)
            pos_name = parts[0]
            char = parts[1]
            if pos_name in row.index and row[pos_name] == char:
                x[j] = 1.0
        return x.reshape(1, -1)

    def predict_species(self, seq_df, species_name):
        """
        预测单个物种的标签和概率.

        Parameters
        ----------
        seq_df : pd.DataFrame — 包含该物种行的完整序列 DataFrame.
        species_name : str

        Returns
        -------
        label : int (0 或 1)
        proba : float (属于类别 1 的概率)
        sps   : float (Sequence Prediction Score 均值)
        """
        if not self.top_models_:
            return 0, 0.5, 0.0

        X_test = self._encode_species(seq_df, species_name)
        sps_list = [m["clf"].decision_function(X_test)[0] for m in self.top_models_]
        mean_sps = float(np.mean(sps_list))
        proba = 1.0 / (1.0 + np.exp(-mean_sps))
        return int(mean_sps > 0), proba, mean_sps

    def predict(self, seq_df, species_list):
        """
        批量预测多个物种.

        Returns
        -------
        pd.DataFrame — columns: species, pred_label, pred_proba, SPS
        """
        rows = []
        for sp in species_list:
            label, proba, sps = self.predict_species(seq_df, sp)
            rows.append(dict(species=sp, pred_label=label, pred_proba=proba, SPS=sps))
        return pd.DataFrame(rows)

    def get_gene_ranks(self):
        """返回基因排名 (按 ensemble 中最大 GSS 排序)."""
        gene_gss = {}
        for m in self.top_models_:
            for g, s in m["clf"].get_GSS().items():
                if s > 0:
                    gene_gss.setdefault(g, []).append(s)
        ranks = {g: dict(max_gss=max(v), mean_gss=float(np.mean(v)), n_models=len(v))
                 for g, v in gene_gss.items()}
        return dict(sorted(ranks.items(), key=lambda x: -x[1]["max_gss"]))


# =====================================================================
#  便捷函数: Leave-One-Species-Out 评估
# =====================================================================

def leave_one_out_eval(seq_df, meta_df, method="esl",
                       label_col="label", order_col="order_chinese",
                       **kwargs):
    """
    Leave-One-Species-Out 交叉验证便捷函数.

    Parameters
    ----------
    seq_df : pd.DataFrame
    meta_df : pd.DataFrame
    method : str, "esl" 或 "esl_psc"
    label_col, order_col : str
    **kwargs : 传给 ESLClassifier 或 ESLPSCClassifier 的参数.

    Returns
    -------
    results : pd.DataFrame — 逐物种预测结果
    metrics : dict — accuracy, balanced_accuracy, AUC
    """
    species_list = list(seq_df.index)
    n = len(species_list)
    rows = []

    if method == "esl":
        X_all, fn, gid, pn, gn = one_hot_encode_sequences(seq_df)
        y_all = np.array([int(meta_df.loc[s, label_col]) for s in species_list])

        for i, sp in enumerate(species_list):
            mask = np.ones(n, dtype=bool); mask[i] = False
            clf = ESLClassifier(**kwargs)
            clf.fit(X_all[mask], y_all[mask], gid,
                    feature_names=fn, position_names=pn, gene_names=gn)
            pred = clf.predict(X_all[i:i+1])[0]
            proba = clf.predict_proba(X_all[i:i+1])[0, 1]
            rows.append(dict(
                species=sp, true_label=int(y_all[i]),
                pred_label=pred, pred_proba=proba,
            ))

    elif method == "esl_psc":
        for i, sp in enumerate(species_list):
            train_sp = [s for s in species_list if s != sp]
            psc = ESLPSCClassifier(
                order_col=order_col, label_col=label_col, **kwargs
            )
            psc.fit(seq_df.loc[train_sp], meta_df.loc[train_sp])
            pred, proba, _ = psc.predict_species(seq_df, sp)
            rows.append(dict(
                species=sp, true_label=int(meta_df.loc[sp, label_col]),
                pred_label=pred, pred_proba=proba,
            ))
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n}]")

    results = pd.DataFrame(rows)
    acc = accuracy_score(results["true_label"], results["pred_label"])
    bal = balanced_accuracy_score(results["true_label"], results["pred_label"])
    try:
        auc = roc_auc_score(results["true_label"], results["pred_proba"])
    except Exception:
        auc = float("nan")
    return results, dict(accuracy=acc, balanced_accuracy=bal, AUC=auc)
