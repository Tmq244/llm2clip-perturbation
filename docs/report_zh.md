# LLM2CLIP 与 CLIP 语义扰动实验报告

## 实验设置

本实验比较两个模型在 MSCOCO 2014 5k image-text retrieval 测试集上的图文相似度变化：

- CLIP：`openai/clip-vit-large-patch14-336`
- LLM2CLIP：`microsoft/LLM2CLIP-Openai-L-14-336` + `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`

对每个图像-caption 对，计算：

```text
s_original = cosine(image_emb, text_emb(original_caption))
s_corrupted = cosine(image_emb, text_emb(corrupted_caption))
delta = s_original - s_corrupted
```

`delta` 越大，表示扰动后图文相似度下降越明显。对于 object swap 和 color/spatial swap，这通常表示模型更能识别 caption 中的语义错误。对于 semantic distractor，`delta` 表示模型受到额外无关文本影响的程度；它既可以被理解为对附加语义更敏感，也可能意味着对无关信息的鲁棒性较弱。

## 扰动类型

| 扰动 | 构造方式 | 样本数 |
|---|---|---:|
| Semantic distractor | 在原 caption 外加入另一张图的无关 caption，并用多种模板标注相关/无关信息 | 25010 |
| Object swap | 替换 caption 中的核心物体词，例如 `horse -> bicycle` | 14233 |
| Color/spatial swap | 替换颜色词或空间关系词，例如 `red -> blue`、`on -> under` | 18131 |

## 图表

### 1. 扰动敏感度

![Sensitivity to perturbation](../outputs/semantic_perturbation_eval/figures_v2/01_sensitivity_to_perturbation.png)

该图展示三种扰动下的平均 `delta`。可以看到：

- 在 `object_swap` 上，LLM2CLIP 的 `delta_mean = 0.0671`，明显高于 CLIP 的 `0.0356`，说明 LLM2CLIP 对核心物体替换更敏感。
- 在 `attribute_spatial_swap` 上，LLM2CLIP 的 `delta_mean = 0.0210`，也高于 CLIP 的 `0.0057`，说明 LLM2CLIP 对颜色/空间扰动更敏感。
- 在 `semantic_distractor` 上，CLIP 的 `delta_mean = 0.0439`，高于 LLM2CLIP 的 `0.0302`。在当前模板设置下，CLIP 的相似度被额外无关描述拉低得更多。

### 2. 加扰动前后的相似度对比

![Similarity original vs perturbed](../outputs/semantic_perturbation_eval/figures_v2/02_similarity_original_vs_perturbed.png)

该图直接比较 `s_original_mean` 和 `s_corrupted_mean`：

- 原始 caption 上，LLM2CLIP 的平均相似度更高：`0.2817` vs CLIP 的 `0.2635`。
- Object swap 后，LLM2CLIP 从 `0.2852` 降到 `0.2181`，下降幅度明显大于 CLIP。
- Color/spatial swap 后，两个模型的下降都较小，但 LLM2CLIP 下降更明显。
- Semantic distractor 后，CLIP 从 `0.2635` 降到 `0.2196`，LLM2CLIP 从 `0.2817` 降到 `0.2515`。这说明显式标注“无关描述”后，LLM2CLIP 相对更能保留目标 caption 的相似度。

### 3. 扰动使相似度下降的样本比例

![Positive delta rate](../outputs/semantic_perturbation_eval/figures_v2/03_positive_delta_rate.png)

该图展示 `s_original > s_corrupted` 的样本比例：

- Semantic distractor：CLIP 为 `93.9%`，LLM2CLIP 为 `75.0%`。
- Object swap：CLIP 为 `93.9%`，LLM2CLIP 为 `95.0%`。
- Color/spatial swap：CLIP 为 `62.9%`，LLM2CLIP 为 `75.3%`。

这说明 object swap 对两个模型都很稳定；color/spatial swap 更难，但 LLM2CLIP 的稳定性更好。Semantic distractor 中，LLM2CLIP 不总是降低相似度，可能是因为模板明确告诉模型哪些句子是 unrelated，从而减弱了无关 caption 的影响。

### 4. Delta 分布

![Delta distribution](../outputs/semantic_perturbation_eval/figures_v2/04_delta_distribution_boxplot.png)

箱线图展示了每类扰动下 delta 的分布。相比只看均值，它能反映样本间差异：

- Object swap 的分布整体为正，说明大多数替换都会降低相似度。
- Color/spatial swap 的分布更靠近 0，说明这类细粒度扰动对模型更难。
- LLM2CLIP 在 color/spatial swap 上的分布整体高于 CLIP，支持其对属性/空间变化更敏感的结论。

### 5. Semantic distractor 不同模板的影响

![Semantic distractor by template](../outputs/semantic_perturbation_eval/figures_v2/05_semantic_distractor_by_template.png)

该图按 5 个 semantic distractor 模板分别统计 `delta_mean`。这个图主要用于检查模板措辞是否影响结果。

当前实验使用的模板包括：

- `target_unrelated`
- `revert target_unrelated`
- `caption_unrelated_note`
- `image_shows_unrelated_sentence`
- `ignore_unrelated_actual`

如果某个模板的 delta 明显不同，说明模型不仅受无关 caption 内容影响，也受提示格式、句子顺序和是否显式要求忽略无关句子的影响。

## 结果汇总

| 模型 | 扰动 | n | s_original_mean | s_corrupted_mean | delta_mean | positive_rate |
|---|---|---:|---:|---:|---:|---:|
| CLIP | Original | 25010 | 0.2635 | - | - | - |
| LLM2CLIP | Original | 25010 | 0.2817 | - | - | - |
| CLIP | Semantic distractor | 25010 | 0.2635 | 0.2196 | 0.0439 | 0.9389 |
| LLM2CLIP | Semantic distractor | 25010 | 0.2817 | 0.2515 | 0.0302 | 0.7504 |
| CLIP | Object swap | 14233 | 0.2645 | 0.2289 | 0.0356 | 0.9392 |
| LLM2CLIP | Object swap | 14233 | 0.2852 | 0.2181 | 0.0671 | 0.9497 |
| CLIP | Color/spatial swap | 18131 | 0.2640 | 0.2583 | 0.0057 | 0.6289 |
| LLM2CLIP | Color/spatial swap | 18131 | 0.2832 | 0.2622 | 0.0210 | 0.7533 |

## 主要结论

核心结论是一个理想的**鲁棒性表现**：LLM2CLIP **对无关噪声更容忍**，同时**对有意义的语义变化更敏感**——这正是理想视觉语言模型应有的特性。

1. **对无关噪声更容忍（semantic distractor）**：加入带「无关 / 忽略」标注的干扰 caption 后，CLIP 相似度下降更大（`delta = 0.0439` vs `0.0302`，受影响样本 `93.9%` vs `75.0%`）。LLM2CLIP 更好地保留目标 caption 的相似度——它可能利用模板中的指令信息（如 `unrelated`、`ignore`、`actual image shows`）过滤无关文本，而 CLIP 做不到。不过该结论需结合模板分组结果（见第 5 节）进一步验证，因为句子顺序和指令强度也会影响模型行为。

2. **对有意义的语义变化更敏感（object & color/spatial swap）**：物体替换下 LLM2CLIP 相似度下降明显更大（`delta = 0.0671` vs `0.0356`，约翻倍；两模型都约 `94–95%` 样本下降）。颜色/空间替换下也更大（`0.0210` vs `0.0057`），但两者的绝对 delta 都很小（CLIP 的 25% 分位 delta ≤ 0），颜色/空间仍是共同短板。

3. **干净对齐更强**：原始 caption 上 LLM2CLIP 平均相似度更高（`0.2817` vs `0.2635`）。

> 补充说明：两个模型的相似度量纲不同（干净基线 `0.264` vs `0.282`），上面的 delta 按原始值比较；方向在按相对下降归一化后仍然成立，但“约翻倍”这类说法应理解为定性结论，而非精确倍数。

## 后续建议

- 对 semantic distractor 按模板分别报告结果，而不仅看总体均值。
- 将 semantic distractor 分成两类：相关 caption 在前 vs 无关 caption 在前，检查顺序效应。
- 单独比较带 `ignore` 指令和不带 `ignore` 指令的模板，分析 LLM2CLIP 是否更善于利用自然语言指令。
- 补充 image-to-text retrieval 的 Recall@1/5/10，避免只依赖 cosine 均值解释模型优劣。
