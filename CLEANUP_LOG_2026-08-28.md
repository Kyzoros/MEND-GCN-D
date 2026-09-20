# 项目清理日志（2026-08-28）

## 概述
- 根目录文件数：清理前 130+ → 清理后约 55
- 删除（A 类临时/日志 + B 类废弃脚本）：**71 个**
- 归档到 `_archive/`（C 类，有数据产物依赖）：**19 个**

> 归档 ≠ 删除：C 类脚本生成了仍在使用的数据产物（如 `ingredient_chembl_mapping.csv`、
> `herb_signature_cache_fused.npz`、`Drug_AI_best_view/*.pkl`），
> 为保留可复现性移入 `_archive/`，确认不再需要后可整目录删除。

---

## A 类：真删（临时脚本 / 测试产物 / 日志）

- `_tmp_check_cache.py`
- `_tmp_check_images.py`
- `_tmp_disease_candidates.py`
- `_tmp_extract_imgs.py`
- `_tmp_find_newest.py`
- `_tmp_name_coverage.py`
- `_tmp_pubmed_feasibility.py`
- `_verify_symmap_join.py`
- `_tmp_font_msyh0.pdf`
- `_tmp_font_simsun21.pdf`
- `_tmp_font_simsun0.pdf`
- `_tmp_font_simhei0.pdf`
- `_tmp_rl_test.pdf`
- `scripts/_cleanup_scan_result.json`
- `ablation_expanded_run.log`
- `ablation_run.log`
- `build_paper_tables.log`
- `merge_kegg_ppi.err`
- `merge_kegg_ppi.log`
- `merge_kegg_ppi.pid`
- `merge_run.log`
- `probe_route_c_a.log`
- `retrain_out.txt`
- `run_ext.log`
- `run_fused.err`
- `run_fused.log`
- `run_kegg.err`
- `run_kegg.log`
- `run_kegg_v2.err`
- `run_kegg_v2.log`
- `run_kegg_v2.pid`
- `run_ppi.err`
- `run_ppi.log`
- `run_ppi_v2.err`
- `run_ppi_v2.log`
- `run_ppi_v2.pid`
- `tms_ablation_D.log`

## B 类：真删（已废弃 / 被取代的脚本）

- `generate_paper_pdf.py`
- `generate_paper_framework.py`
- `generate_paper_framework.js`
- `paper_evaluation_extended.py`
- `paper_external_validation.py`
- `paper_extras.py`
- `build_paper_tables.py`
- `tms_ablation.py`
- `probe_route_c_a.py`
- `build_improvement_table.py`
- `build_expanded_gold_standard.py`
- `build_disease_sign_labels.py`
- `AD_RA_herb_disease_sign.csv`
- `AD_RA_sign_validation_report.md`
- `analyze_gold_mechanism.py`
- `analyze_gold_stratified.py`
- `verify_dual_ga.py`
- `build_symmap_herb_map.py`
- `build_symmap_external_gold.py`
- `analyze_multi.py`
- `analyze_multi2.py`
- `figures/make_signature_pca.py`
- `figures/make_figure1.py`
- `run_ad.py`
- `run_hdmap_m.py`
- `run_permutation_top50.py`
- `test_kegg_vs_ppi.py`
- `test_rwr_sensitivity.py`
- `run_kegg.py`
- `run_kegg_batch.py`
- `plot_kegg.py`
- `plot_kegg_batch.py`
- `plot_comparison.py`
- `pubmed_validation.py`

### 取代关系对照
| 已删除（旧） | 现用（新） |
|---|---|
| `generate_paper_pdf.py` / `generate_paper_framework.*` | `make_paper_pdf.py` / `make_paper_pdf_zh.py` |
| `paper_evaluation_extended.py` 等路线A系列 | `run_D5~D8_*.py` |
| `tms_ablation.py` / `probe_route_c_a.py` | `run_D7_ablation_permutation.py` |
| `pubmed_validation.py` | `run_D6_pubmed_validation.py` |
| `figures/make_figure1.py` / `make_signature_pca.py` | `figures/make_fig1_pipeline.py` / `make_fig2_comparison.py` |

## C 类：归档到 `_archive/`

- `补全SMILES_综合方案.py`
- `smiles_batch_complete.py`
- `tcmsp_mol2_to_smiles.py`
- `fetch_smiles.py`
- `verify_smiles_quality.py`
- `expand_gene_mapping_v3.py`
- `fix_content_ratio.py`
- `重复草药.py`
- `merge_kegg_ppi.py`
- `diagnose_drugai_views.py`
- `retrain_drugai_rdkit.py`
- `scripts/dedup_fasta.py`
- `scripts/enhance_ingredient_targets.py`
- `scripts/enhance_validation_set.py`
- `scripts/expand_gold_standard.py`
- `scripts/fetch_chembl_activities.py`
- `scripts/match_chembl_local.py`
- `CLEANUP_PROPOSAL.md`
- `48h_learning_plan.md`

---

## 保留的核心文件（未动）

**运行主干**：`main.py` `config.py` `config.yaml` `data_loader.py` `perturbation_signature.py`
`hdmap_s.py` `hdmap_m.py` `hdcti_adapter.py` `drugai_adapter.py` `kegg_network.py`
`kegg_pathway_filter.py` `temporal_signature.py` `transfer_learning.py`
`evaluation.py` `evaluation_metrics.py` `prepare_diseases.py` `convert_diseases.py`
`apply_filtering.py` `build_morgan_cache.py` `pre_run_check.py`

**论文可复现脚本（补充文件引用，务必保留）**：
`run_D5_correct_compare.py` `run_D6_pubmed_validation.py` `run_D7_ablation_permutation.py`
`run_D8_final_comparison.py` `run_P0a_target_tightening.py` `run_P0b_clean_gold.py`
`run_P0_independent_validation.py` `run_B_probe.py` `run_C_goldstandard_check.py`
`run_herb_external_validation.py` `make_paper_pdf.py` `make_paper_pdf_zh.py`
`figures/make_fig1_pipeline.py` `figures/make_fig2_comparison.py`

**文档**：`README.md` `STATUS.md` `ARCHITECTURE.md` `VERIFICATION_REPORT.md`

---

## ⚠️ 未处理的大文件（占 ~3.7 GB，需你确认后再删）

| 文件 | 大小 | 说明 |
|---|---|---|
| `Drug_AI_best_view/preprocessed_data.pkl` | 1520 MB | DrugAI 模型**预处理中间缓存**；模型已训练完（`best_model_rdkit.pkl`），仅在需重训时才用到 |
| `data/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-5.0.259.tab2.txt` | 368 MB | 酵母 PPI，本项目只用人源 |
| `data/BIOGRID-ORGANISM-LATEST.tab2.zip` | 121 MB | BIOGRID 压缩包，已解压 |
| `data/BIOGRID-ORGANISM-Escherichia_coli_K12_W3110-5.0.259.tab2.txt` | 82 MB | 大肠杆菌 PPI，非人源 |
| `data/BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-5.0.259.tab2.txt` | 36 MB | 裂殖酵母，非人源 |
| `data/BIOGRID-ORGANISM-Arabidopsis_thaliana_Columbia-5.0.259.tab2.txt` | 30 MB | 拟南芥，非人源 |
| `data/ingredient_all.xlsx` vs `data/tcmbank_ingredient_all.xlsx` | 61 MB ×2 | 疑似同源重复，需核对 |

生成时间：2026-08-28 21:23
