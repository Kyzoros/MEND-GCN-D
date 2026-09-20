# KEGG 通路富集分析 — 用于 Systems-Mapping 模型验证
# ============================================================
# 用途: 对模型找到的最优草药组合的"共同扰动基因"做 KEGG 富集分析，
#       验证预测结果的生物学合理性。
#
# 输入: 共同扰动基因列表 (gene symbols)，如 "AD_common_genes.txt"
#       该文件由 main.py 在步骤5自动生成
#
# 用法:
#   Rscript kegg_enrichment.R                       # 默认读取 AD_common_genes.txt
#   Rscript kegg_enrichment.R --input my_genes.txt --disease RA
#
# 依赖: install.packages(c("clusterProfiler", "msigdbr", "org.Hs.eg.db", "ggplot2", "dplyr"))
#       BiocManager::install(c("clusterProfiler", "org.Hs.eg.db"))

library(clusterProfiler)
library(msigdbr)
library(org.Hs.eg.db)
library(ggplot2)
library(dplyr)

# ==================== 参数解析 ====================
args <- commandArgs(trailingOnly = TRUE)
input_file <- "AD_common_genes.txt"
output_prefix <- "kegg_enrichment"

if (length(args) >= 1) {
  for (i in seq(1, length(args), by = 2)) {
    if (args[i] == "--input" && i + 1 <= length(args)) {
      input_file <- args[i + 1]
    } else if (args[i] == "--output" && i + 1 <= length(args)) {
      output_prefix <- args[i + 1]
    } else if (args[i] == "--disease" && i + 1 <= length(args)) {
      # 自动匹配疾病对应的共同基因文件
      input_file <- paste0(args[i + 1], "_common_genes.txt")
    }
  }
}

cat("KEGG 富集分析\n")
cat("输入文件:", input_file, "\n")

# ==================== 1. 加载共同扰动基因 ====================
if (!file.exists(input_file)) {
  stop("文件不存在: ", input_file, "\n请先运行 main.py 生成共同基因文件")
}

# 读取基因列表 (gene symbols, 每行一个)
gene_symbols <- readLines(input_file)
gene_symbols <- trimws(gene_symbols)
gene_symbols <- gene_symbols[gene_symbols != ""]
gene_symbols <- unique(gene_symbols)

cat("基因 symbol 数量:", length(gene_symbols), "\n")
cat("前10个基因:", paste(head(gene_symbols, 10), collapse = ", "), "\n")

# ==================== 2. Gene Symbol → Entrez ID 转换 ====================
cat("\n转换 gene symbol → Entrez ID...\n")
gene_map <- bitr(gene_symbols,
                 fromType = "SYMBOL",
                 toType   = "ENTREZID",
                 OrgDb    = org.Hs.eg.db)

entrez_ids <- gene_map$ENTREZID
cat("成功转换:", length(entrez_ids), "/", length(gene_symbols), "个基因\n")

if (length(entrez_ids) < 5) {
  stop("可转换基因数不足 (<5)，无法进行富集分析")
}

# ==================== 3. 准备 KEGG 基因集 ====================
cat("\n加载 MSigDB KEGG 基因集...\n")
all_c2 <- msigdbr(species = "Homo sapiens", category = "C2")

# 严格筛选：只保留 KEGG_ 开头，排除 MEDICUS/REFERENCE
kegg_gene_sets <- all_c2 %>%
  filter(grepl("^KEGG_", gs_name)) %>%
  filter(!grepl("MEDICUS|REFERENCE", gs_name))

cat("可用 KEGG 通路数:", length(unique(kegg_gene_sets$gs_name)), "\n")

# 构建 TERM2GENE 和 TERM2NAME
term2gene <- kegg_gene_sets[, c("gs_name", "entrez_gene")]
term2name <- data.frame(
  gs_name        = unique(kegg_gene_sets$gs_name),
  stringsAsFactors = FALSE
)
term2name$gs_description <- gsub("_", " ",
                                  gsub("^KEGG_", "", term2name$gs_name))

# ==================== 4. 执行富集分析 ====================
cat("\n执行 KEGG 富集分析...\n")
kegg_result <- enricher(
  gene          = entrez_ids,
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.2,
  minGSSize     = 5,
  maxGSSize     = 500,
  TERM2GENE     = term2gene,
  TERM2NAME     = term2name
)

# ==================== 5. 输出结果 ====================
if (!is.null(kegg_result) && nrow(kegg_result@result) > 0) {
  kegg_df <- as.data.frame(kegg_result)
  
  # 添加 gene symbols 列（把 Entrez 转回 symbol）
  kegg_df$geneSymbols <- sapply(kegg_df$geneID, function(ids) {
    eids <- unlist(strsplit(ids, "/"))
    syms <- gene_map$SYMBOL[match(eids, gene_map$ENTREZID)]
    paste(syms[!is.na(syms)], collapse = "/")
  })
  
  # 排序并保存
  kegg_df <- kegg_df[order(kegg_df$p.adjust), ]
  
  csv_path <- paste0(output_prefix, "_results.csv")
  write.csv(kegg_df, csv_path, row.names = FALSE, fileEncoding = "UTF-8")
  cat("\n富集结果已保存:", csv_path, "\n")
  cat("显著通路数:", nrow(kegg_df), "\n\n")
  
  # 打印前15条
  print(head(kegg_df, 15)[, c("Description", "p.adjust", "Count", "geneSymbols")])
  
  # ==================== 6. 可视化 ====================
  # 点图
  p1 <- dotplot(kegg_result, showCategory = min(20, nrow(kegg_df)),
                title = "KEGG Pathway Enrichment",
                size = "Count", color = "p.adjust") +
    theme(axis.text.y = element_text(size = 9))
  
  png_path <- paste0(output_prefix, "_dotplot.png")
  ggsave(png_path, p1, width = 12, height = max(6, nrow(kegg_df) * 0.3),
         dpi = 150)
  cat("点图已保存:", png_path, "\n")
  
  # 条形图
  top10 <- head(kegg_df, 10)
  p2 <- ggplot(top10, aes(x = reorder(Description, -p.adjust),
                           y = -log10(p.adjust), fill = Count)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(x = "", y = "-log10(p.adjust)",
         title = "Top 10 KEGG Enriched Pathways") +
    theme_minimal()
  
  bar_path <- paste0(output_prefix, "_barplot.png")
  ggsave(bar_path, p2, width = 10, height = 6, dpi = 150)
  cat("条形图已保存:", bar_path, "\n")
  
} else {
  cat("\n未找到显著富集的 KEGG 通路\n")
  cat("（这可能意味着共同扰动基因不集中在已知通路中，也是有效发现）\n")
}
