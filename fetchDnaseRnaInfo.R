# sudo apt-get install libcurl4-openssl-dev libxml2-dev libssl-dev
# source("https://bioconductor.org/biocLite.R")
# biocLite("biomaRt")
# biocLite("BSgenome")
# biocLite('BSgenome.Hsapiens.UCSC.hg19')

library(biomaRt)
library(BSgenome)
library(BSgenome.Hsapiens.UCSC.hg19)

# Combine the rna data nd dnase data
combineRnaDnase <- function(rnaSeqFname, DnaseFname, chrom_num) {
  # Get all Chromosome 19 genes ---------------------------------------------------------------------------------------------------------------------------------------------
  chr19_genes <- getAllChromosomeGenes(chrom_num)
  
  # Read the gene expresison data file --------------------------------------------------------------------------------------------------------------------------------------
  fn = read.table(rnaSeqFname, header = T)
  rnaData <- fn[grep("ENSG",fn$gene_id),] 
  rnaData$gene_id <- gsub("\\.[0-9]+", "",rnaData$gene_id)  
  length(unique(rnaData$gene_id))  
  
  # Combine the two data frame based on ensenmbl id -------------------------------------------------------------------------------------------------------------------------
    common_ensg <- intersect(chr19_genes$ensembl_gene_id, rnaData$gene_id)
  
    # Preprocess Rna-Seq Data
    data <- rnaData[which(rnaData$gene_id %in% common_ensg),]
    t <- unique(chr19_genes[which(chr19_genes$ensembl_gene_id %in% common_ensg),  c('ensembl_gene_id', 'chromosome_name', 'start_position', 'end_position', 'strand')])
    colnames(t) <- c('gene_id', 'chromosome_name', 'start_position', 'end_position', 'strand')
    data <- merge(t, data, by = 'gene_id')
    
    geneExpData <- data.frame()
    for (i in 1 : length(common_ensg)) {
      dat = data[which(data$gene_id == common_ensg[i]), ]
      geneExpData <- rbind(geneExpData, dat[1, c('gene_id', 'chromosome_name', 'start_position', 'end_position', 'strand', 'TPM', 'FPKM')])
    }
  
    #  Appending chr string to each chromosome name
    geneExpData$chromosome_name <- sub("^", "chr", geneExpData$chromosome_name)
    geneExpData[which(geneExpData$strand == -1), ]$strand <- '-'
    geneExpData[which(geneExpData$strand == 1), ]$strand <- '+'

  # Get all the gene sequences --------------------------------------------------------------------------------------------------------------------------------------------------
    geneInfo <- unique(fetchGeneSeq(geneExpData))
    
    geneInfo$rank <- rank(-geneInfo$FPKM, ties.method = 'min')
    geneInfo$percentile <- geneInfo$rank/nrow(geneInfo)
    
  # Get dnase information for the entire chromosome -----------------------------------------------------------------------------------------------------------------------------
    dnasePeakInfo = getChrmDnaseBand(fName = DnaseFname, chrNum = chrom_num)
  
  return(list("GeneInfo" = geneInfo, "dnaseInfo" = dnasePeakInfo))
    
}

# Get all Genes of a given chromosome -------------------------------------------------------------------------------------------------------------------------------------------
getAllChromosomeGenes <- function(chrom_num) {
  grch37 = useEnsembl(biomart="ensembl",GRCh=37, dataset="hsapiens_gene_ensembl")
  chr_genes <- getBM(attributes=c('ensembl_gene_id', 'ensembl_transcript_id','hgnc_symbol','entrezgene','chromosome_name','start_position','end_position', 'strand'), 
                       filters = c('chromosome_name'), values =c(toString(chrom_num)), mart = grch37)  
}

# Get 5k Upstream Gene Sequences ------------------------------------------------------------------------------------------------------------------------------------------------
fetchGeneSeq <- function(df) {
  pos_strand_dat = df[which(df$strand == '+'), ]
  pos_strand_dat$upstream_start <- (pos_strand_dat$start_position - 5000)
  pos_strand_dat$upstream_end <- (pos_strand_dat$start_position - 1)
  pos_strand_seq_data = getSeq(Hsapiens, pos_strand_dat$chromosome_name, start = pos_strand_dat$upstream_start, end = pos_strand_dat$end_position, strand = '+')
  pos_strand_dat$sequence <- as.character(pos_strand_seq_data)
  
  neg_strand_dat = df[which(df$strand == '-'), ]
  neg_strand_dat$upstream_start <- (neg_strand_dat$end_position + 1)
  neg_strand_dat$upstream_end <- (neg_strand_dat$end_position + 5000)
  neg_strand_seq_data = getSeq(Hsapiens, neg_strand_dat$chromosome_name, start = neg_strand_dat$upstream_start, end = neg_strand_dat$upstream_end, strand = '-')
  neg_strand_dat$sequence <- as.character(neg_strand_seq_data)  
  
  return(rbind(pos_strand_dat, neg_strand_dat))
}

# Create a huge band of dnase peak information from start to last location ------------------------------------------------------------------------------------------------------
getChrmDnaseBand <- function(fName, chrNum) {
  
  dnase_data = as.data.frame(read.table(fName))
  colnames(dnase_data) = c('chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak')
  chrLoc = paste0('chr',chrNum)
  
  dat = dnase_data[which(dnase_data$chrom == chrLoc),]
  dnasePeak = rep(0, max(dat$chromEnd))

  # First pass: Put all ones to the places which has dnase 
  for (i in 1:nrow(dat)) {
    t = dat[i,]
    dnasePeak[t$chromStart:t$chromEnd] = 1
  }
  
  # Second pass: Mark all the dnase peaks with 2 
  for (i in 1:nrow(dat)) {
    t = dat[i,]
    dnasePeak[t$chromStart + t$peak] = 2
  }
  
  return(dnasePeak)
}

# Get the data required for the 4 tissues
# data = combineRnaDnase('./../data/tissue/Lung/samp1_37yrsm_rna.tsv', './../data/tissue/Lung/samp1_37yrsm_dnase.bed', 19)
setwd("~/Documents/Projects/GeneExpPred")
data = combineRnaDnase('./olddata/Brain/ENCFF691GEO.tsv', './olddata/Brain/ENCFF399UZY.bed', 19)
geneInfo <- data$GeneInfo
geneInfo$tissue <- 'Brain'
dnaseInfo <- data$dnaseInfo

load('./data/keepGeneIds.Rda')
geneInfo = geneInfo[which(geneInfo$gene_id %in% keepInds), ]
geneInfo$rank <- rank(-geneInfo$FPKM, ties.method = 'min')
geneInfo$percentile <- geneInfo$rank/nrow(geneInfo)

write.table(geneInfo, file = './data/Brain/brain_geneInfo.csv', sep = ',', row.names = F, quote = F)
write.table(dnaseInfo, file = './data/Brain/brain_dnaseInfo.csv', sep = ',', row.names = F, quote = F)



# c = read.csv('./../data/thyroid_geneInfo.csv')
# t=c[which(c$gene_id %in% 'ENSG00000130203'),]


