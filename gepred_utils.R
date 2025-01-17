# # sudo apt-get install libcurl4-openssl-dev libxml2-dev libssl-dev
# # source("https://bioconductor.org/biocLite.R")
# # biocLite("biomaRt")
# # biocLite("BSgenome")
# # biocLite('BSgenome.Hsapiens.UCSC.hg19')
# 
library(biomaRt)
library(BSgenome)
library(BSgenome.Hsapiens.UCSC.hg19)
# 
# # Get genetic information -------------------------------------------------------------------------------------------------------------------------------------------------------
  # Get 5k Upstream Gene Sequences
  fetchGeneSeq <- function(df) {
    mart <- useMart(biomart = "ENSEMBL_MART_ENSEMBL", dataset="hsapiens_gene_ensembl", host="grch37.ensembl.org")

    upstream_info=getBM(attributes=c('gene_flank','start_position','end_position','chromosome_name','strand','ensembl_gene_id'),filters=c('ensembl_gene_id','upstream_flank'),values=list(df$gene_id, 5000), mart=mart, checkFilters=FALSE)
    upstream_info[,c('strand', 'chromosome_name', 'start_position', 'end_position')] <- NULL
    colnames(upstream_info) <- c('upstream_sequence', 'gene_id')

    downstream_info=getBM(attributes=c('gene_flank','start_position','end_position','chromosome_name','strand','ensembl_gene_id'),filters=c('ensembl_gene_id','downstream_flank'),values=list(df$gene_id, 5000), mart=mart, checkFilters=FALSE)
    downstream_info[,c('strand', 'chromosome_name', 'start_position', 'end_position')] <- NULL
    colnames(downstream_info) <- c('downstream_sequence', 'gene_id')

    flanking_info <- merge(upstream_info, downstream_info, by = 'gene_id')
    df <- merge(df, flanking_info, by = 'gene_id')
  }

  # Get all Genes of a given chromosome
  getAllChromosomeGenes <- function(option, fn = NULL, chrom_num = 'All') {
    if (option == 1) {
      chr_genes <- read.table(fn, header = T)
    }
    else {
      grch37 <- useEnsembl(biomart="ensembl",GRCh=37, dataset="hsapiens_gene_ensembl")
      if (chrom_num == 'All'){
        chr_genes <- getBM(attributes=c('ensembl_gene_id','hgnc_symbol','entrezgene','chromosome_name','start_position','end_position', 'strand'), mart = grch37)
      } else {
        chr_genes <- getBM(attributes=c('ensembl_gene_id', 'hgnc_symbol','entrezgene','chromosome_name','start_position','end_position', 'strand'),
                           filters = c('chromosome_name'), values =c(toString(chrom_num)), mart = grch37)
      }

      chr_genes <- unique(chr_genes)

      # Remove genes wthich has no hgnc symbol or entrez gene
      chr_genes <- chr_genes[!(is.na(chr_genes$hgnc_symbol) | chr_genes$hgnc_symbol==""), ]
      chr_genes <- chr_genes[!(is.na(chr_genes$entrezgene) | chr_genes$entrezgene==""), ]

      # Remove genes which has duplicat ensembl gene id and hgnc symbol
      chr_genes <- chr_genes[row.names(unique(chr_genes[,c("ensembl_gene_id", "hgnc_symbol")])),]
    }


    return(unique(chr_genes))
  }

  #Fetch and combine gene info, sequences, and gene expression data
  getGeneticDetails <- function(gex_fileNm) {
    # Read the gene information file else fetch from the avaialble ensembl database
    gene_info <- getAllChromosomeGenes(option = 2);

    # Read gene expression data
    fn = read.table(gex_fileNm, header = T)
    gex_info <- fn[grep("ENSG",fn$gene_id),]
    gex_info$gene_id <- gsub("\\.[0-9]+", "", gex_info$gene_id)
    length(unique(gex_info$gene_id))

    # combine gene info and gene expression info based on ensenmbl id
    common_ensg <- intersect(gene_info$ensembl_gene_id, gex_info$gene_id)

    # process rna-seq data
    data <- gex_info[which(gex_info$gene_id %in% common_ensg),]
    t <- unique(gene_info[which(gene_info$ensembl_gene_id %in% common_ensg),  c('ensembl_gene_id', 'chromosome_name', 'start_position', 'end_position', 'strand')])
    colnames(t) <- c('gene_id', 'chromosome_name', 'start_position', 'end_position', 'strand')
    data <- merge(t, data, by = 'gene_id')

    geneExpData <- data.frame()
    for (i in 1 : length(common_ensg)) {
      dat = data[which(data$gene_id == common_ensg[i]), ]
      geneExpData <- rbind(geneExpData, dat[1, c('gene_id', 'chromosome_name', 'start_position', 'end_position', 'strand', 'TPM', 'FPKM')])
    }

    # Processing data: Appending chr string to each chromosome name and changing the strand from 1/-1 to +/-
    geneExpData$chromosome_name <- sub("^", "chr", geneExpData$chromosome_name)
    geneExpData[which(geneExpData$strand == -1), ]$strand <- '-'
    geneExpData[which(geneExpData$strand == 1), ]$strand <- '+'

    # Get all the gene sequences --------------------------------------------------------------------------------------------------------------------------------------------------
    gex_seq_info <- unique(fetchGeneSeq(geneExpData))
    
    # Annotate the upstream and downsream seq start and end positions

    # Positive Strand Upstream
    pos_strand_dat = gex_seq_info[which(gex_seq_info$strand == '+'), ]
    pos_strand_dat$upstream_start <- (pos_strand_dat$start_position - 5000)
    pos_strand_dat$upstream_end <- (pos_strand_dat$start_position - 1)
    
    # Positive Strand Downstream
    pos_strand_dat$downstream_start <- (pos_strand_dat$end_position + 1)
    pos_strand_dat$downstream_end <- (pos_strand_dat$end_position + 5000)
    
    # Negative Strand Upstream
    neg_strand_dat = gex_seq_info[which(gex_seq_info$strand == '-'), ]
    neg_strand_dat$upstream_start <- (neg_strand_dat$end_position + 1)
    neg_strand_dat$upstream_end <- (neg_strand_dat$end_position + 5000)
    
    # Negative Strand Upstream
    neg_strand_dat$downstream_start <- (neg_strand_dat$start_position - 5000)
    neg_strand_dat$downstream_end <- (neg_strand_dat$start_position - 1)
    
    gex_seq_info <- rbind(pos_strand_dat, neg_strand_dat)

    return(gex_seq_info)

  }

# Get epigenetic information ----------------------------------------------------------------------------------------------------------------------------------------------------
  # Create huge band of epdigenetic info peak information from start to last location
  getBandInfo <- function(gepi_data, max_end_chrom) {
    peakInfo = rep(0, max_end_chrom)

    # First pass: Put all ones to the places which has dnase
    for (i in 1:nrow(gepi_data)) {
      t = gepi_data[i,]
      peakInfo[t$chromStart:t$chromEnd] = 1
    }

    # Second pass: Mark all the dnase peaks with 2
    for (i in 1:nrow(gepi_data)) {
      t = gepi_data[i,]
      peakInfo[t$chromStart + t$peak] = 2
    }
    return(peakInfo)
  }

  # Read epigenetic files 
  readEpigeneticFiles <- function(gepi_fileNm) {
    # If file read just one file, else read multiple files and append
    if (file_test("-f", gepi_fileNm)) {
      gepi_data = as.data.frame(read.table(gepi_fileNm))
      colnames(gepi_data) = c('chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak')
    } else {
      file_list <- list.files(gepi_fileNm)
      
      for(i in 1 :length(file_list)) {
        data = as.data.frame(read.table(paste0(gepi_fileNm, file_list[i])))
        data$epigen <- strsplit(file_list[i], '\\.')[[1]][1]
        if (i == 1) {
          gepi_data <- data
        } else {
          gepi_data <- rbind(gepi_data, data)
        }
      }
      
      colnames(gepi_data) = c('chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak', 'epigen')
    }
    return(gepi_data)
  }
  
  # Get epigenetic info for a given data
  getEpigeneticDetails <- function(gepi_data) {
    unq_epigen <- unique(gepi_data$epigen)
    gepi_peak_data <- matrix(0, length(unq_epigen), max(gepi_data$chromEnd))
    for (i in 1:length(unq_epigen)) {
      dat <- gepi_data[which(gepi_data$epigen %in% unq_epigen[i]), ]
      gepi_peak_data[i, ] = getBandInfo(dat, max(gepi_data$chromEnd))
    }
    return(gepi_peak_data)
  }
   
# String Utils -----------------------------------------------------------------------------------------------------------------------------------------------------
  strReverse <- function(x) {
    x = sapply(lapply(strsplit(x, NULL), rev), paste, collapse="")
    return(x)
  }
  

# Merge genetic and epigenetic information ----------------------------------------------------------------------------------------------------------------------
  mergeGeneEpigeneInfo <- function(gex_info, gepi_info, chrom, stream) {
    addEpigeneTracks <- function(gene_info, epigene_info) {
      if (max(gene_info$downstream_end) > dim(epigene_info)[2]) {
        epigene_info <- cbind(epigene_info, matrix(0, dim(epigene_info)[1], (max(gene_info$downstream_end) - dim(epigene_info)[2])))
      }
      
      if (max(gene_info$upstream_end) > dim(epigene_info)[2]) {
        epigene_info <- cbind(epigene_info, matrix(0, dim(epigene_info)[1], (max(gene_info$upstream_end) - dim(epigene_info)[2])))
      }
      return(epigene_info)
      
    }
    
    mergeTracks <- function(gene_info, gepi_info, stream) {
      gene_epigene <- data.frame()
      
      for (i in 1:nrow(gene_info)) {
        dat <- gene_info[i,]
        
        for (j in 1:nrow(gepi_info)){
          upstr <- toString(gepi_info[j, dat$upstream_start:dat$upstream_end])
          upstr <- gsub(", ", "", upstr, ignore.case = TRUE)
          
          downstr <- toString(gepi_info[j, dat$downstream_start:dat$downstream_end])
          downstr <- gsub(", ", "", downstr, ignore.case = TRUE)
          
          if(dat[,'strand'] == "-"){
            upstr = strReverse(upstr)
            downstr = strReverse(downstr)
          }
          
          if (stream == "up") {
            dat[,paste0('chipseq', j)] <- upstr # Upstream
          } else if (stream == "down") {
            dat[,paste0('chipseq', j)] <- downstr # Downstream
          } else {
            dat[,paste0('chipseq', j)] <- paste0(upstr, downstr) # Both
          }
        }
        
        if (stream == "up") {
          dat$sequence <- dat$upstream_sequence # Upstream
        } else if (stream == "down") {
          dat$sequence <- dat$downstream_sequence # Downstream
        } else {
          dat$sequence <- paste0(dat$upstream_sequence, dat$downstream_sequence) # Both
        }
        
        gene_epigene <- rbind(gene_epigene, dat)
      }
      return(gene_epigene)
    }
  
    if (chrom == 'All') {
      chrom = unique(gex_info$chromosome_name)
      gex_gepi = data.frame()
      for (i in 1:length(chrom)) {
        gepi_indx = which(gepi_info$chrom == chrom[i])
        gex_indx = which(gex_info$chromosome_name == chrom[i])
        
        if (length(gex_indx) > 0 && length(gepi_indx) > 0) {
          print('yes')
          gex_dat = gex_info[gex_indx,]
          gepi_dat = getEpigeneticDetails(gepi_info[gepi_indx,])
          
          gepi_dat = addEpigeneTracks(gex_dat, gepi_dat)
          
          gex_gepi = rbind(gex_gepi, mergeTracks(gex_dat, gepi_dat, stream))
        }
        print(paste0(chrom[i], 'complete'))
        print(nrow(gex_gepi))
      }
      return(gex_gepi)
    } 
    else {
      gex_gepi = data.frame()
      chrom = strsplit(chrom, ',')[[1]]
      print(chrom)
      for (i in 1:length(chrom)) {
        chr = paste0('chr',chrom[i])
        gepi_indx = which(gepi_info$chrom == chr)
        gex_indx = which(gex_info$chromosome_name == chr)
        
        if (length(gex_indx) > 0 && length(gepi_indx) > 0) {
          print('yes')
          gex_dat = gex_info[gex_indx,]
          gepi_dat = getEpigeneticDetails(gepi_info[gepi_indx,])
          
          gepi_dat = addEpigeneTracks(gex_dat, gepi_dat)
          
          gex_gepi = rbind(gex_gepi, mergeTracks(gex_dat, gepi_dat, stream))
        }
        print(paste0(chrom[i], 'complete'))
        print(nrow(gex_gepi))
        
      }
      return(gex_gepi)
    }
  }
  

  # Fetch both genetic and epigenetic information ----------------------------------------------------------------------------------------------------------------------
  # Read epigenetic information
  gepi_fileNm <- './data/Brain/epigenetics/'
  gepi_info <- readEpigeneticFiles(gepi_fileNm)
  write.table(gepi_info, file = './data/Brain/brain_epigeneInfo.csv', sep = ',', row.names = F, col.names = F, quote = F)
  # 
  # # Genetic information
  gex_info <- getGeneticDetails('./data/Brain/genetics/ENCFF570KQR.tsv')
  write.table(gex_info, file = './data/Brain/brain_geneInfo.csv', sep = ',', row.names = F, col.names = F, quote = F)
  
  gex_gepi <- mergeGeneEpigeneInfo(gex_info, gepi_info, 'All', 'both')
  
  
  
  # t <- gex_info
  # gex_gepi <- data.frame()
  # err_indx <- c()
  # for (i in 1:nrow(t)) {
  #   print(i)
  # 
  #   dat <- t[i, ]
  #   if(dat$upstream_start < 0 || dat$downstream_start < 0){
  #     print(i)
  #     next
  #   }
  #   hasErr <- FALSE
  #   for (j in 1: 6) {
  #     tryCatch(
  #       if (dat[,'strand'] == "+"){
  # 
  #         # dat[,paste0('chipseq', j)] <- gsub(', ', '', toString(gepi_info[j, dat$upstream_start:dat$upstream_end])) # Upstream
  #        dat[,paste0('chipseq', j)] <- gsub(', ', '', toString(gepi_info[j, dat$downstream_start:dat$downstream_end])) # Downstream
  #         dat[,paste0('chipseq', j)] <- paste0(gsub(', ', '', toString(gepi_info[j, dat$upstream_start:dat$upstream_end])),
  #                                            gsub(', ', '', toString(gepi_info[j, dat$downstream_start:dat$downstream_end]))) # Both
  #       } else {
  #         
  #         upstr <- toString(gepi_info[j, dat$upstream_start:dat$upstream_end])
  #         upstr <- gsub(", ", "", upstr, ignore.case = TRUE)
  # 
  #         # downstr <- toString(gepi_info[j, dat$downstream_start:dat$downstream_end])
  #         # downstr <- gsub(", ", "", downstr, ignore.case = TRUE)
  # 
  #         dat[,paste0('chipseq', j)] <- strReverse(upstr) # Upstream
  #         # dat[,paste0('chipseq', j)] <- strReverse(downstr) # Downstream
  #         # dat[,paste0('chipseq', j)] <- paste0(strReverse(upstr), strReverse(downstr)) # Both
  # 
  #       }
  #       , error = function(c) {
  #         err_indx <- c(err_indx, i)
  #         hasErr <- TRUE
  #       }
  #     )
  #   }
  #   if (!hasErr) {
  #     # dat$sequence <- paste0(dat$upstream_sequence, dat$downstream_sequence)
  #     # dat$sequence <- dat$upstream_sequence
  #     dat$sequence <- dat$downstream_sequence
  #     gex_gepi <- rbind(gex_gepi, dat)
  #   }
  # }

  # write.table(gex_gepi, file = './data/Brain/brain_chr19_upstream.csv', sep = '\t', row.names = F, quote = F)
  write.table(gex_gepi, file = './data/tissue/brain_chr_all.csv', sep = '\t', row.names = F, quote = F)
  # write.table(gex_gepi, file = './data/Brain/brain_chr19.csv', sep = '\t', row.names = F, quote = F)
  
  # write.table(gex_gepi, file = './data/Brain/brain_upstream.csv', sep = '\t', row.names = F, quote = F)
  
  
  