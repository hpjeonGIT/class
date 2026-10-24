# Bioinformatic; Learn Bulk RNA-Seq Data Analysis From Scratch
- Instructor

## Section 1: Course Introduction & Disclaimer

### 1. Course Introduction, Disclaimer & Important Message to Our Learners

## Section 2: Module-1: Basics of Molecular Biology (Optional)

### 2. What is DNA?
- A long polymer with 3 components

### 3. Where is DNA Located in Our Cells?
- In the nucleus of a cell

### 4. What is Role of DNA?
- Control of physical features
- How does DNA control phenotype?
  - DNA
    - Non-coding DNA
    - Coding DNA (Gene)
      - Protein: is structural and functional unit of our body

### 5. Difference Between Eukaryotic and Prokaryotic Genes
- Algorithms used for prokaryotic genes can not be used for eukaryotic genes
- Tandem arrangement of genes and no introns in Prokayotes
- Disperse arrangements of genes and lots of introns in Eukaryotes

### 6. What is Inside of Gene (Coding Regions of DNA)?
- What does coding DNA have?
  - ORF (Open Reading Frame)
  - Genetic code is specific combination of three necleotide
  - Degeneracy
  - Exons and introns
    - DNA: Exon+Intron+Exon+Intron+Exon+Intron...
- What else coding DNA have?
  - Promoter Sequences + Coding Sequences + Termination Sequences
  - Pseudo Gene + Termination Sequences
- Some more features
  - Start code: AUG
  - Stop code: UAA
  - Transcription start site
  - Transcription stop site

### 7. Post Transcriptional Modifications
- Nascent mRNA: EXON1+INTRON1+EXON2+INTRON2+EXON3

## Section 3: Introduction of RNA-Seq

### 8. Why There is Need of RNA-Seq Analysis
- Differential expression of genes
  - We perform RNA-Seq to Study Differential Expression of Genes
  - Ex: Why normal cells become cancer cells?

### 9. Basic Workflow of RNA-Seq Analysis
- Differential expression of genes
  - Normal cell vs cancer cell
  - Total RNA -> Reverse transcriptaes -> cDNA

### 10. Next Generation Sequencing Workflow
- Sanger Sequencing
  - One DNA sequence at a time
- Basic overview of next generation sequencing workflow
    1. DNA extraction
    2. Library preparation
    3. Sequencing: millions of sequence at a time
    4. Analysis: FastQ format -> Align reads -> BAM -> Idnetify variants -> VCF

### 11. Basic File Obtained During RNA-Seq Analysis
- FASTQ file -> SAM/BAM files -> Feature count matrix -> DEGs
- FASTQ format
  - Header
  - Sequence
  - Quality score: 0-40 Q score map to symbols
- SAM/BAM file format
  - Header section
  - Alignment section
    - QNAME
    - FLAG
    - RNAME
    - POS
    - MAPQ
    - CIGAR
    - RNEXT
    - PNEXT
    - TLEN
    - SEQ
    - QUAL
    - Optional fields
- Reads: each sequence fragment in FASTQ file 

## Section 4: Practical Demonstration of RNA-Seq Reads to Feature Count Matrix in Linux

### 12. Basic Workflow of RNA-Seq Data Analysis
- RNA-Seq Data analysis
  - FASTQ files (containing Reads) -> Feature count matrix (containing counts of Reads) -> Rstudio

### 13. Installation of Linux Part-1

### 14. Installation of Linux Part-2

### 15. Fundamentals of Linux Operating System

### 16. Installation of Necessary Programs In Linux Environment (Part-1)
- fastQC: for quality check of the raw reads
- Trimmomatic: for trimming of the reads
  - http://www.usadellab.org/cms/?page=trimmomatic
  - https://github.com/usadellab/Trimmomatic
  - Download a binary for quick use - this is a java app
- HISAT2: for the alignment of the reads with reference genome
- Subread: for building of count matrix for DEG analysis
- sudo apt-get -y install fastqc hisat2 subread

### 17. Installation of Necessary Programs In Linux Environment (Part-2)

### 18. Installation of SAM Tools in Linux (Part-3)
- sudo apt install samtools

### 19. Downloading of Timmomatic Tool

### 20. Quality Check of the Reads with FASTQC (Part-1)
```bash
$ fastqc test_udemy.fastq 
null
Started analysis of test_udemy.fastq
Analysis complete for test_udemy.fastq
```
- The above command produces test_udemy_fastqc.html

<img src="./chap20_snapshot.png" height="300">

### 21. Quality Check of the Reads with FASTQC (Part-2)
- Basic statistics
- Per base sequence quality
  - Green: good quality
  - Yellow: intermediate
  - Red: Poor quality
- Per tile sequence quality
- Per sequence quality scores
  - Histogram view
  - For average quality score, the reads is the value in the X-axis (37 in the test_dumy.fastq)
- Per base sequence content
- Per sequence GC content
- Per base N content: high N number may imply bad quality
- Sequence length distribution
  - Histogram of sequence length
- Sequence duplication levels
- Overrepresented sequences
- Adapter content

### 22. Use of Timmomatic Tool to Remove Poor Quality Reads
- Unzip Trimmomatic-0.41.zip
- May need the latest java runtime
```bash
$ java -jar trimmomatic-0.41.jar SE -threads 4 test_udemy.fastq test_trimmed.fastq TRAILING:10 -phred33
TrimmomaticSE: Started with arguments:
 -threads 4 test_udemy.fastq test_trimmed.fastq TRAILING:10 -phred33
Input Reads: 305 Surviving: 305 (100.00%) Dropped: 0 (0.00%)
TrimmomaticSE: Completed successfully
```
  - SE for Single-End sequencing
  - PE for Paired-End sequencing (more expensive)
  - -phred33: quality score of 33
  - In this test, 0 Reads dropped
- Now run fastqc to generate html
  - Confirm that the quality has been improved

<img src="./chap22_before.png" height="300">

<img src="./chap22_after.png" height="300">

### 23. Use of HISAT2 for Alignment of Reads with Reference Genome
- Reference genome from hisat2 site: 
  - https://daehwankimlab.github.io/hisat2/download/
  - Download https://genome-idx.s3.amazonaws.com/hisat/grch38_genome.tar.gz
    - 3GB
```bash
$ hisat2 -q --rna-strandness R -x grch38/genome -U test_trimmed.fastq | samtools sort -o test_trimmed.bam
305 reads; of these:
  305 (100.00%) were unpaired; of these:
    14 (4.59%) aligned 0 times
    269 (88.20%) aligned exactly 1 time
    22 (7.21%) aligned >1 times
95.41% overall alignment rate
```

### 24. If Hisat2 Gives You Error Then Try This
- When memory size is not large enough
  - Increase swap memory size
  - sudo fallocate -l 8G /swapfile # will create a swap file
  - sudo chmod 600 /swapfile
  - sudo mkswap /swapfile
  - sudo swapon /swapfile
  - free -h

### 25. Building the Hisat2 Compatible Genome From FASTA
- https://www.ensembl.org
- Search Atlantic Salmon then click FASTA
  - Select GCA_021399835.1 and then click Get Files -> FASTA in Download
  - In the FTP menu, download unmasked.fa.bgz
  - Unzip using bgzip (sudo apt install tabix)
  - hisat2-build XXX.fa genome_index

### 26. Downloading of GTF File to Build the Feature Count Matrix
- Gene Transfer Format (GTF)
  - Col 1-9
- From ensemble.org, find homo sapiens and download gtf file

### 27. Building of Feature Count Matrix With Subread Tool
- featureCounts -S 1 -a Homo_sapiens.gtf -o test_featurecounts.txt test_trimmed.bam
```bash
$ featureCounts -S 1 -a genes.gtf -o test_featurecounts.txt test_trimmed.bam 
The "-S" option has been depreciated.

        ==========     _____ _    _ ____  _____  ______          _____  
        =====         / ____| |  | |  _ \|  __ \|  ____|   /\   |  __ \ 
          =====      | (___ | |  | | |_) | |__) | |__     /  \  | |  | |
            ====      \___ \| |  | |  _ <|  _  /|  __|   / /\ \ | |  | |
              ====    ____) | |__| | |_) | | \ \| |____ / ____ \| |__| |
        ==========   |_____/ \____/|____/|_|  \_\______/_/    \_\_____/
	  v2.0.6

//========================== featureCounts setting ===========================\\
||                                                                            ||
||             Input files : 1 BAM file                                       ||
||                                                                            ||
||                           test_trimmed.bam                                 ||
||                                                                            ||
||             Output file : test_featurecounts.txt                           ||
||                 Summary : test_featurecounts.txt.summary                   ||
||              Paired-end : no                                               ||
||        Count read pairs : no                                               ||
||              Annotation : genes.gtf (GTF)                                  ||
||      Dir for temp files : ./                                               ||
||                                                                            ||
||                 Threads : 1                                                ||
||                   Level : meta-feature level                               ||
||      Multimapping reads : not counted                                      ||
|| Multi-overlapping reads : not counted                                      ||
||   Min overlapping bases : 1                                                ||
||                                                                            ||
\\============================================================================//

//================================= Running ==================================\\
||                                                                            ||
|| Load annotation file genes.gtf ...                                         ||
||    Features : 5087789                                                      ||
||    Meta-features : 78941                                                   ||
||    Chromosomes/contigs : 70                                                ||
||                                                                            ||
|| Process BAM file test_trimmed.bam...                                       ||
||    Single-end reads are included.                                          ||
||    Total alignments : 357                                                  ||
||    Successfully assigned alignments : 211 (59.1%)                          ||
||    Running time : 0.00 minutes                                             ||
||                                                                            ||
|| Write the final count table.                                               ||
|| Write the read assignment summary.                                         ||
||                                                                            ||
|| Summary of counting results can be found in file "test_featurecounts.txt.  ||
|| summary"                                                                   ||
||                                                                            ||
\\============================================================================//
```
- Load featurecounts.txt in a spreadsheet
  - Delimited by space
  - Libre office failed to load due to the limit of characters in a cell

### 28. How to Process Multipipe FastQ Files Using Bash Scripts
- fastqc *.fastq
- script1.sh
```sh
#!/bin/bash
# Set the path to the input directory as the current directory
input_dir=$(pwd)
# Loop over all fq.gz files in the input directory
for file in "$input_dir"/*.fastq; do
    # Extract the filename without the extension
    filename=$(basename "$file" .fastq)
    # Run the trimmomatic command on the file
    java -jar trimmomatic-0.39.jar SE -threads 32 "$file" "$input_dir/$filename"_trimmed.fastq TRAILING:10 -phred33 
done
```

### 29. Experimental Design of Airway Cell Line Study That will Use In DEG Analysis
- Ref: https://doi.org/10.1371/journal.pone.0099625
  - Title: RNA-Seq Transcriptome Profiling Identifies CRISPLD2 as a Glucocorticoid Responsive Gene that Modulates Cytokine Function in Airway Smooth Muscle Cells

## Section 5: Basic Concepts of R and R-STudio

### 30. Introduction of the Section

### 31. Installation of R and R-Studio
- When installing R from a source code, use `--enable-R-shlib` for R-studio
  - If libR.so is not found, R-studio will not work
 ./configure --prefix=/home/hpjeon/sw_local/R/4.6.0  --enable-R-shlib 

### 32. Setting Working Directory in R-Studio
- In Rstudio:

<img src="./ch32_cwd.png" height="200">

- Or `setwd("~/XXX/ZZZ/YYY")`

### 33. Basic Data Types Used in R
- Vector
- Matrix
- DataFRame
- List

### 34. Creating a Variable
- Variable provides a named storage that our program can manipulate

### 35. What is Package & Function in R?
- A function is a block of code which only runs when it is called
- Package contains many related functions
  - ggplot
  - Tidyverse
  - DESeq
  - `install.packages("ggplot")`

### 36. Brief Introduction of Bioconductor
- 
```R
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c("GenomicFeatures")
#Installation of DESeq2                     
BiocManager::install("DESeq2")
```

## Section 6: Differential Expression of Gene Analysis in R Using DESeq2 Package

### 37. Installation of DESeq2 in R-Studio For DEGs Analysis

### 38. What is CSV format & Saving MetaData File in CSV format

### 39. Uploading of Feature Count Matrix and Meta Data in R-Studio
- counts.csv and metadata2.csv # files are from Assignment 6
```R
> cnt <- read.csv("counts.csv")
> str(cnt) # shows the content of the dataframe
'data.frame':	64102 obs. of  8 variables:
 $ SRR1039508: int  679 0 467 260 60 0 3251 1433 519 394 ...
 $ SRR1039509: int  448 0 515 211 55 0 3679 1062 380 236 ...
 $ SRR1039512: int  873 0 621 263 40 2 6177 1733 595 464 ...
 $ SRR1039513: int  408 0 365 164 35 0 4252 881 493 175 ...
 $ SRR1039516: int  1138 0 587 245 78 1 6721 1424 820 658 ...
 $ SRR1039517: int  1047 0 799 331 63 0 11027 1439 714 584 ...
 $ SRR1039520: int  770 0 417 233 76 0 5176 1359 696 360 ...
 $ SRR1039521: int  572 0 508 229 60 0 7995 1109 704 269 ...
> met <- read.csv("metadata2.csv")
> str(met)
'data.frame':	8 obs. of  2 variables:
 $ cellLine     : chr  "N61311" "N61311" "N052611" "N052611" ...
 $ dexamethasone: chr  "untreated" "treated" "untreated" "treated" ...
```

### 40. Basic Quality Check of Feature Count Matrix and Meta Data
- Make sure the row name sin colData of cnt matches the row names of met
  - `all(colnames(cnt)%in% rownames(met))` must be TRUE
- Check the order of row names and column names
  - `all(colnames(cnt) == rownames(met))` must be TRUE

### 41. Use of DESeq2 for DEG Analysis (Part-1)
- Building DESeq data set
- Differential Analysis of Gene
```R
> library(DESeq2)
> dds <- DESeqDataSetFromMatrix(countData = cnt, 
                              colData = met,
                              design = ~dexamethasone)
> dds
class: DESeqDataSet 
dim: 64102 8 
metadata(1): version
assays(1): counts
rownames(64102): ENSG00000000003 ENSG00000000005 ... LRG_98 LRG_99
rowData names(0):
colnames(8): SRR1039508 SRR1039509 ... SRR1039520 SRR1039521
colData names(2): cellLine dexamethasone
```

### 42. DESeq2: Concept of Leaky Expression (Part-2)
- Differential expression of Genes
- Leaky expression of genes

### 43. DESEq2: Removing Low Counts Reads Genes (Part-3)

### Assignment 9: Assignment-8: Dropping Rows with Low Count

### 44. DESeq2: Use of DESeq2 Function for DEG Analysis (Part-3)

### Assignment 10: Assignment-9: Use of DESeq Function

### 45. What is Size Factor Estimation in DESEq2 ?

### 46. What is dispersion Estimation in DESeq2?

### 47. Hypothesis testing in DESeq2 for DEG Analysis

### 48. Concept of P-value and P-Adjusted values

### 49. Getting Differentially Expressed Gene at Different Alpha Value

### Assignment 11: Assignment-10: Getting DEGs at 0.05 Alpha Value

### 50. Converting Gene IDs to Gene Name

### Assignment 12: Assignment-11: Converting Genes IDs to Gene Name

## Section 7: Quality Checking of RNA-Seq Data

### 51. Basic Quality Check Parameters
### 52. Basic Concepts of PCA Plot
### 53. Building PCA Plot of RNA-Seq Data
### Assignment 13: Assignment-12: Generation of PCA Plot
### 54. Size Factor Estimation and Its Calculation
### Assignment 14: Assignment-13: Estimating Size Factor
### 55. Dispersion Estimates and Building of Dispresion Plot
### Assignment 15: Assignment-14: Building Dispersion Plot

### 56. Basic Understanding of Tidyverse & ggplo2
### 57. Installation of Tidyverse & ggplot2 and Sample Dataset
### 58. Basic Functionality of Tidyverse Functions; Filter, Arrange, and Mutate
### 59. Basic Functionality of ggplot2 to Build the Plots
### 60. Building MA Plot
### Assignment 16: Assignment-15: Building MA Plot for DEGs
### 61. Getting Idea About Best Genes
### Assignment 17: Assignment-16: Extraction of top 30 Best Genes
### 62. Building Volcano Plot-Part1
### 63. Building Volcano Plot -Part2
### Assignment 18: Assignment-17: Volcano Plot of Data
### 64. Building HeatMap of DEGs
### Assignment 19: Assignment-18: HeatMap of Best 30 DEGs
### 65. Simple Gene Ontology and Pathway Analysis of Genes
11min

