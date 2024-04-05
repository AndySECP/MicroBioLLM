# MicroBioLLM

## Unraveling the secrets of the microbiome

The overarching goal here is to leverage language model methodologies, but adapted for the biological sequences domain and in this case, microbiology and microbiome data. This involves processing genetic data (e.g., DNA sequences) to classify or predict specific organisms' phylogenetic characteristics.

What is a microbiome?
The microbiome is the community of microorganisms (such as fungi, bacteria and viruses) that exists in a particular environment. In humans, the term is often used to describe the microorganisms that live in or on a particular part of the body, such as the skin or gastrointestinal tract. Microbiome can be very complex, like the soil microbiome. Microbiome are crucial for human health and understanding precisely their composition is crucial for a lot of applications.

To characterize the taxonomic and genomic composition of a microbiome, we can use whole metagenome sequencing which analyze the genetic material (DNA) of all microorganisms present in a particular environmental sample. Unlike traditional sequencing methods that target specific genes or organisms, whole metagenome sequencing provides a comprehensive view of the genetic diversity and functional potential of all microbes within a sample.

The process involves extracting DNA from the sample, which may contain bacteria, archaea, fungi, viruses, and other microorganisms. Next, the DNA is fragmented into smaller pieces and sequenced using high-throughput sequencing technologies, such as next-generation sequencing (NGS). The resulting sequence data are then analyzed bioinformatically to identify and characterize the microbial species present, as well as to predict their metabolic pathways and potential interactions within the ecosystem.

The traditional methods for identifying the organisms present organisms in a sample are using public database and align it against the DNA in the sample.

Whole metagenome sequencing enables researchers to gain insights into the composition, diversity, and functional capabilities of complex microbial communities, such as those found in the human gut, soil, oceans, and other environments. This technique has numerous applications in microbiology, ecology, environmental science, and human health, including the study of microbial ecology, microbial diversity, host-microbiome interactions, and the role of the microbiome in health and disease.

What are the challenges linked to microbiome analysis?
Whole metagenome sequencing poses several challenges, and one of the significant challenges is the presence of "dark matter." Dark matter refers to genetic material that remains undetected or poorly characterized using standard sequencing and bioinformatics techniques.

Dark matter may represent novel genetic material or microbial taxa that are not well-represented in existing reference databases, making it challenging to assign taxonomic or functional annotations.

This is where the application of AI presents a lot of potential: what if we could train a model to identify these species?

AI technologies have enabled a more efficient mining of microbial dark matter to generate a better understanding of microbial communities and their potential applications but they can be inefficient and not very comprehensive.

We want to use the Mistral model to build an efficient model for identifying unknown organisms in a sample.

## Methodology

This work presents an innovative approach to genomic data representation through the use of k-mers, fixed-length subsequences derived from longer DNA sequences, and outlines a comprehensive methodology for processing and tokenizing these k-mers for subsequent analysis with deep learning models.

## BUILDING THE DATABASE
We downloaded 500 genomes assemblies from a public database of genomes: Refseq (https://www.ncbi.nlm.nih.gov/refseq/about/prokaryotes/). We downloaded only complete genomes for now.

We started with only bacterial genomes from the 5 following most commonly sequenced families: Enterobacteriaceae, Bacillaceae, Pseudomonadaceae, Staphylococcaceae, Streptococcaceae. In reality microbiomes are much more complex, and contains thousands of organisms, and not only bacteria, but also fungi, archaea, virus.. The final goal would be to add more organism to this database to allow identification of less represented organisms.

## DATA REPRESENTATION USING K-MERS
The fundamental step in our approach involves segmenting genomic sequences into k-mers of a predefined length, specifically chosen to be eight nucleotides long. This segmentation strategy is predicated on the hypothesis that k-mers of this length offer an optimal balance between capturing the complexity of genomic sequences and maintaining computational efficiency. The process begins with the extraction of sequences from gzipped FASTA formatted files, ensuring that only valid nucleotide sequences are considered. Subsequent to the extraction, a counting mechanism enumerates the occurrence of each unique k-mer across the dataset, facilitating a quantitative assessment of sequence composition.

## FREQUENCY ANALYSIS AND DATAFRAME REPRESENTATION
Upon determining the frequency of each k-mer within the genomic sequences, the data is organized into a structured format suitable for analysis. This organization is accomplished by mapping k-mer frequencies to their corresponding taxa, thereby generating a comprehensive dataset that reflects the compositional diversity of the sequences studied. This dataset is then represented as a dataframe, offering a tabular view of the information that is amenable to further computational processes.

## TOKENIZATION AND DATASET PREPARATION
A pivotal aspect of our methodology involves the tokenization of k-mer sequences, a process that converts the sequences into a format recognizable by machine learning algorithms. By treating k-mers as analogous to words in natural language processing (NLP), we employ tokenization techniques to parse and encode the genomic data. This approach leverages pre-existing NLP infrastructures, specifically utilizing a tokenizer from a pre-trained model, to facilitate the adaptation of deep learning models to genomic data. The tokenized data is subsequently partitioned into training and testing sets, ensuring that the models are both robust and generalizable.

## MODELING
Upon dataset preparation, the Mistral 7B model was fine-tuned with the objective of optimizing its performance for sequence classification. The process employed a configuration with minimal intervention, key parameters included a learning rate of 2e-5, a training batch size per device of four, and a training duration spanning three epochs, among others.

The fine-tuning process was conducted on a dataset comprising 250 genomes, with a distinct subset of 50 genomes reserved for testing. The adapted model achieved an accuracy rate of 82% (9 misclassified) in classifying genomic sequences into organism families, supporting the hypothesis that species within the same family share characteristic k-mer signatures. This result, achieved with limited parameter optimization and limited data, illustrates the feasibility of applying deep learning models, developed for NLP, to the analysis of genomic sequences.

The findings from this hackathon provide a promising avenue for the application of deep learning in genomics, particularly through the adaptation of models developed for language processing. The results with the Mistral 7B model in classifying genomic sequences into organism families, despite minimal optimization, demonstrates the potential of this methodology. Future research will focus on expanding the dataset to encompass a wider diversity of genomes, enhancing the model's ability to classify not only at the family level but also at the species level, and potentially identifying novel species within known families. This direction aims to further the capabilities of deep learning in genomics, contributing to the advancement of bioinformatics and the understanding of genetic diversity and evolution.

## METHODOLOGICAL SIGNIFICANCE
This study's methodological contribution lies in its novel application of NLP techniques to the field of genomics, specifically through the use of k-mer based tokenization. By drawing parallels between genomic sequences and linguistic structures, we demonstrate the feasibility of adapting deep learning models originally designed for language processing to the analysis of DNA sequences. This interdisciplinary approach not only enriches the analytical toolkit available to bioinformaticians but also opens new avenues for the application of deep learning in genomics.

## FUTURE DIRECTIONS
Looking forward, this work lays the groundwork for further exploration into the optimization of k-mer lengths for various genomic applications, the potential for integrating more sophisticated deep learning models, and the scalability of the proposed methodology to accommodate the increasing volume of genomic data. Through continued refinement and adaptation, the approaches outlined herein hold the promise for significant advancements in the understanding and application of genomic data.