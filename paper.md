---
title: 'sentropy: Similarity-Sensitive Entropy in Python'
tags:
  - sentropy
  - Shannon entropy
  - Simpson's index
  - metagenomics
  - immunomics
  - computational pathology
  - medical imaging
  - machine learning
  - Python
  - data science
  - diversity
authors:
  - name: Phuc Nguyen, PhD
    orcid: 0000-0001-9993-8434
    affiliation: 1
  - name: Rohit Arora, PhD
    orcid: 0000-0001-7128-6403
    affiliation: 1
  - name: Elliot D. Hill, MSc
  - orcid: 0009-0004-1987-3749
    affiliation: 2
  - name: Jasper Braun, PhD
    orcid: 0000-0003-1250-4399
    affiliation: 1
  - name: Alexandra Morgan
    orcid: 0000-0001-7787-0547
    affiliation: 1
  - name: Liza M. Quintana, MD
    orcid: 0000-0002-5043-7425
    affiliation: 1
  - name: Gabrielle Mazzoni
    affiliation: "1,3"
  - name: Ghee Rye Lee, MMSc, MM
    orcid: 0000-0001-6614-0223
    affiliation: "1,4"
  - name: Rima Arnaout, MD
    orcid: 0000-0002-7134-0040
    affiliation: 5
  - name: Ramy Arnaout, MD, DPhil
    orcid: 0000-0001-6955-9310
    affiliation: "1,6,7"
affiliations:
- name: Department of Pathology, Beth Israel Deaconess Medical Center, Boston, MA 02215, United States
  index: 1
- name: Department of Biostatistics and Bioinformatics and Duke AI Health, Duke University School of Medicine, Durham, NC 27710, USA
  index: 2
- name: University of Virginia School of Medicine, Charlottesville, VA 22903, USA
  index: 3
- name: Department of Surgery, Duke University School of Medicine, Durham, NC 27710, USA
  index: 4
- name: Department of Medicine, Bakar Institute for Computational Health Sciences, and Center for Intelligent Imaging, University of California San Francisco, San Francisco, CA 94143, USA
  index: 5
- name: Division of Clinical Informatics, Beth Israel Deaconess Medical Center, Boston, MA 02215, United States
  index: 6
- name: Harvard Medical School, Boston, MA 02115, USA
  index: 7
date: 26 June 2026
bibliography: paper.bib
---

# Summary

Entropy, the main measure of information content, uncertainty, diversity, and disorder across many fields, derives exclusively from elements' frequencies, ignoring the rich information encoded by elements' similarities and differences. Similarity-sensitive entropy or "sentropy" captures this missing information [@leinsterEntropyDiversityAxiomatic2020]. The `sentropy` Python package calculates traditional entropy's sentropic counterparts, including variants by Rényi order $\alpha$/Hill viewpoint parameter $q$ (the Rényi sentropies, with Shannon sentropy for $\alpha=q=1$), sentropic versions of Hill's diversity indicies [@hill1973diversity] (the similarity-sensitive effective/D-number forms of Shannon entropy, Simpson's index, the Berger-Parker index, etc.), and both the Leinster-Cobbold-Reeve (LCR) family of measures and the Vendi scores (quantum sentropies) [@nguyenComparingFrameworksSimilarityaware]. `sentropy` also calculates sentropic $\alpha$, $\beta$, and $\gamma$ diversities, relative sentropy (the sentropic generalization of Kullback-Leibler divergence), and cross sentropy. The package has 100% code coverage and is optimized for speed and large datasets.

# Statement of need
Shannon entropy and related "traditional" entropies measure the information encoded by the frequency distribution of a system's unique elements. Similarities among elements encode additional information, with applications in multiple fields:

- **Biological sciences:** immunomes (similarities among antibodies and among T-cell receptors) [@aroraRepertoirescaleMeasuresAntigen2022], microbiomes (microorganisms), viromes (viruses), transcriptomes (RNA transcripts), sequencing libraries (sequences), biomes (species)
- **Medical sciences:** imaging datasets (images) [@couch2024beyond], diseases, patient cohorts (patients), drug candidates, diets (foods)
- **Physical sciences:** materials, chemical compounds, minerals, geography and terrain
- **Engineering/industry:** machine learning (image/text/video datasets), supply chains, industries (e.g. for antitrust determination), investment porfolios
- **Social sciences:** populations, societies, political organizations, boards of directors, classrooms, committees (to optimize representation), sports teams (players), museums (exhibits/artwork)

The `sentropy` package is designed to calculate the full range of sentropic quantities in these and other contexts.

# State of the field
Because entropy and diversity are closely related, packages that calculate diversity provide some of `sentropy`'s functionality. Specifically, packages for calculating frequency- and similarity-sensitive $\alpha$ and $\beta$ diversities in effective-number form exist for the R (*rdiversity*) and Julia (*Diversity*) programming languages, but not to our knowledge for Python, which is more widely used, especially in machine learning. The Python package *cdiversity* calculates Hill-number diversity but without similarity. The *vendi-score* Python package calculates Vendi scores but not $\alpha/\beta/\gamma$ diversities, relative sentropies, etc., and only for the Vendi framework.


# Software design
`sentropy` is designed for speed (via vectorization and parallelization) and large datasets. Large datasets present a special challenge: because sentropic measures require calculating the similarity between each pair of unique species in the dataset, and because pairwise similarities are usually inputted as elements of an $n\times n$ matrix ($Z$), direct implementations risk running out of computer memory for large $n$. For example, assuming standard 8-byte floating point precision, a single dataset of a million unique elements would require 8TB of memory. Immunomes, microbiomes, and imaging datasets routinely contain tens or hundreds of thousands of unique elements, as do transcriptomes, cell atlases, and other complex datasets. Moreover, a single study can involve tens to hundreds of such datasets (e.g., one per sample/volunteer/patient) and thereby many millions of unique elements in all. *rdiversity* and *Diversity* both require the similarity matrix to be stored in memory, but for the applications above, such matrices are likely to be too large to store in working memory and may even be too large to write to disk. For this reason, `sentropy` instead also implements on-the-fly row-by-row calculations, to better handle such cases.

# Research impact statement

Since its initial release, `sentropy` (originally released as *greylock*) has been adopted in peer-reviewed scientific literature spanning computational biology and information theory. The package was employed and cited in *Nature Communications* [@ferreiraSelfsupervisedLearningLabelfree2025a] for self-supervised cardiac ultrasound segmentation, where it supported analysis of over 18,000 clinical echocardiograms. Additionally, it formed the computational backbone of a theoretical investigation into similarity-sensitive entropy metrics published in Physical Review E [@nguyenComparingFrameworksSimilarityaware], involving comparative analysis across 53 benchmark ML datasets. `sentropy` also has applications in data efficiency and data pruning, important in machine learning [@chinnENRICHingMedicalImaging2023].

# AI usage disclosure
The authors used GitHub Copilot for inline code suggestions. The authors reviewed and validated all AI-suggested edits and bear full responsibility for the final work.