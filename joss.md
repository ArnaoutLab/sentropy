# sentropy: Similarity-Sensitive Entropy in Python

## Summary

Entropy, the main measure of information content, uncertainty, diversity, and disorder across many fields, derives exclusively from elements' frequencies, ignoring the rich information encoded by elements' similarities and differences. Similarity-sensitive entropy or "sentropy" captures this missing information. The *sentropy* Python package calculates traditional entropy's sentropic counterparts, including variants by Rényi order $\alpha$/Hill viewpoint parameter $q$ (the Rényi sentropies, with Shannon sentropy for $\alpha=q=1$), sentropic versions of Hill's diversity indicies (the similarity-sensitive effective/D-number forms of Shannon entropy, Simpson's index, the Berger-Parker index, etc.), and both the Leinster-Cobbold-Reeve (LCR) family of measures and the Vendi scores (quantum sentropies). *sentropy* also calculates sentropic $\alpha$, $\beta$, and $\gamma$ diversities, relative sentropy (the sentropic generalization of Kullback-Leibler divergence), and cross sentropy. The package has 100% code coverage and is optimized for speed and large datasets.

## Statement of need
Shannon entropy and related "traditional" entropies measure the information encoded by the frequency distribution of a system's unique elements. Similarities among elements encode additional information, with applications in multiple fields:

- **Biological sciences:** immunomes (similarities among antibodies and among T-cell receptors), microbiomes (microorganisms), viromes (viruses), transcriptomes (RNA transcripts), sequencing libraries (sequences), biomes (species)
- **Medical sciences:** imaging datasets (images), diseases, patient cohorts (patients), drug candidates, diets (foods)
- **Physical sciences:** materials, chemical compounds, minerals, geography and terrain
- **Engineering/industry:** machine learning (image/text/video datasets), supply chains, industries (e.g. for antitrust determination), investment porfolios
- **Social sciences:** populations, societies, political organizations, boards of directors, classrooms, committees (to optimize representation), sports teams (players), museums (exhibits/artwork)

The *sentropy* package is designed to calculate the full range of sentropic quantities in these and other contexts.

## State of the field
Because entropy and diversity are closely related, packages that calculate diversity provide some of *sentropy*'s functionality. Specifically, packages for calculating frequency- and similarity-sensitive $\alpha$ and $\beta$ diversities in effective-number form exist for the R (*rdiversity*) and Julia (*Diversity*) programming languages, but not to our knowledge for Python, which is more widely used, especially in machine learning. The Python package *cdiversity* calculates Hill-number diversity but without similarity. The *vendi-score* Python package calculates Vendi scores but not $\alpha/\beta/\gamma$ diversities, relative sentropies, etc., and only for the Vendi framework.


## Software design
*sentropy* is designed for speed (via vectorization and parallelization) and large datasets. Large datasets present a special challenge: because sentropic measures require calculating the similarity between each pair of unique species in the dataset, and because pairwise similarities are usually inputted as elements of an $n\times n$ matrix ($Z$), direct implementations risk running out of computer memory for large $n$. For example, assuming standard 8-byte floating point precision, a single dataset of a million unique elements would require 8TB of memory. Immunomes, microbiomes, and imaging datasets routinely contain tens or hundreds of thousands of unique elements, as do transcriptomes, cell atlases, and other complex datasets. Moreover, a single study can involve tens to hundreds of such datasets (e.g., one per sample/volunteer/patient) and thereby many millions of unique elements in all. *rdiversity* and *Diversity* both require the similarity matrix to be stored in memory, but for the applications above, such matrices are likely to be too large to store in working memory and may even be too large to write to disk. For this reason, *sentropy* instead also implements on-the-fly row-by-row calculations, to better handle such cases.

## Research impact statement

Since its initial release, *sentropy* (originally released as *greylock*) has been adopted in peer-reviewed scientific literature spanning computational biology and information theory. The package was employed and cited in *Nature Communications* (Ferreira et al., *Nat Commun* 16:4070, 2025) for self-supervised cardiac ultrasound segmentation, where it supported analysis of over 18,000 clinical echocardiograms. Additionally, it formed the computational backbone of a theoretical investigation into similarity-sensitive entropy metrics published in Physical Review E (Nguyen et al., *Phys Rev E* 113:055305, 2026), involving comparative analysis across 53 benchmark ML datasets. *sentropy* also has applications in data efficiency and data pruning, important in machine learning (e.g. Chinn et al. *JAMIA* 2023, 30:1079, 2023).

## AI usage disclosure
The authors used GitHub Copilot for inline code suggestions. The authors reviewed and validated all AI-suggested edits and bear full responsibility for the final work.