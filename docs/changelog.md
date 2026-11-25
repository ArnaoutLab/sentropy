# Changelog

Newest first. Version numbers follow the project's release tags.

* Version 1.7.0: implemented batch similarity function
  * 1.7.2
  	* Documentation updates
  	* warning for negative similarity matrix entries
  	* fixed backend mixing in interset_ordinariness
  * 1.7.1: 
    * Remove some code duplication,
    * Incorporate code optimization to handle edge cases
    * Fixed bugs in [issue #13](https://github.com/ArnaoutLab/sentropy/issues/13) and [issue #14](https://github.com/ArnaoutLab/sentropy/issues/14)
* Version 1.6.0: revamped CLI usage
* Version 1.5.0: optimized interset ordinariness to include deduplication, redefined Vendi score of overall set not to be the power mean of subsets' Vendi score
* Version 1.4.0: implemented interset ordinariness, auto-tuning Ray hyperparameters, slight improvement to sparse matrix API.
* Version 1.3.0: sped up the SKL computation.
* Version 1.2.0: implemented the Vendi score.
* Version 1.1.0: implemented the similarity-sensitive crossentropy.
