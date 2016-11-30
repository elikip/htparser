# BIST Parsers
## Graph & Transition based dependency parsers using BiLSTM feature extractors.

The techniques behind the parser are described in the paper [Easy-First Dependency Parsing with Hierarchical Tree LSTMs](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/798/208). Further materials could be found [here](http://elki.cc/#/article/Easy-First%20Dependency%20Parsing%20with%20Hierarchical%20Tree%20LSTMs). 

#### Required software

 * Python 2.7 interpreter
 * [PyCNN library](https://github.com/clab/cnn-v1/tree/master/pycnn)

#### Train a parsing model

The software requires having a `training.conll` and `development.conll` files formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat).

To train a parsing model with for either parsing architecture type the following at the command prompt:

    python src/parser.py --outdir [results directory] --train training.conll --dev development.conll [--extrn path_to_external_embeddings_file]

We use the same external embedding used in [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://arxiv.org/abs/1505.08075) which can be downloaded from the authors [github repository](https://github.com/clab/lstm-parser/) and [directly here](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).

Note 1: The reported test result is the one matching the highest development score.

Note 2: The parser calculates (after each iteration) the accuracies excluding punctuation symbols by running the `eval.pl` script from the CoNLL-X Shared Task and stores the results in directory specified by the `--outdir`.

Note 3: The external embeddings parameter is optional and could be omitted.

#### Parse data with your parsing model

The command for parsing a `test.conll` file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat) with a previously trained model is:

    python src/parser.py --predict --outdir [results directory] --test test.conll [--extrn extrn.vectors] --model [trained model file] --params [param file generate during training]

The parser will store the resulting conll file in the out directory (`--outdir`).

#### Citation

If you make use of this software for research purposes, we'll appreciate citing the following:

    @article{DBLP:journals/tacl/KiperwasserG16a,
        author    = {Eliyahu Kiperwasser and
                    Yoav Goldberg},
        title     = {Easy-First Dependency Parsing with Hierarchical Tree LSTMs},
        journal   = {{TACL}},
        volume    = {4},
        pages     = {445--461},
        year      = {2016},
        url       = {https://transacl.org/ojs/index.php/tacl/article/view/798},
        timestamp = {Tue, 09 Aug 2016 14:51:09 +0200},
        biburl    = {http://dblp.uni-trier.de/rec/bib/journals/tacl/KiperwasserG16a},
        bibsource = {dblp computer science bibliography, http://dblp.org}
    }

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact elikip@gmail.com

#### Credits

[Eliyahu Kiperwasser](http://elki.cc)

[Yoav Goldberg](https://www.cs.bgu.ac.il/~yoavg/uni/)

