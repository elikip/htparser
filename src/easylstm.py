from pycnn import *
from utils import ParseForest, read_conll, write_conll
import utils, time, random
import numpy as np


class EasyFirstLSTM:
    def __init__(self, words, pos, rels, w2i, options):
        random.seed(1)
        self.model = Model()
        self.trainer = AdamTrainer(self.model)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.k = options.window
        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.oracle = options.oracle
        self.layers = options.lstm_layers
        self.wordsCount = words
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.builders = [LSTMBuilder(self.layers, self.ldims, self.ldims, self.model), LSTMBuilder(self.layers, self.ldims, self.ldims, self.model)]

        self.blstmFlag = options.blstmFlag
        if self.blstmFlag:
            self.surfaceBuilders = [LSTMBuilder(self.layers, self.ldims, self.ldims * 0.5, self.model), LSTMBuilder(self.layers, self.ldims, self.ldims * 0.5, self.model)]
        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

	    self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.model.add_lookup_parameters("extrn-lookup", (len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.model["extrn-lookup"].init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

	    print 'Load external embedding. Vector dimensions', self.edim

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.model.add_lookup_parameters("word-lookup", (len(words) + 3, self.wdims))
        self.model.add_lookup_parameters("pos-lookup", (len(pos) + 3, self.pdims))
        self.model.add_lookup_parameters("rels-lookup", (len(rels), self.rdims))

        self.nnvecs = 2

        self.model.add_parameters("word-to-lstm", (self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)))
        self.model.add_parameters("word-to-lstm-bias", (self.ldims))
        self.model.add_parameters("lstm-to-lstm", (self.ldims, self.ldims * self.nnvecs + self.rdims))
        self.model.add_parameters("lstm-to-lstm-bias", (self.ldims))

        self.model.add_parameters("hidden-layer", (self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)))
        self.model.add_parameters("hidden-bias", (self.hidden_units))

        self.model.add_parameters("hidden2-layer", (self.hidden2_units, self.hidden_units))
        self.model.add_parameters("hidden2-bias", (self.hidden2_units))

        self.model.add_parameters("output-layer", (2, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.model.add_parameters("output-bias", (2))

        self.model.add_parameters("rhidden-layer", (self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)))
        self.model.add_parameters("rhidden-bias", (self.hidden_units))

        self.model.add_parameters("rhidden2-layer", (self.hidden2_units, self.hidden_units))
        self.model.add_parameters("rhidden2-bias", (self.hidden2_units))

        self.model.add_parameters("routput-layer", (2 * (len(self.irels) + 0), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.model.add_parameters("routput-bias", (2 * (len(self.irels) + 0)))


    def  __getExpr(self, forest, i, train):
        roots = forest.roots
        nRoots = len(roots)

        if self.builders is None:
            input = concatenate([ concatenate(roots[j].lstms) if j>=0 and j<nRoots else self.empty for j in xrange(i-self.k, i+self.k+2) ])
        else:
            input = concatenate([ concatenate([roots[j].lstms[0].output(), roots[j].lstms[1].output()])
                                  if j>=0 and j<nRoots else self.empty for j in xrange(i-self.k, i+self.k+2) ])

        if self.hidden2_units > 0:
            routput = (self.routLayer * self.activation(self.rhid2Bias + self.rhid2Layer * self.activation(self.rhidLayer * input + self.rhidBias)) + self.routBias)
        else:
            routput = (self.routLayer * self.activation(self.rhidLayer * input + self.rhidBias) + self.routBias)

        if self.hidden2_units > 0:
            output = (self.outLayer * self.activation(self.hid2Bias + self.hid2Layer * self.activation(self.hidLayer * input + self.hidBias)) + self.outBias)
        else:
            output = (self.outLayer * self.activation(self.hidLayer * input + self.hidBias) + self.outBias)

        return routput, output


    def __evaluate(self, forest, train):
        nRoots = len(forest.roots)
        nRels = len(self.irels)
        for i in xrange(nRoots - 1):
            if forest.roots[i].scores is None:
                output, uoutput = self.__getExpr(forest, i, train)
                scrs = output.value()
                uscrs = uoutput.value()
                forest.roots[i].exprs = [(pick(output, j * 2) + pick(uoutput, 0), pick(output, j * 2 + 1) + pick(uoutput, 1)) for j in xrange(len(self.irels))]
                forest.roots[i].scores = [(scrs[j * 2] + uscrs[0], scrs[j * 2 + 1] + uscrs[1]) for j in xrange(len(self.irels))]


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)


    def Init(self):
        self.word2lstm = parameter(self.model["word-to-lstm"])
        self.lstm2lstm = parameter(self.model["lstm-to-lstm"])

        self.word2lstmbias = parameter(self.model["word-to-lstm-bias"])
        self.lstm2lstmbias = parameter(self.model["lstm-to-lstm-bias"])

        self.hid2Layer = parameter(self.model["hidden2-layer"])
        self.hidLayer = parameter(self.model["hidden-layer"])
        self.outLayer = parameter(self.model["output-layer"])

        self.hid2Bias = parameter(self.model["hidden2-bias"])
        self.hidBias = parameter(self.model["hidden-bias"])
        self.outBias = parameter(self.model["output-bias"])

        self.rhid2Layer = parameter(self.model["rhidden2-layer"])
        self.rhidLayer = parameter(self.model["rhidden-layer"])
        self.routLayer = parameter(self.model["routput-layer"])

        self.rhid2Bias = parameter(self.model["rhidden2-bias"])
        self.rhidBias = parameter(self.model["rhidden-bias"])
        self.routBias = parameter(self.model["routput-bias"])

        evec = lookup(self.model["extrn-lookup"], 1) if self.external_embedding is not None else None
        paddingWordVec = lookup(self.model["word-lookup"], 1)
        paddingPosVec = lookup(self.model["pos-lookup"], 1) if self.pdims > 0 else None

        paddingVec = tanh(self.word2lstm * concatenate(filter(None, [paddingWordVec, paddingPosVec, evec])) + self.word2lstmbias )
	self.empty = (concatenate([self.builders[0].initial_state().add_input(paddingVec).output(), self.builders[1].initial_state().add_input(paddingVec).output()]))


    def getWordEmbeddings(self, forest, train):
        for root in forest.roots:
            c = float(self.wordsCount.get(root.norm, 0))
            root.wordvec = lookup(self.model["word-lookup"], int(self.vocab.get(root.norm, 0)) if not train or (random.random() < (c/(0.25+c))) else 0)
            root.posvec = lookup(self.model["pos-lookup"], int(self.pos[root.pos])) if self.pdims > 0 else None

            if self.external_embedding is not None:
                if root.form in self.external_embedding:
                    root.evec = lookup(self.model["extrn-lookup"], self.extrnd[root.form] )
                elif root.norm in self.external_embedding:
                    root.evec = lookup(self.model["extrn-lookup"], self.extrnd[root.norm] )
                else:
                    root.evec = lookup(self.model["extrn-lookup"], 0)
            else:
                root.evec = None

            root.ivec = (self.word2lstm * concatenate(filter(None, [root.wordvec, root.posvec, root.evec]))) + self.word2lstmbias

        if self.blstmFlag:
            forward  = self.surfaceBuilders[0].initial_state()
            backward = self.surfaceBuilders[1].initial_state()

            for froot, rroot in zip(forest.roots, reversed(forest.roots)):
                forward = forward.add_input( froot.ivec )
                backward = backward.add_input( rroot.ivec )
                froot.fvec = forward.output()
                rroot.bvec = backward.output()
            for root in forest.roots:
                root.vec = concatenate( [root.fvec, root.bvec] )
        else:
            for root in forest.roots:
                root.vec = tanh( root.ivec )


    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, False)):
                self.Init()
                forest = ParseForest(sentence)
                self.getWordEmbeddings(forest, False)

                for root in forest.roots:
                    root.lstms = [self.builders[0].initial_state().add_input(root.vec),
                                  self.builders[1].initial_state().add_input(root.vec)]

                while len(forest.roots) > 1:

                    self.__evaluate(forest, False)
                    bestParent, bestChild, bestScore = None, None, float("-inf")
                    bestIndex, bestOp = None, None
                    roots = forest.roots

                    for i in xrange(len(forest.roots) - 1):
                        for irel, rel in enumerate(self.irels):
                            for op in xrange(2):
                                if bestScore < roots[i].scores[irel][op] and (i + (1 - op)) > 0:
                                    bestParent, bestChild = i + op, i + (1 - op)
                                    bestScore = roots[i].scores[irel][op]
                                    bestIndex, bestOp = i, op
                                    bestRelation, bestIRelation = rel, irel

                    for j in xrange(max(0, bestIndex - self.k - 1), min(len(forest.roots), bestIndex + self.k + 2)):
                        roots[j].scores = None

                    roots[bestChild].pred_parent_id = forest.roots[bestParent].id
                    roots[bestChild].pred_relation = bestRelation

                    roots[bestParent].lstms[bestOp] = roots[bestParent].lstms[bestOp].add_input((self.activation(self.lstm2lstmbias + self.lstm2lstm *
                        	concatenate([roots[bestChild].lstms[0].output(), lookup(self.model["rels-lookup"], bestIRelation), roots[bestChild].lstms[1].output()]))))

                    forest.Attach(bestParent, bestChild)

                renew_cg()
                yield sentence


    def Train(self, conll_path):
        mloss = 0.0
        errors = 0
        batch = 0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        ltotal = 0

        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, True))
            random.shuffle(shuffledData)

            errs = []
            eeloss = 0.0

            self.Init()

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                forest = ParseForest(sentence)
                self.getWordEmbeddings(forest, True)

                for root in forest.roots:
                    root.lstms = [self.builders[0].initial_state().add_input(root.vec),
                        	  self.builders[1].initial_state().add_input(root.vec)]

                unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

                while len(forest.roots) > 1:
                    self.__evaluate(forest, True)
                    bestValidOp, bestValidScore = None, float("-inf")
                    bestWrongOp, bestWrongScore = None, float("-inf")

                    bestValidParent, bestValidChild = None, None
                    bestValidIndex, bestWrongIndex = None, None
                    roots = forest.roots

                    rootsIds = set([root.id for root in roots])

                    for i in xrange(len(forest.roots) - 1):
                        for irel, rel in enumerate(self.irels):
                            for op in xrange(2):
                                child = i + (1 - op)
                                parent = i + op

                                oracleCost = unassigned[roots[child].id] + (0 if roots[child].parent_id not in rootsIds or roots[child].parent_id  == roots[parent].id else 1)

                                if oracleCost == 0 and (roots[child].parent_id != roots[parent].id or roots[child].relation == rel):
                                    if bestValidScore < forest.roots[i].scores[irel][op]:
                                        bestValidScore = forest.roots[i].scores[irel][op]
                                        bestValidOp = op
                                        bestValidParent, bestValidChild = parent, child
                                        bestValidIndex = i
                                        bestValidIRel, bestValidRel = irel, rel
                                        bestValidExpr = roots[bestValidIndex].exprs[bestValidIRel][bestValidOp]
                                elif bestWrongScore < forest.roots[i].scores[irel][op]:
                                    bestWrongScore = forest.roots[i].scores[irel][op]
                                    bestWrongParent, bestWrongChild = parent, child
                                    bestWrongOp = op
                                    bestWrongIndex = i
                                    bestWrongIRel, bestWrongRel = irel, rel
                                    bestWrongExpr = roots[bestWrongIndex].exprs[bestWrongIRel][bestWrongOp]

                    if bestValidScore < bestWrongScore + 1.0:
                        loss = bestWrongExpr - bestValidExpr
                        mloss += 1.0 + bestWrongScore - bestValidScore
                        eloss += 1.0 + bestWrongScore - bestValidScore
                        errs.append(loss)

                    if not self.oracle or bestValidScore - bestWrongScore > 1.0 or (bestValidScore > bestWrongScore and random.random() > 0.1): 
                        selectedOp = bestValidOp
                        selectedParent = bestValidParent
                        selectedChild = bestValidChild
                        selectedIndex = bestValidIndex
                        selectedIRel, selectedRel = bestValidIRel, bestValidRel
                    else:
                        selectedOp = bestWrongOp
                        selectedParent = bestWrongParent
                        selectedChild = bestWrongChild
                        selectedIndex = bestWrongIndex
                        selectedIRel, selectedRel = bestWrongIRel, bestWrongRel

                    if roots[selectedChild].parent_id  != roots[selectedParent].id or selectedRel != roots[selectedChild].relation:
                        lerrors += 1
                        if roots[selectedChild].parent_id  != roots[selectedParent].id:
                            errors += 1
                            eerrors += 1

                    etotal += 1

                    for j in xrange(max(0, selectedIndex - self.k - 1), min(len(forest.roots), selectedIndex + self.k + 2)):
                        roots[j].scores = None

                    unassigned[roots[selectedChild].parent_id] -= 1

                    roots[selectedParent].lstms[selectedOp] = roots[selectedParent].lstms[selectedOp].add_input(
                        	    self.activation( self.lstm2lstm *
                                	noise(concatenate([roots[selectedChild].lstms[0].output(), lookup(self.model["rels-lookup"], selectedIRel),
                                        	           roots[selectedChild].lstms[1].output()]), 0.0) + self.lstm2lstmbias))

                    forest.Attach(selectedParent, selectedChild)

                if len(errs) > 50.0:
                    eerrs = ((esum(errs)) * (1.0/(float(len(errs)))))
                    scalar_loss = eerrs.scalar_value()
                    eerrs.backward()
                    self.trainer.update()
                    errs = []
                    lerrs = []

                    renew_cg()
                    self.Init()

        if len(errs) > 0:
            eerrs = (esum(errs)) * (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []

            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss/iSentence
