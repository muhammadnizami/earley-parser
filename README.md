# error-correcting-earley-parser
Partial implementation of Aho and Peterson's "A Minimum-Distance Error-Correcting Parser for Context-Free Languages", SIAM J. COMpUa’. Vol. 1, No. 4, December 1972

The algorithm is only partially implemented for demo purposes. This implementation hasn't covered all cases mentioned in the paper. This implementation also has not been fully verified and tested yet.

This implementation is built by extending Hardik Vala's implementation.

### Usage

Example usage:

```
python errorcorrectingearleyparser.py sample-grammar.txt < sample-sentence-insertion-error.txt
```

More generally, you can run the parser as follow,

```
python errorcorrectingearleyparser.py <grammar_file>
```

which reads sentences from standard in, one at a time, printing the parses to standard output using pretty_print(), with parses separated by an extra newline. For sentences that do not have a parse according to the grammar, it prints the sentence back out, unchanged.

Running with the `draw` option, like so,

```
python errorcorrectingearleyparser.py draw <grammar_file>
```

displays the parses using NLTK's tree-drawing.

See `sample-grammar.txt` on how to format your grammar. The parser program requires each rule to only produce non-terminal or terminal symbols, not both. The symbols `<GAM>`, `I`, `H`, `S'`, and `E_*` are reserved for internal processing. Defining such symbols in the grammar file will cause undefined behavior.

To show the charts, you can add the argument `--show-chart`. To show the extended grammar, you can add the argument `--show-grammar`

### License

MIT
