"""
Error Correcting Earley Parser.

Modified from Hardik's Earley Parser implementation by Muhammad Nizami
"""

import argparse
import sys
import string

from collections import defaultdict
from nltk.tree import Tree

class Rule(object):
	"""
	Represents a CFG rule.
	"""

	def __init__(self, lhs, rhs):
		# Represents the rule 'lhs -> rhs', where lhs is a non-terminal and
		# rhs is a list of non-terminals and terminals.
		self.lhs, self.rhs = lhs, rhs

	def __contains__(self, sym):
		return sym in self.rhs

	def __eq__(self, other):
		if type(other) is Rule:
			return self.lhs == other.lhs and self.rhs == other.rhs

		return False

	def __getitem__(self, i):
		return self.rhs[i]

	def __len__(self):
		return len(self.rhs)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return self.lhs + ' -> ' + ' '.join(self.rhs)

	def error_score(self):
		return 0

class ErrorRule(Rule):

	def error_score(self):
		return 1


class Grammar(object):
	"""
	Represents a CFG.
	"""

	def __init__(self):
		# The rules are represented as a dictionary from L.H.S to R.H.S.
		self.rules = defaultdict(list)

	def add(self, rule):
		"""
		Adds the given rule to the grammar.
		"""
		
		self.rules[rule.lhs].append(rule)

	def get_alphabet(self):
		symbols = set([])
		alphabet = set([])
		for key in self.rules:
			for rule in self.rules[key]:
				for sym in rule.rhs:
					symbols.add(sym)
		for sym in symbols:
			if self.is_terminal(sym):
				if sym not in alphabet:
					alphabet.add(sym)
		return alphabet

	def get_tags(self):
		symbols = set([])
		tags = set([])
		for key in self.rules:
			for rule in self.rules[key]:
				for sym in rule.rhs:
					symbols.add(sym)
		for sym in symbols:
			if self.is_tag(sym):
				if sym not in tags:
					tags.add(sym)
		return tags

	@staticmethod
	def load_grammar(fpath):
		"""
		Loads the grammar from file (from the )
		"""

		grammar = Grammar()
		
		with open(fpath) as f:
			for line in f:
				line = line.strip()

				if len(line) == 0:
					continue

				entries = line.split('->')
				lhs = entries[0].strip()
				for rhs in entries[1].split('|'):
					grammar.add(Rule(lhs, rhs.strip().split()))

		return grammar

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		s = [str(r) for r in self.rules['S']]

		for nt, rule_list in self.rules.iteritems():
			if nt == 'S':
				continue

			s += [str(r) for r in rule_list]

		return '\n'.join(s)

	# Returns the rules for a given Non-terminal.
	def __getitem__(self, nt):
		return self.rules[nt]

	def is_terminal(self, sym):
		"""
		Checks is the given symbol is terminal.
		"""

		return len(self.rules[sym]) == 0

	def is_tag(self, sym):
		"""
		Checks whether the given symbol is a tag, i.e. a non-terminal with rules
		to solely terminals.
		"""

		if not self.is_terminal(sym):
			return all(self.is_terminal(s) for r in self.rules[sym] for s in
				r.rhs)

		return False

	def is_nullable(self, sym):
		"""
		Checks whether the given symbol is nullable, i.e. a non-terminal with rules
		to null.
		"""

		if not self.is_terminal(sym):
			return any(r.rhs==[] for r in self.rules[sym])


class ErrorGrammar(Grammar):
	@staticmethod
	def load_grammar(fpath):
		"""
		Loads the grammar from file (from the )
		"""

		grammar = ErrorGrammar()

		with open(fpath) as f:
			for line in f:
				line = line.strip()

				if len(line) == 0:
					continue

				entries = line.split('->')
				lhs = entries[0].strip()
				for rhs in entries[1].split('|'):
					rhssymbols = rhs.strip().split()
					grammar.add(Rule(lhs, rhssymbols))

		alphabet = grammar.get_alphabet()
		tags = grammar.get_tags()
		#first step in Algorithm 1 in AHO's paper
		added_rules_1 = []
		for key in grammar.rules:
			for rule in grammar.rules[key]:
				new_rule_1 = ErrorGrammar.error_rule_1(grammar,rule)
				if new_rule_1 is not None:
					added_rules_1.append(new_rule_1)
		for rule in added_rules_1:
			grammar.add(rule)

		#second step
		for sym in alphabet.union(tags):
			grammar.add(ErrorRule("E_"+sym,[sym]))
			for sym2 in alphabet:
				if sym2 != sym:
					grammar.add(ErrorRule("E_"+sym,[sym2]))
			grammar.add(ErrorRule("E_"+sym,["H", sym]))
			grammar.add(ErrorRule("I",[sym]))
			grammar.add(ErrorRule("E_"+sym,[])) #empty string

		#third step
		grammar.add(Rule("S'",["S"]))
		grammar.add(Rule("S'",["S","H"]))
		grammar.add(Rule("H",["H","I"]))
		grammar.add(Rule("H",["I"]))
		return grammar

	@staticmethod
	def error_rule_1(grammar,rule):
		if grammar.is_tag(rule.lhs):
			return None
		new_rule_rhs = []
		for sym in rule.rhs:
			if not grammar.is_terminal(sym) and not grammar.is_tag(sym):
				new_rule_rhs.append(sym)
			else:
				new_rule_rhs.append("E_"+sym)
		if new_rule_rhs != rule.rhs:
			return ErrorRule(rule.lhs,new_rule_rhs)
		else:
			return None

class State(object):
	"""
	Represents a state in the error-correcting Earley algorithm.
	"""

	GAM = '<GAM>'

	def __init__(self, rule, dot=0, sent_pos=0, chart_pos=0, error_count=0, back_pointers=[]):
		# CFG Rule.
		self.rule = rule
		# Dot position in the rule.
		self.dot = dot
		# Sentence position.
		self.sent_pos = sent_pos
		# Chart index.
		self.chart_pos = chart_pos
		# Error counter
		self.error_count = error_count
		# Pointers to child states (if the given state was generated using
		# Completer).
		self.back_pointers = back_pointers

	def __eq__(self, other):
		if type(other) is State:
			return self.rule == other.rule and self.dot == other.dot and \
				self.sent_pos == other.sent_pos

		return False

	def __len__(self):
		return len(self.rule)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		def str_helper(state):
			return ('(' + state.rule.lhs + ' -> ' +
			' '.join(state.rule.rhs[:state.dot] + ['*'] + 
				state.rule.rhs[state.dot:]) +
			(', [%d, %d], %d)' % (state.sent_pos, state.chart_pos, state.error_count)))

		return (str_helper(self) +
			' (' + ', '.join(str_helper(s) for s in self.back_pointers) + ')')

	def next(self):
		"""
		Return next symbol to parse, i.e. the one after the dot
		"""

		if self.dot < len(self):
			return self.rule[self.dot]

	def is_complete(self):
		"""
		Checks whether the given state is complete.
		"""

		return len(self) == self.dot

	@staticmethod
	def init():
		"""
		Returns the state used to initialize the chart in the Earley algorithm.
		"""

		return State(Rule(State.GAM, ["S"]))


class ChartEntry(object):
	"""
	Represents an entry in the chart used by the Earley algorithm.
	"""

	def __init__(self, states):
		# List of Earley states.
		self.states = states

	def __iter__(self):
		return iter(self.states)

	def __len__(self):
		return len(self.states)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return '\n'.join(str(s) for s in self.states)

	def add(self, state):
		"""
		Add the given state (if it hasn't already been added).
		"""

		if state not in self.states:
			self.states.append(state)


class Chart(object):
	"""
	Represents the chart used in the Earley algorithm.
	"""

	def __init__(self, entries):
		# List of chart entries.
		self.entries = entries

	def __getitem__(self, i):
		return self.entries[i]

	def __len__(self):
		return len(self.entries)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return '\n\n'.join([("Chart[%d]:\n" % i) + str(entry) for i, entry in
			enumerate(self.entries)])

	@staticmethod
	def init(l):
		"""
		Initializes a chart with l entries (Including the dummy start state).
		"""

		return Chart([(ChartEntry([]) if i > 0 else
				ChartEntry([State.init()])) for i in range(l)])


class ErrorEarleyParse(object):
	"""
	Represents the Error-correcting Earley-generated parse for a given sentence according to a
	given grammar.
	"""

	def __init__(self, sentence, grammar):
		self.words = sentence.split()
		self.grammar = grammar

		self.chart = Chart.init(len(self.words) + 1)

	def predictor(self, state, pos):
		"""
		Error-correcting Earley Predictor.
		"""

		for rule in self.grammar[state.next()]:
			self.chart[pos].add(State(rule, dot=0,
				sent_pos=state.chart_pos, chart_pos=state.chart_pos))

	def scanner(self, state, pos):
		"""
		Error-correcting Earley Scanner.
		"""

		if state.chart_pos < len(self.words):
			word = self.words[state.chart_pos]

			if any((word in r) for r in self.grammar[state.next()]):
				self.chart[pos + 1].add(State(Rule(state.next(), [word]),
					dot=1, sent_pos=state.chart_pos,
					chart_pos=(state.chart_pos + 1),
					error_count = state.error_count))

	def completer(self, state, pos):
		"""
		Error-correcting Earley Completer.
		"""

		for prev_state in self.chart[state.sent_pos]:
			if prev_state.next() == state.rule.lhs:
				self.chart[pos].add(State(prev_state.rule,
					dot=(prev_state.dot + 1), sent_pos=prev_state.sent_pos,
					chart_pos=pos,
					back_pointers=(prev_state.back_pointers + [state]),
					error_count=prev_state.error_count + state.error_count + state.rule.error_score()))

	def parse(self):
		"""
		Parses the sentence by running the Earley algorithm and filling out the
		chart.
		"""

		# Checks whether the next symbol for the given state is a tag.
		def is_tag(state):
			return self.grammar.is_tag(state.next())

		for i in range(len(self.chart)):
			for state in self.chart[i]:
				if not state.is_complete():
					if is_tag(state):
						self.scanner(state, i)
					else:
						self.predictor(state, i)
				else:
					self.completer(state, i)

	def has_parse(self):
		"""
		Checks whether the sentence has a parse.
		"""

		for state in self.chart[-1]:
			if state.is_complete() and state.rule.lhs == 'S' and \
				state.sent_pos == 0 and state.chart_pos == len(self.words):
				return True

		return False

	def get(self):
		"""
		Returns the minimum error parse if it exists, otherwise returns None.
		"""

		def get_helper(state):
			if self.grammar.is_tag(state.rule.lhs):
				return Tree(state.rule.lhs, [state.rule.rhs[0]])

			return Tree(state.rule.lhs,
				[get_helper(s) for s in state.back_pointers])

		found_state = None
		errors = float("inf")
		for state in self.chart[-1]:
			if state.is_complete() and state.rule.lhs == 'S' and \
				state.sent_pos == 0 and state.chart_pos == len(self.words):
				if state.error_count < errors:
					found_state = state
					errors = state.error_count

		if found_state is not None:
			return get_helper(found_state)

		return None


def main():
	"""
	Main.
	"""

	parser_description = ("Runs the Earley parser according to a given "
		"grammar.")
	
	parser = argparse.ArgumentParser(description=parser_description)

	parser.add_argument('draw', nargs='?', default=False)
	parser.add_argument('grammar_file', help="Filepath to grammer file")
	parser.add_argument('--show-chart', action="store_true")
	parser.add_argument('--show-grammar', action="store_true")

	args = parser.parse_args()

	grammar = ErrorGrammar.load_grammar(args.grammar_file)
	if args.show_grammar:
		print grammar

	def run_parse(sentence):
		parse = ErrorEarleyParse(sentence, grammar)
		parse.parse()
		return parse.get(), parse.chart

	while True:
		try:
			sentence = raw_input()

			# Strip the sentence of any puncutation.
			stripped_sentence = sentence
			for p in string.punctuation:
				stripped_sentence = stripped_sentence.replace(p, '')

			parse, chart = run_parse(stripped_sentence)
			if args.show_chart:
				print chart
				print '\n'
			if parse is None:
				print sentence + '\n'
			else:
				if args.draw:
					parse.draw()
				else:
					parse.pretty_print()
		except EOFError:
			sys.exit()

		if args.draw:
			sys.exit()

		
if __name__ == '__main__':
	main()
